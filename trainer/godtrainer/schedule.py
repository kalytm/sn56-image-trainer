"""Step-based training schedule (image-v1, D3).

The champion's cosine spans the member's wall-clock budget and evaluates every fixed N steps, so
the number of epochs depends on the GPU and task length, and evaluation cost is reserved with a
fixed multiplier. Here, after a short calibration, the cosine horizon is set in optimizer steps:

    horizon = calib + min(train_seconds / seconds_per_step, epoch_cap * n_cases - calib)

where ``train_seconds`` is what remains after the measured cost of every planned evaluation and the
end-of-member tail (final raw/EMA evaluation and the delta-average evaluation). Evaluations sit on
a grid whose spacing is set by ``gamma`` (>1 packs points toward the end; the shipped recipe uses 1,
uniform, because the dev suite found the best states at 35-50% of a time-bound horizon). The deadline always
wins: optional evaluations are skipped when they cannot finish, and the horizon shrinks (never
grows, so the LR never rises again) when measured progress falls behind the plan.

All functions take the clock as an argument so the control logic can be tested without a GPU.
"""

import math
import statistics


def plan_horizon(start_step, available_s, sec_per_step, n_cases, epoch_cap, eval_cost_s,
                 max_inloop_evals, eval_frac_max=0.35):
    """Return (horizon_steps, n_inloop_evals).

    ``available_s`` excludes the end-of-member tail. In-loop evaluations are reduced until their
    total cost is at most ``eval_frac_max`` of ``available_s`` (never below zero evaluations).
    ``epoch_cap <= 0`` disables the cap (time-bound only).
    """
    if sec_per_step <= 0 or not math.isfinite(sec_per_step):
        raise ValueError("sec_per_step must be positive and finite")
    available_s = max(0.0, available_s)
    n = max(0, int(max_inloop_evals))
    while n > 0 and n * eval_cost_s > eval_frac_max * available_s:
        n -= 1
    time_steps = int((available_s - n * eval_cost_s) // sec_per_step)
    horizon = start_step + max(0, time_steps)
    if epoch_cap > 0 and n_cases > 0:
        cap = int(math.ceil(epoch_cap * n_cases))
        if cap < horizon:
            horizon = max(start_step + 1, cap)
            # a capped run needs fewer evaluations only if they no longer fit between the points
            n = min(n, max(0, horizon - start_step - 1))
    return max(start_step + 1, horizon), n


def eval_grid(start_step, horizon, n_points, gamma=2.0):
    """``n_points`` steps in (start_step, horizon], the last equal to ``horizon``; with gamma > 1
    the spacing shrinks toward the end (position = (i / n) ** (1 / gamma))."""
    if n_points <= 0 or horizon <= start_step:
        return []
    span = horizon - start_step
    pts = []
    for i in range(1, n_points + 1):
        s = start_step + int(round(span * (i / n_points) ** (1.0 / gamma)))
        s = min(horizon, max(start_step + 1, s))
        if not pts or s > pts[-1]:
            pts.append(s)
    if pts[-1] != horizon:
        pts.append(horizon)
    return pts


class StepSchedule:
    """Per-member loop control. ``eval_s`` is the measured cost of one holdout evaluation; an
    evaluation point scores ``cands_per_eval`` candidates (raw and EMA)."""

    def __init__(self, budget_end, n_cases, epoch_cap=0.0, max_evals=8, calib_steps=25, gamma=2.0,
                 eval_frac_max=0.35, safety=1.2, cands_per_eval=2, tail_extra_evals=1, fixed_s=5.0):
        self.budget_end = budget_end
        self.n_cases = n_cases
        self.epoch_cap = epoch_cap
        self.max_evals = max_evals
        self.calib_steps = max(1, calib_steps)
        self.gamma = gamma
        self.eval_frac_max = eval_frac_max
        self.safety = safety
        self.cands_per_eval = cands_per_eval
        self.tail_extra_evals = tail_extra_evals
        self.fixed_s = fixed_s
        self.horizon = None
        self.grid = []
        self.planned = None
        self.skipped_evals = 0
        self.replans = 0
        self._dts = []

    # --- measurements -------------------------------------------------------------------
    def observe_step(self, dt):
        self._dts.append(float(dt))

    def reset_timing(self):
        """Drop step samples after an execution-mode change (memory fallback to full checkpointing)."""
        self._dts = []

    def sec_per_step(self, fallback=3.0):
        """Median of the most recent step times (robust to slow first steps and eval stalls)."""
        recent = self._dts[-15:]
        return statistics.median(recent) if recent else fallback

    def point_cost(self, eval_s):
        return self.cands_per_eval * eval_s * self.safety

    def tail_s(self, eval_s):
        """Reserved after the last training step: final raw/EMA point plus extra (delta-average)."""
        return self.point_cost(eval_s) + self.tail_extra_evals * eval_s * self.safety + self.fixed_s

    # --- planning -----------------------------------------------------------------------
    def maybe_plan(self, step, now, eval_s):
        if self.horizon is not None or step < self.calib_steps:
            return False
        sps = self.sec_per_step()
        available = self.budget_end - now - self.tail_s(eval_s)
        # the planner counts in-loop points; the final point at the horizon is in the tail
        self.horizon, n_inloop = plan_horizon(step, available, sps, self.n_cases, self.epoch_cap,
                                              self.point_cost(eval_s), self.max_evals - 1,
                                              self.eval_frac_max)
        self.grid = eval_grid(step, self.horizon, n_inloop + 1, self.gamma)[:-1]
        self.planned = {"start": step, "horizon": self.horizon, "inloop_evals": len(self.grid),
                        "sec_per_step": sps, "eval_s": eval_s, "available_s": available,
                        "epochs": (self.horizon / self.n_cases) if self.n_cases else None}
        return True

    def progress(self, step):
        """Cosine position in [0, 1]; 0 during calibration."""
        if self.horizon is None:
            return 0.0
        return min(1.0, step / max(1, self.horizon))

    def should_stop(self, step, now, eval_s):
        if self.horizon is not None and step >= self.horizon:
            return True
        return now + self.sec_per_step() + self.tail_s(eval_s) > self.budget_end

    def eval_due(self, step, now, eval_s):
        """True when a grid point has been reached and its evaluation fits before the deadline."""
        due = False
        while self.grid and step >= self.grid[0]:
            self.grid.pop(0)
            due = True
        if not due:
            return False
        if now + self.point_cost(eval_s) + self.sec_per_step() + self.tail_s(eval_s) > self.budget_end:
            self.skipped_evals += 1
            return False
        return True

    def replan(self, step, now, eval_s):
        """Shrink the horizon (never grow it) if the remaining time cannot reach it."""
        if self.horizon is None or step >= self.horizon:
            return False
        usable = self.budget_end - now - self.tail_s(eval_s) - len(self.grid) * self.point_cost(eval_s)
        reachable = step + max(0, int(usable // self.sec_per_step()))
        if reachable >= self.horizon:
            return False
        self.horizon = max(step + 1, reachable)
        self.grid = [g for g in self.grid if g < self.horizon]
        self.replans += 1
        return True

    def summary(self):
        return {"planned": self.planned, "horizon": self.horizon, "skipped_evals": self.skipped_evals,
                "replans": self.replans}
