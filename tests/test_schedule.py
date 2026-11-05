import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trainer"))

from godtrainer.schedule import StepSchedule, eval_grid, plan_horizon  # noqa: E402


def test_plan_horizon_time_bound_subtracts_eval_cost():
    horizon, n = plan_horizon(25, available_s=1000.0, sec_per_step=2.0, n_cases=10_000, epoch_cap=0,
                              eval_cost_s=40.0, max_inloop_evals=7)
    assert n == 7  # 280 s <= 35% of 1000 s
    assert horizon == 25 + (1000 - 7 * 40) // 2


def test_plan_horizon_drops_evals_that_would_dominate():
    horizon, n = plan_horizon(25, available_s=300.0, sec_per_step=1.0, n_cases=10_000, epoch_cap=0,
                              eval_cost_s=60.0, max_inloop_evals=7)
    assert n == 1 and horizon == 25 + 240


def test_plan_horizon_epoch_cap_binds():
    horizon, n = plan_horizon(25, available_s=10_000.0, sec_per_step=1.0, n_cases=160, epoch_cap=3.0,
                              eval_cost_s=10.0, max_inloop_evals=7)
    assert horizon == 480 and n == 7


def test_plan_horizon_rejects_bad_step_time():
    with pytest.raises(ValueError):
        plan_horizon(0, 100.0, 0.0, 10, 0, 1.0, 3)


def test_eval_grid_is_increasing_tail_dense_and_ends_at_horizon():
    g = eval_grid(25, 1025, 8, gamma=2.0)
    assert g[-1] == 1025 and g == sorted(set(g)) and len(g) == 8
    gaps = [b - a for a, b in zip(g, g[1:])]
    assert all(x >= y for x, y in zip(gaps, gaps[1:]))  # spacing shrinks toward the end
    assert eval_grid(25, 27, 8) == [26, 27]
    assert eval_grid(25, 25, 8) == []


def simulate(budget_s, step_s, eval_s, n_cases=640, epoch_cap=0.0, slow_after=None, slow_factor=1.0,
             max_evals=8, cands=2, deltaavg=True):
    """Mirror of the engine's control flow on a fake clock. Returns (sched, end_time, evals, steps)."""
    now = 0.0
    s = StepSchedule(budget_s, n_cases, epoch_cap=epoch_cap, max_evals=max_evals, calib_steps=25,
                     cands_per_eval=cands, tail_extra_evals=1 if deltaavg else 0)
    step, evals, last_eval, ps = 0, 0, 0, []
    while True:
        if s.maybe_plan(step, now, eval_s):
            pass
        elif s.horizon is not None and step % s.calib_steps == 0:
            s.replan(step, now, eval_s)
        if s.should_stop(step, now, eval_s):
            break
        ps.append(s.progress(step))
        dt = step_s * (slow_factor if slow_after is not None and step >= slow_after else 1.0)
        now += dt
        step += 1
        s.observe_step(dt)
        if s.eval_due(step, now, eval_s):
            now += cands * eval_s
            evals += 1
            last_eval = step
            s.replan(step, now, eval_s)
    if step > last_eval:  # final raw/EMA evaluation
        now += cands * eval_s
        evals += 1
    if deltaavg:
        now += eval_s
    return s, now, evals, step, ps


@pytest.mark.parametrize("budget,step_s,eval_s", [(2400, 1.5, 30), (2400, 3.0, 90), (600, 2.0, 60), (9000, 4.0, 45)])
def test_simulated_member_never_overruns_and_reaches_its_horizon(budget, step_s, eval_s):
    s, end, evals, steps, ps = simulate(budget, step_s, eval_s)
    assert end <= budget
    assert steps == s.horizon
    assert ps == sorted(ps) and ps[-1] < 1.0
    assert 1 <= evals <= 8


def test_simulated_epoch_cap_finishes_early():
    s, end, evals, steps, _ = simulate(9000, 1.0, 20, n_cases=160, epoch_cap=3.0)
    assert steps == 480 and end < 9000 and evals == 8


def test_simulated_slowdown_shrinks_horizon_and_respects_deadline():
    s, end, evals, steps, ps = simulate(2400, 1.5, 30, slow_after=300, slow_factor=2.5)
    assert end <= 2400
    assert s.replans >= 1 and s.horizon < s.planned["horizon"]
    assert ps == sorted(ps)


def test_evaluation_that_cannot_finish_is_skipped():
    s = StepSchedule(1000.0, 100, max_evals=4, calib_steps=1)
    s.observe_step(1.0)
    assert s.maybe_plan(1, 0.0, 10.0)
    first = s.grid[0]
    # pretend the clock jumped close to the deadline: a due evaluation must be skipped
    assert not s.eval_due(first, 1000.0 - s.tail_s(10.0) - 5.0, 10.0)
    assert s.skipped_evals == 1
