"""Engine-level deadline tests on a fake clock (codex review image-v1-007fffb).

They run the real ``holdout_score``, ``evaluate_candidates``, ``_base_eval``, ``_select`` and
``train_member`` control flow; only tensor work is replaced by stubs that advance the clock.
"""
import math
import random
import types
from unittest import mock

import pytest

from godtrainer import engine
from godtrainer.engine import HoldoutAborted, Trainer

N_HOLD, BANDS, NOISES, BATCH = 3, 16, 4, 2
BATCHES = 2 * N_HOLD * BANDS * (NOISES // BATCH)  # 192 forward batches per holdout evaluation


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def adv(self, dt):
        self.t += dt


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(engine, "time", types.SimpleNamespace(time=c))
    monkeypatch.setattr(engine, "log", lambda *a: None)
    return c


def make_trainer(clock, batch_cost, deadline=10_000.0, **cfg_over):
    """``batch_cost(i)`` = seconds for the i-th holdout forward batch overall."""
    t = object.__new__(Trainer)
    cfg = dict(select_rule="paired", select_k=1.0, select_metric="mean", deadline_ts=deadline, save_reserve=90.0,
               base_eval_frac=0.15, holdout_noises=NOISES, holdout_noise_batch=BATCH)
    cfg.update(cfg_over)
    t.cfg = types.SimpleNamespace(**cfg)
    t.bands = BANDS
    t.device = None
    t.holdout_items = [{"scaled": mock.MagicMock(), "caption": "c", "sha256": f"h{i}"} for i in range(N_HOLD)]
    t.holdout_time = None
    t.base_bands = None
    t.summary = {"evals": [], "steps": 0}
    t.base_score = 0.5
    t.vectors = {"base": [0.5] * N_HOLD}
    t.lora = mock.MagicMock()
    t.batches = 0

    def batch(guider, extra, item, prompt, x, b, s, start, stop):
        clock.adv(batch_cost(t.batches))
        t.batches += 1
        return [0.4] * (stop - start)
    t._holdout_batch = batch
    return t


# --- holdout_score ------------------------------------------------------------------------------

def test_expired_cutoff_runs_no_forward(clock):
    clock.t = 11.0
    t = make_trainer(clock, lambda i: 1.0)
    with pytest.raises(HoldoutAborted):
        t.holdout_score(None, None, limit=10.0)
    assert t.batches == 0


def test_raw_abort_skips_ema_and_runs_no_forward_after_expiry(clock):
    clock.t = 11.0
    t = make_trainer(clock, lambda i: 1.0)
    out = t.evaluate_candidates(None, None, [("m0:raw@9", None, None), ("m0:ema@9", {}, 1.0)], 9, limit=10.0)
    assert t.batches == 0 and all(math.isnan(sc) for sc, _ in out)
    assert t.lora.restore.call_count == 1  # finally still restores after the raw abort


def test_slow_final_batch_result_is_discarded(clock):
    t = make_trainer(clock, lambda i: 30.0 if i == BATCHES - 1 else 0.001)
    with pytest.raises(HoldoutAborted):
        t.holdout_score(None, None, limit=10.0)
    assert t.batches == BATCHES and not hasattr(t, "last_holdout_unit")
    assert t.holdout_time >= 30.0


def test_projection_aborts_after_first_batch_and_feeds_cost_back(clock):
    t = make_trainer(clock, lambda i: 1.0)
    with pytest.raises(HoldoutAborted):
        t.holdout_score(None, None, limit=50.0)
    assert t.batches == 1 and t.holdout_time == pytest.approx(BATCHES)


def test_known_cost_that_cannot_fit_aborts_at_entry(clock):
    t = make_trainer(clock, lambda i: 1.0)
    t.holdout_time = 100.0
    with pytest.raises(HoldoutAborted):
        t.holdout_score(None, None, limit=50.0)
    assert t.batches == 0


def test_holdout_that_fits_returns_mean_and_measures_time(clock):
    t = make_trainer(clock, lambda i: 0.1)
    assert t.holdout_score(None, None, limit=1000.0) == pytest.approx(0.4)
    assert t.holdout_time == pytest.approx(0.1 * BATCHES)


def test_base_eval_retry_uses_scaled_estimate(clock):
    # 4 noises cost 768 s > 15% of 2610 s; retry at 2 noises (384 s) must run, not abort on the stale estimate
    t = make_trainer(clock, lambda i: 4.0, deadline=2700.0)
    t.remaining = lambda: t.cfg.deadline_ts - clock()
    assert t._base_eval(None, None) == pytest.approx(0.4)
    assert t.cfg.holdout_noises == 2 and clock() < 400


# --- train_member ------------------------------------------------------------------------------

def member_trainer(clock, monkeypatch, prep_s, opt_s, eval_cost, deadline=10_000.0):
    t = make_trainer(clock, eval_cost, deadline=deadline)
    t.cfg.__dict__.update(
        lora_plus_ratio=1.0, te_lr_mult=0.3, lr=1e-4, weight_decay=0.0, ema=0.99, schedule="steps", band_max=16,
        band_min=0, epoch_cap=0.0, eval_count=8, calib_steps=25, eval_grid_gamma=2.0, eval_frac_max=0.35,
        eval_safety=1.2, delta_average=1, warmup_steps=20, lr_final_frac=0.1, case_cycle=True, late_frac=0.0,
        late_bands=16, edge_band_count=0, edge_band_repeat=1, optimal_noise_candidates=1,
        multires_noise_iterations=0, paired_prompts=False, balanced_prompts=True, empty_prompt_frac=0.5,
        grad_clip=1.0, seed=0, eval_every=150, band_power=0.0)
    t.lora = mock.MagicMock(model_up_params=[], model_down_params=[], direct_params=[], te_up_params=[],
                            te_down_params=[], params=[])
    t.lora.state.return_value = {}
    t.lora.state_from_flat.return_value = {}
    t.train_items = [{"scaled": mock.MagicMock(), "caption": "c"} for _ in range(20)]
    t.rng = random.Random(0)
    t.step_time = None
    t.best = (0.5, "identity", [{}], 1.0)
    t.save = lambda *a: None
    t.loss = lambda *a: (clock.adv(opt_s), mock.MagicMock())[1]
    stacked = mock.MagicMock()
    stacked.mean.return_value.item.return_value = 0.5
    monkeypatch.setattr(engine.torch, "stack", lambda xs: stacked, raising=False)
    monkeypatch.setattr(engine.torch, "randn", lambda *a, **k: (clock.adv(prep_s), mock.MagicMock())[1], raising=False)
    return t


def test_step_time_includes_preparation_and_the_cosine_reaches_its_horizon(clock, monkeypatch):
    # codex 007fffb P2: 3 s noise preparation + 1 s optimisation per step
    t = member_trainer(clock, monkeypatch, prep_s=3.0, opt_s=1.0, eval_cost=lambda i: 10.0 / BATCHES)
    t.holdout_time = 10.0
    t.train_member(None, None, 0, budget_end=600.0)
    sched = t.summary["schedules"][0]
    assert sched["planned"]["sec_per_step"] == pytest.approx(4.0)
    assert sched["steps"] == sched["horizon"]
    assert clock() <= 600.0


def test_slow_evaluations_do_not_spill_past_the_member_budget(clock, monkeypatch):
    # codex 007fffb P3: measured holdout 10 s, actual evaluations 300 s, member budget 600 s
    t = member_trainer(clock, monkeypatch, prep_s=0.0, opt_s=1.0, eval_cost=lambda i: 300.0 / BATCHES)
    t.holdout_time = 10.0
    t.train_member(None, None, 0, budget_end=600.0)
    assert clock() <= 600.0 + 300.0 / BATCHES  # at most one forward batch past the cutoff
    assert t.summary.get("skipped_evals")
