"""D2 cross-fit soup: folds, the fixed apply rule, and the member/soup wiring of Trainer._run_crossfit."""
import math
import types
from unittest import mock

import pytest

from godtrainer.engine import Trainer
from test_engine_deadlines import BATCHES, clock, member_trainer  # noqa: F401  (clock is a fixture)


def items(n, aug=0):
    out = [{"sha256": f"h{i}", "caption": "c", "scaled": mock.MagicMock(), "aug": False} for i in range(n)]
    out += [{"sha256": f"a{i}", "caption": "c", "scaled": mock.MagicMock(), "aug": True} for i in range(aug)]
    return out


def make(n=12, aug=0, remaining=3000.0, **cfg):
    t = object.__new__(Trainer)
    base = dict(seed=0, crossfit=True, crossfit_min_images=8, crossfit_min_member_s=900.0, crossfit_eval_max=4,
                save_reserve=90.0, holdout_noises=4)
    base.update(cfg)
    t.cfg = types.SimpleNamespace(**base)
    t.items = items(n, aug)
    t.summary = {"evals": [], "steps": 0}
    t.remaining = lambda: remaining
    return t


def test_folds_are_disjoint_balanced_deterministic_and_skip_augmented():
    t = make(n=11, aug=5)
    a, b = t.crossfit_folds()
    ids_a, ids_b = {i["sha256"] for i in a}, {i["sha256"] for i in b}
    assert not ids_a & ids_b and len(ids_a | ids_b) == 11 and abs(len(a) - len(b)) <= 1
    assert all(not i["aug"] for i in a + b)
    assert t.crossfit_folds() == (a, b)


@pytest.mark.parametrize("n,remaining,expected", [(12, 3000.0, True), (7, 3000.0, False), (12, 1500.0, False)])
def test_fixed_apply_rule(n, remaining, expected):
    t = make(n=n, remaining=remaining)
    assert t._crossfit_ok() is expected
    assert t.summary["crossfit_rule"]["applied"] is expected


def test_disabled_by_default():
    t = make(crossfit=False)
    assert t._crossfit_ok() is False


def run_with_members(t, tmp_path, member_tags):
    """Stub the expensive parts; record what each member saw."""
    t.out_dir = str(tmp_path)
    t.vectors = {}
    t.lora = mock.MagicMock(scale=1.0)
    t.last_holdout_unit = [0.5] * 4
    saves, seen = [], []
    t.save = lambda states, scale, score, tag: saves.append((len(states), tag))
    t._base_eval = lambda guider, extra, cutoff=None: 0.5

    def member(guider, extra, mi, budget_end):
        seen.append({"train": {i["sha256"] for i in t.train_items}, "hold": {i["sha256"] for i in t.holdout_items},
                     "base_tag": t.base_tag, "global": t._global_select()})
        tag = member_tags[mi]
        return None if tag is None else (0.4, tag, {"m": mi})
    t.train_member = member
    t._run_crossfit(None, None, 0.0)
    return saves, seen


def test_members_train_and_select_on_complementary_folds_and_ship_the_soup(tmp_path):
    t = make(n=12, crossfit_eval_max=4)
    saves, seen = run_with_members(t, tmp_path, ["m0:ema@300", "m1:deltaavg2@400"])
    a, b = t.crossfit_folds()
    ids_a, ids_b = {i["sha256"] for i in a}, {i["sha256"] for i in b}
    assert seen[0]["train"] == ids_a and seen[0]["hold"] <= ids_b and len(seen[0]["hold"]) == 4
    assert seen[1]["train"] == ids_b and seen[1]["hold"] <= ids_a
    assert [s["base_tag"] for s in seen] == ["m0:base", "m1:base"]
    assert not any(s["global"] for s in seen)
    assert saves == [(1, "crossfit-m0:m0:ema@300"), (2, "crossfit-soup:m0:ema@300+m1:deltaavg2@400")]
    assert t.summary["final"]["best"].startswith("crossfit-soup:")
    assert all(d["best_oof_biased"] == 0.4 for d in t.summary["crossfit"])


def test_member_without_gain_is_not_averaged_in(tmp_path):
    t = make()
    saves, _ = run_with_members(t, tmp_path, ["m0:ema@300", "m1:base"])
    assert saves == [(1, "crossfit-m0:m0:ema@300")]
    assert t.summary["final"]["best"] == "crossfit-single:m0:ema@300"
    t = make()
    saves, _ = run_with_members(t, tmp_path, ["m0:base", "m1:raw@500"])
    assert saves == [(1, "crossfit-single:m1:raw@500")]


def test_no_member_gain_keeps_identity(tmp_path):
    t = make()
    saves, _ = run_with_members(t, tmp_path, ["m0:base", "m1:base"])
    assert saves == [] and t.summary["final"]["best"] == "identity"


def test_member_budgets_split_the_remaining_time(tmp_path, clock):  # noqa: F811
    t = make()
    t.remaining = lambda: 3090.0 - clock()
    ends = []
    t.out_dir, t.vectors, t.last_holdout_unit = str(tmp_path), {}, [0.5]
    t.lora = mock.MagicMock(scale=1.0)
    t.save = lambda *a: None
    t._base_eval = lambda guider, extra, cutoff=None: 0.5

    def member(guider, extra, mi, budget_end):
        ends.append(budget_end)
        clock.t = budget_end
        return (0.4, f"m{mi}:ema@1", {})
    t.train_member = member
    t._run_crossfit(None, None, 0.0)
    assert ends == [pytest.approx(1500.0), pytest.approx(3000.0)]


def test_train_member_never_touches_the_global_best_in_crossfit_mode(clock, monkeypatch):  # noqa: F811
    t = member_trainer(clock, monkeypatch, prep_s=0.0, opt_s=1.0, eval_cost=lambda i: 5.0 / BATCHES)
    saved = []
    t.save = lambda *a: saved.append(a)
    t.global_select = False
    t.best = None  # any global comparison would fail
    t.holdout_time = 5.0
    t.cfg.delta_average = 2
    res = t.train_member(None, None, 0, budget_end=400.0)
    assert saved == [] and res is not None and math.isfinite(res[0])
