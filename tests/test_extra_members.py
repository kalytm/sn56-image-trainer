"""D6: adaptive extra members after an epoch-capped member (Trainer.run wiring and train.py validation)."""
import types
from unittest import mock

import pytest

from godtrainer import engine
from godtrainer.engine import Trainer
from test_engine_deadlines import clock  # noqa: F401  (fixture)
from test_engine_selection import recipe_args, run_train_validation


@pytest.fixture(autouse=True)
def stub_soup(monkeypatch):
    monkeypatch.setattr(engine, "soup_tensors", lambda states, device: ("soup", len(states)))


def make(clock, tmp_path, deadline=3000.0, **cfg):  # noqa: F811
    t = object.__new__(Trainer)
    base = dict(seed=0, soup=1, retrain_all=False, crossfit=False, schedule="steps", epoch_cap=5.0,
                extra_members=1, extra_member_min_frac=0.8, save_reserve=90.0, holdout_noises=4)
    base.update(cfg)
    t.cfg = types.SimpleNamespace(**base)
    t.items, t.device, t.out_dir = [], None, str(tmp_path)
    t.summary = {"evals": [], "steps": 0, "members": []}
    t.remaining = lambda: deadline - clock()
    t.lora = mock.MagicMock(scale=1.0)
    t.holdout_items, t.train_items, t.holdout_time, t.last_holdout_unit = [1, 2, 3], [4] * 18, 50.0, [0.5] * 3
    t._crossfit_ok = lambda: False
    t._base_eval = lambda guider, extra, cutoff=None: 0.5
    t._better = lambda a, ta, b, tb: a < b
    t.saves, t.ends, t.soups = [], [], []
    t.save = lambda states, scale, score, tag: t.saves.append((len(states), tag))

    def soup_eval(guider, extra, cands, step, limit=None):
        t.soups.append(cands[0][0])
        return [(0.35, cands[0][0])]
    t.evaluate_candidates = soup_eval
    return t


def members(t, clock, walls, scores=None):  # noqa: F811
    """Each member runs for walls[mi] seconds (capped at its budget) and returns scores[mi]."""
    scores = scores or [0.40] * len(walls)

    def member(guider, extra, mi, budget_end):
        t.ends.append(budget_end)
        clock.t = min(budget_end, clock.t + walls[mi])
        return (scores[mi], f"m{mi}:ema@100", {"m": mi})
    t.train_member = member


def test_epoch_capped_member_frees_time_for_a_second_member_and_a_soup(clock, tmp_path):  # noqa: F811
    t = make(clock, tmp_path)
    members(t, clock, [1000.0, 1000.0])
    t.run(None, None)
    # member 0 gets the whole budget (no soup reserve: a second member is not yet known)
    assert t.ends[0] == pytest.approx(3000.0 - 90.0)
    # member 1 gets everything left after the save and soup reserves
    assert t.ends[1] == pytest.approx(3000.0 - 90.0 - (50.0 * 1.3 + 15))
    assert [m["member"] for m in t.summary["members"]] == [0, 1]
    assert t.summary["extra_members"][0]["added"] is True
    assert t.soups and t.saves[-1] == (2, t.soups[0])  # the soup won (0.35 < 0.40)
    assert t.lora.reinit.call_args_list == [mock.call(1000)]


def test_member_that_used_most_of_the_time_is_not_followed(clock, tmp_path):  # noqa: F811
    t = make(clock, tmp_path)
    members(t, clock, [2000.0, 2000.0])
    t.run(None, None)
    assert len(t.summary["members"]) == 1 and t.summary["extra_members"][0]["added"] is False
    assert not t.soups and not t.lora.reinit.called


@pytest.mark.parametrize("over", [dict(extra_members=0), dict(epoch_cap=0.0), dict(schedule="time")])
def test_disabled_without_an_epoch_capped_step_schedule(clock, tmp_path, over):  # noqa: F811
    t = make(clock, tmp_path, **over)
    members(t, clock, [500.0, 500.0])
    t.run(None, None)
    assert len(t.summary["members"]) == 1 and "extra_members" not in t.summary


def test_member_count_is_bounded(clock, tmp_path):  # noqa: F811
    t = make(clock, tmp_path, deadline=10_000.0, extra_members=2)
    members(t, clock, [500.0] * 5)
    t.run(None, None)
    assert len(t.summary["members"]) == 3


def test_fixed_soup_keeps_the_even_split(clock, tmp_path):  # noqa: F811
    t = make(clock, tmp_path, soup=2, extra_members=0)
    members(t, clock, [5000.0, 5000.0])
    t.run(None, None)
    usable = 3000.0 - 90.0 - (50.0 * 1.3 + 15)
    assert t.ends == [pytest.approx(usable / 2), pytest.approx(usable)]


def test_failed_member_stops_the_loop(clock, tmp_path):  # noqa: F811
    t = make(clock, tmp_path)
    calls = []

    def member(guider, extra, mi, budget_end):
        calls.append(mi)
        clock.t += 100.0
        return None
    t.train_member = member
    t.run(None, None)
    assert calls == [0] and t.summary["final"]["best"] == "identity"


@pytest.mark.parametrize("args,msg", [
    (["--extra-members", "1", "--schedule", "steps"], "--extra-members needs"),
    (["--extra-members", "1", "--schedule", "time", "--epoch-cap", "5"], "--extra-members needs"),
    (["--extra-members", "1", "--schedule", "steps", "--epoch-cap", "5", "--soup", "2"], "--extra-members needs"),
    (["--extra-members", "-1"], "--extra-members must be"),
    (["--extra-member-min-frac", "0"], "--extra-members must be"),
])
def test_train_rejects_invalid_extra_member_settings(tmp_path, args, msg):
    r = run_train_validation(args, tmp_path)
    assert r.returncode != 0 and msg in r.stderr


@pytest.mark.parametrize("family", ["krea2", "ideogram4", "z-image", "flux"])  # qwen-image: D9, no cap
def test_shipped_recipes_enable_d6_with_valid_settings(family, tmp_path):
    args = recipe_args(family)
    last = lambda name: [args[i + 1] for i, a in enumerate(args) if a == name][-1]  # argparse: last wins
    assert last("--epoch-cap") == "5" and last("--extra-members") == "1"
    r = run_train_validation(args, tmp_path)
    assert "No module named 'torch'" in r.stderr
