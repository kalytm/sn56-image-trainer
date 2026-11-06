"""D8: one low-LR continuation of the incumbent when no further member fits (Trainer.run wiring)."""
from unittest import mock

import pytest

from test_engine_deadlines import clock  # noqa: F401  (fixture)
from test_engine_selection import run_train_validation
from test_extra_members import make, stub_soup  # noqa: F401  (autouse fixture)


def stub(t, clock, plan):
    """plan[mi] = (wall_s, steps, score, tag_step or None for the member's final step)."""
    calls = []

    def member(guider, extra, mi, budget_end, lr_scale=1.0):
        wall, steps, score, at = plan[mi]
        calls.append((mi, lr_scale, budget_end))
        clock.t = min(budget_end, clock.t + wall)
        t.summary["steps"] += steps
        return (score, f"m{mi}:raw@{steps if at is None else at}", {"m": mi})
    t.train_member = member
    return calls


def cfg(**kw):
    base = dict(continue_lr_frac=0.3, continue_min_frac=0.25)
    base.update(kw)
    return base


def test_capped_member_with_time_left_is_continued(clock, tmp_path):  # noqa: F811
    # B2 qwen shape: member 0 stops at its cap with the best at its final step; a new member does
    # not fit (needs 0.8 x 2000 s) but a continuation does (needs 0.25 x 2000 s)
    t = make(clock, tmp_path, **cfg())
    calls = stub(t, clock, {0: (2000.0, 1680, 0.40, None), 1: (600.0, 500, 0.38, None)})
    t.run(None, None)
    assert t.summary["extra_members"][0]["added"] is False
    c = t.summary["continuation"][0]
    assert c["added"] and c["at_end"] and c["from"] == "m0:raw@1680"
    assert [(mi, s) for mi, s, _ in calls] == [(0, 1.0), (1, 0.3)]
    t.lora.load_state.assert_called_once_with({"m": 0})
    assert not t.lora.reinit.called
    assert t.best[1] == "m1:raw@500" and t.saves[-1] == (1, "m1:raw@500")
    assert not t.soups  # a continuation is never averaged
    assert t.summary["members"][1]["continuation"] is True


def test_losing_continuation_keeps_the_incumbent(clock, tmp_path):  # noqa: F811
    t = make(clock, tmp_path, **cfg())
    stub(t, clock, {0: (2000.0, 1680, 0.40, None), 1: (600.0, 500, 0.41, None)})
    t.run(None, None)
    assert t.best[1] == "m0:raw@1680" and t.saves[-1] == (1, "m0:raw@1680")


def test_best_before_the_final_step_is_not_continued(clock, tmp_path):  # noqa: F811
    t = make(clock, tmp_path, **cfg())
    calls = stub(t, clock, {0: (2000.0, 1680, 0.40, 1404)})
    t.run(None, None)
    assert t.summary["continuation"][0]["at_end"] is False and len(calls) == 1
    assert not t.lora.load_state.called


def test_not_enough_time_is_not_continued(clock, tmp_path):  # noqa: F811
    t = make(clock, tmp_path, **cfg(continue_min_frac=0.5))  # needs 1000 s, has 3000-2000-90-80 = 830 s
    calls = stub(t, clock, {0: (2000.0, 1680, 0.40, None)})
    t.run(None, None)
    c = t.summary["continuation"][0]
    assert c["at_end"] and not c["added"] and c["avail_s"] == pytest.approx(830.0) and len(calls) == 1


def test_disabled_by_default(clock, tmp_path):  # noqa: F811
    t = make(clock, tmp_path, **cfg(continue_lr_frac=0.0))
    calls = stub(t, clock, {0: (2000.0, 1680, 0.40, None)})
    t.run(None, None)
    assert "continuation" not in t.summary and len(calls) == 1


def test_base_incumbent_is_not_continued(clock, tmp_path):  # noqa: F811
    t = make(clock, tmp_path, **cfg())
    calls = stub(t, clock, {0: (2000.0, 1680, 0.60, None)})  # worse than base 0.5
    t.run(None, None)
    assert t.best[1] == "identity" and t.summary["continuation"][0]["added"] is False and len(calls) == 1


def test_after_an_extra_member_only_one_continuation_and_soup_excludes_it(clock, tmp_path):  # noqa: F811
    t = make(clock, tmp_path, deadline=6000.0, **cfg())
    calls = stub(t, clock, {0: (1500.0, 1000, 0.40, None), 1: (1500.0, 1000, 0.42, None),
                            2: (800.0, 400, 0.39, None), 3: (800.0, 400, 0.30, None)})
    t.run(None, None)
    assert [mi for mi, _, _ in calls] == [0, 1, 2] and calls[2][1] == 0.3
    t.lora.load_state.assert_called_once_with({"m": 0})  # continued from the incumbent m0
    assert t.soups and t.soups[0].startswith("soup@")  # soup of members 0 and 1 only
    assert [m["continuation"] for m in t.summary["members"]] == [False, False, True]


@pytest.mark.parametrize("extra,ok", [
    (["--continue-lr-frac", "0.3"], True),
    (["--continue-lr-frac", "1.5"], False),
    (["--continue-lr-frac", "nan"], False),
    (["--continue-lr-frac", "0.3", "--continue-min-frac", "0"], False),
])
def test_validation(tmp_path, extra, ok):
    r = run_train_validation(["--extra-members", "1", "--schedule", "steps", "--epoch-cap", "5"] + extra, tmp_path)
    assert (r.returncode == 0 or "--continue" not in r.stderr) if ok else ("--continue" in r.stderr)


def test_continuation_needs_extra_members(tmp_path):
    r = run_train_validation(["--continue-lr-frac", "0.3"], tmp_path)
    assert r.returncode and "--continue-lr-frac needs --extra-members" in r.stderr
