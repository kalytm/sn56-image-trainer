"""D7 memory fallback through the real train_member loop: retry admission (codex review
image-v1-432f455 finding 1). Fake clock and device; fixtures from the deadline/review tests."""
from unittest import mock

from test_d7_review import setup_member
from test_engine_deadlines import clock  # noqa: F401
from test_memory_fallback import FakeOOM


def oom_once_at(clock, fail_at, fail_s, ok_s):
    calls = {"n": 0, "starts": []}

    def loss(*args):
        call = calls["n"]
        calls["n"] += 1
        calls["starts"].append(clock())
        if call == fail_at:
            clock.adv(fail_s)
            raise FakeOOM()
        clock.adv(ok_s)
        return mock.MagicMock()
    return loss, calls


def test_retry_refused_when_short_but_not_expired(clock, monkeypatch):
    # paired prompts: two loss calls per step; the second prompt of step 0 runs out of memory at t=5,
    # 1.5 x 4 s + the tail reserve no longer fits a 12 s budget, so the member ends without an update
    t = setup_member(clock, monkeypatch)
    t.cfg.paired_prompts = True
    loss, calls = oom_once_at(clock, fail_at=1, fail_s=2, ok_s=3)
    t.loss = loss
    t.train_member(None, None, 0, budget_end=12.0)
    assert t.summary["memory_fallback"]["retried"] is False
    assert calls["n"] == 2 and t.summary["steps"] == 0 and clock() <= 12.0


def test_fixed_step_retraining_refuses_retry_without_time(clock, monkeypatch):
    t = setup_member(clock, monkeypatch)
    loss, calls = oom_once_at(clock, fail_at=3, fail_s=4, ok_s=1)
    t.loss = loss
    t.train_member(None, None, 0, budget_end=clock() + 9.0, fixed_steps=50)
    fb = t.summary["memory_fallback"]
    assert fb["retried"] is False and fb["step"] == 3 and calls["n"] == 4


def test_fixed_step_retraining_retries_with_time(clock, monkeypatch):
    t = setup_member(clock, monkeypatch)
    loss, calls = oom_once_at(clock, fail_at=3, fail_s=1, ok_s=1)
    t.loss = loss
    t.train_member(None, None, 0, budget_end=clock() + 500.0, fixed_steps=10)
    assert t.summary["memory_fallback"]["retried"] is True and t.summary["steps"] == 10
