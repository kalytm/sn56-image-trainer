"""D7 reviewer reproductions written by sn56-codex in review image-v1-432f455 (adopted unchanged)."""
import sys
import weakref
from contextlib import nullcontext
from unittest import mock

import pytest

from godtrainer import engine
from test_engine_deadlines import BATCHES, clock, member_trainer
from test_memory_fallback import FakeOOM, Loss, Param


def setup_member(clock, monkeypatch):
    t = member_trainer(clock, monkeypatch, prep_s=0, opt_s=1,
                       eval_cost=lambda i: 0.1 / BATCHES)
    t.holdout_time = 0.1
    t.cfg.eval_count = 1
    t.diff = object()
    monkeypatch.setattr(engine.torch, "enable_grad", nullcontext, raising=False)
    monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: False, raising=False)
    monkeypatch.setattr(engine.torch.cuda, "OutOfMemoryError", FakeOOM, raising=False)
    monkeypatch.setattr(engine, "enable_checkpointing", lambda *a, **k: 7)
    return t


@pytest.mark.parametrize("fail_at", [0, 34])
def test_no_retry_starts_after_member_deadline(clock, monkeypatch, fail_at):
    t = setup_member(clock, monkeypatch)
    calls = 0
    retry_starts = []

    def loss(*args):
        nonlocal calls
        call = calls
        calls += 1
        if call == fail_at:
            clock.t = 101.0
            raise FakeOOM()
        if call > fail_at:
            retry_starts.append(clock())
        clock.adv(1)
        return mock.MagicMock()

    t.loss = loss
    t.train_member(None, None, 0, budget_end=100.0)
    assert not retry_starts, f"retry began at {retry_starts}, deadline=100"


def test_late_fallback_relearns_d3_cost_before_more_steps(clock, monkeypatch):
    t = setup_member(clock, monkeypatch)
    calls = 0
    starts = []

    def loss(*args):
        nonlocal calls
        call = calls
        calls += 1
        starts.append(clock())
        if call == 34:
            clock.adv(1)
            raise FakeOOM()
        clock.adv(5 if call < 34 else 12)
        return mock.MagicMock()

    t.loss = loss
    t.train_member(None, None, 0, budget_end=206.0)
    assert clock() <= 206.0, (clock(), starts[-5:], t.summary["schedules"])


def test_failed_frame_and_loss_are_released_before_empty_cache(monkeypatch):
    # Checks Python exception/frame retention without pretending to test CUDA allocation.
    t = object.__new__(engine.Trainer)
    t.diff = object()
    t.summary = {}
    t.step_time = 1.0
    p = Param()
    refs = []
    calls = 0

    class Activation:
        pass

    def loss(*args):
        nonlocal calls
        calls += 1
        activation = Activation()
        refs.append(weakref.ref(activation))
        if calls == 2:
            raise FakeOOM()
        out = Loss(p)
        out.activation = activation
        return out

    def empty_cache():
        assert sys.exc_info() == (None, None, None)
        assert p.grad is None
        assert all(ref() is None for ref in refs)

    t.loss = loss
    t.band_weight = lambda b: 1.0
    monkeypatch.setattr(engine.torch.cuda, "OutOfMemoryError", FakeOOM, raising=False)
    monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(engine.torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(engine.torch, "enable_grad", nullcontext, raising=False)
    monkeypatch.setattr(engine, "enable_checkpointing", lambda *a, **k: 7)
    monkeypatch.setattr(engine, "log", lambda *a: None)
    t._forward_backward(None, None, {}, ["a", ""], .5, None, 0,
                        mock.Mock(params=[p]), 0)
    assert p.grad == 2


def test_first_step_fallback_calibrates_checkpointed_cost(clock, monkeypatch):
    t = setup_member(clock, monkeypatch)
    calls = 0

    def loss(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            clock.adv(1)
            raise FakeOOM()
        clock.adv(12)
        return mock.MagicMock()

    t.loss = loss
    t.train_member(None, None, 0, budget_end=1000)
    assert t.summary["schedules"][0]["planned"]["sec_per_step"] == 12
    assert clock() <= 1000
