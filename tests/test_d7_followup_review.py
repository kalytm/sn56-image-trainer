"""D7 follow-up reviewer checks written by sn56-codex in review image-v1-464af3a (adopted unchanged)."""
from unittest import mock

from godtrainer import engine
from test_d7_review import setup_member
from test_engine_deadlines import clock
from test_memory_fallback import FakeOOM


def test_retry_timing_keeps_iteration_preparation(clock, monkeypatch):
    t = setup_member(clock, monkeypatch)
    monkeypatch.setattr(engine.torch, "randn",
                        lambda *a, **k: (clock.adv(10), mock.MagicMock())[1])
    calls = []

    def loss(*args):
        calls.append(clock())
        if len(calls) == 1:
            clock.adv(1)
            raise FakeOOM()
        clock.adv(2)
        return mock.MagicMock()

    t.loss = loss
    t.train_member(None, None, 0, budget_end=21)
    assert t.summary["memory_fallback"]["retried"]
    assert clock() <= 21, (clock(), calls, t.summary["schedules"])


def test_mock_cuda_attributes_restore_across_patches(monkeypatch):
    attrs = [(engine.torch, "enable_grad"), (engine.torch.cuda, "OutOfMemoryError"),
             (engine.torch.cuda, "is_available"), (engine.torch.cuda, "empty_cache")]
    before = [(obj, name, getattr(obj, name)) for obj, name in attrs]
    with monkeypatch.context() as m:
        for obj, name in attrs:
            m.setattr(obj, name, object())
    for obj, name, original in before:
        assert getattr(obj, name) is original
    assert engine.torch.cuda.is_available() is True
    with engine.torch.enable_grad():
        pass
