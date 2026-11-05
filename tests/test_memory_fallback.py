"""Memory fallback (B2 11-05: ideogram4 task ea66240a, 768x1024 images, OOM at step 1 with --no-ckpt).
torch is a MagicMock in the CPU tests (conftest), so OOM is a stand-in exception class and the
checkpoint wrapper is replaced; this checks control flow only."""
import types

import pytest

from godtrainer import engine
from godtrainer.engine import Trainer


class FakeOOM(Exception):
    pass


class Param:
    def __init__(self):
        self.grad = None


class Loss:
    def __init__(self, p):
        self.p = p

    def __mul__(self, w):
        return self

    def backward(self):
        self.p.grad = (self.p.grad or 0) + 1

    def detach(self):
        return "loss"


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(engine.torch.cuda, "OutOfMemoryError", FakeOOM, raising=False)
    wrapped = []
    monkeypatch.setattr(engine, "enable_checkpointing", lambda diff, stride=1: wrapped.append(stride) or 7)
    monkeypatch.setattr(engine, "log", lambda *a, **k: None)
    return wrapped


def trainer(fail_times, ckpt_full=False):
    t = Trainer.__new__(Trainer)
    t.diff = object()
    t.summary = {}
    t.step_time = 1.0
    if ckpt_full:
        t.ckpt_full = True
    p = Param()
    calls = {"n": 0}

    def loss(guider, extra, item, prompt, sigma, noise):
        calls["n"] += 1
        if calls["n"] <= fail_times and prompt == "":  # first prompt's backward already ran
            raise FakeOOM("CUDA out of memory (test)")
        return Loss(p)

    t.loss = loss
    t.band_weight = lambda band: 1.0
    t.calls = calls
    return t, types.SimpleNamespace(params=[p]), p


def test_oom_enables_checkpointing_and_retries(env):
    t, lora, p = trainer(fail_times=2)  # calls 1 ("a") ok, 2 ("") OOM, then retry calls 3 and 4
    losses = t._forward_backward(None, None, {}, ["a", ""], 0.5, None, 3, lora, step=0)
    assert losses == ["loss", "loss"] and t.calls["n"] == 4
    assert p.grad == 2  # the failed attempt's partial gradient was dropped, not accumulated
    assert env == [1] and t.ckpt_full and t.step_time is None
    fb = t.summary["memory_fallback"]
    assert fb["step"] == 0 and fb["checkpointed_blocks"] == 7 and fb["retried"] is True
    assert t.retry_t0 is not None


def test_retry_refused_by_admission_returns_none(env):
    t, lora, p = trainer(fail_times=2)
    seen = []
    out = t._forward_backward(None, None, {}, ["a", ""], 0.5, None, 3, lora, step=9,
                              admit=lambda failed_s: seen.append(failed_s) or False)
    assert out is None and t.calls["n"] == 2 and len(seen) == 1
    assert p.grad is None  # the partial gradient of the failed attempt is not left for an update
    assert env == [1] and t.ckpt_full  # later members still get the memory-safe mode
    assert t.summary["memory_fallback"]["retried"] is False and t.retry_t0 is None


def test_second_oom_after_fallback_raises(env):
    t, lora, _ = trainer(fail_times=99)
    with pytest.raises(FakeOOM):
        t._forward_backward(None, None, {}, [""], 0.5, None, 3, lora, step=0)
    assert t.calls["n"] == 2 and env == [1]


def test_already_fully_checkpointed_raises_without_retry(env):
    t, lora, _ = trainer(fail_times=1, ckpt_full=True)
    with pytest.raises(FakeOOM):
        t._forward_backward(None, None, {}, [""], 0.5, None, 3, lora, step=0)
    assert t.calls["n"] == 1 and env == [] and "memory_fallback" not in t.summary


def test_no_oom_path_unchanged(env):
    t, lora, p = trainer(fail_times=0)
    losses = t._forward_backward(None, None, {}, ["a", ""], 0.5, None, 3, lora, step=5)
    assert len(losses) == 2 and t.calls["n"] == 2 and p.grad == 2
    assert env == [] and "memory_fallback" not in t.summary and t.step_time == 1.0

