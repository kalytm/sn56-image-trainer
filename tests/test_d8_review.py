"""D8 reviewer checks written by sn56-codex in review image-v1-9fd3fe6 (adopted unchanged)."""
import types
from unittest import mock

import numpy as np
import pytest

from godtrainer import engine
from godtrainer.engine import Trainer, Lora, HoldoutAborted
from test_engine_deadlines import clock, member_trainer, BATCHES
from test_extra_members import make
from test_continuation import cfg, stub


class Tensor:
    """Minimal shape-enforcing tensors, not an autograd/device simulation."""
    def __init__(self, a):
        self.a = np.asarray(a)
    def __truediv__(self, n):
        return Tensor(self.a / n)
    def to(self, device):
        return self
    def copy_(self, other):
        np.copyto(self.a, other.a)


def test_final_delta_average_does_not_crash_continuation(clock, tmp_path, monkeypatch):
    t = make(clock, tmp_path, **cfg())
    monkeypatch.setattr(engine.torch, 'cat',
                        lambda xs, dim: Tensor(np.concatenate([x.a for x in xs], axis=dim)))
    raw = {'w': (Tensor(np.zeros((4, 2))), Tensor(np.zeros((2, 4))))}
    averaged = engine.soup_tensors([raw, raw], None)
    assert averaged['w'][0].a.shape == (4, 4)
    t.lora.state.return_value = raw
    t.lora.direct_names = set()
    t.lora.wrappers = {'w': types.SimpleNamespace(up=raw['w'][0], down=raw['w'][1])}
    t.lora.load_state = types.MethodType(Lora.load_state, t.lora)

    def member(g, e, mi, budget_end, lr_scale=1.0):
        clock.adv(2000 if mi == 0 else 100)
        t.summary['steps'] += 100
        return .4, f'm{mi}:deltaavg2@100', averaged

    t.train_member = member
    t.run(None, None)


@pytest.mark.parametrize('vec,expected', [
    ([.299, .5, .399], 'm0:raw@100'),  # small mean gain, large paired SE, earlier local step
    ([.3, .3, .3], 'm1:raw@50'),
    (None, 'm0:raw@100'),
])
def test_continuation_uses_real_d1_and_independent_vectors(clock, tmp_path, vec, expected):
    t = make(clock, tmp_path, **cfg())
    t.cfg.__dict__.update(select_rule='paired', select_k=1., deadline_ts=3000)
    t._better = types.MethodType(Trainer._better, t)
    t.evaluate_candidates = types.MethodType(Trainer.evaluate_candidates, t)
    sequence = iter([[.4]*3, vec])

    def score(*a, **kw):
        v = next(sequence)
        if v is None:
            raise HoldoutAborted(10000, 0, 1)
        t.last_holdout_unit = v
        return sum(v)/len(v)

    t.holdout_score = score

    def member(g, e, mi, budget_end, lr_scale=1.):
        clock.adv(2000 if mi == 0 else 600)
        step = 100 if mi == 0 else 50
        t.summary['steps'] += step
        best, ok = t._select(g, e, t.lora, None, (.5, 'base', {}), step, mi, limit=budget_end)
        return best

    t.train_member = member
    t.run(None, None)
    assert t.best[1] == expected
    assert t.vectors['m0:raw@100'] == [.4]*3
    if vec is None:
        assert 'm1:raw@50' not in t.vectors
    else:
        assert t.vectors['m1:raw@50'] == vec


def test_real_member_applies_lr_scale_to_every_optimizer_step(clock, monkeypatch):
    records = []

    class Optimizer:
        def __init__(self, groups, **kw):
            self.param_groups = groups
        def step(self):
            records.append([g['lr'] for g in self.param_groups])
        def zero_grad(self, **kw):
            pass

    monkeypatch.setattr(engine.torch.optim, 'AdamW', Optimizer)
    def run(scale):
        t = member_trainer(clock, monkeypatch, 0, 1, lambda i: 1/BATCHES)
        t.lora.model_up_params = [mock.MagicMock()]
        t.lora.model_down_params = [mock.MagicMock()]
        t.cfg.lora_plus_ratio = 2
        records.clear()
        t.train_member(None, None, 1, clock()+100, fixed_steps=30, lr_scale=scale)
        assert t.cfg.lr == 1e-4
        return np.array(records)
    ordinary, continuation = run(1.), run(.3)
    assert ordinary.shape == continuation.shape == (30, 2)
    assert np.allclose(continuation, ordinary * .3, rtol=1e-12, atol=0)
    assert np.allclose(continuation[:, 0], 2*continuation[:, 1])


def test_soup_receives_only_independent_members(clock, tmp_path, monkeypatch):
    t = make(clock, tmp_path, deadline=6000, **cfg())
    calls = stub(t, clock, {0: (1500, 1000, .4, None), 1: (1500, 1000, .42, None),
                            2: (800, 400, .39, None)})
    states_seen = []
    monkeypatch.setattr(engine, 'soup_tensors', lambda states, dev: states_seen.append(states) or {})
    t.run(None, None)
    assert [mi for mi, _, _ in calls] == [0, 1, 2]
    assert states_seen == [[{'m': 0}, {'m': 1}]]
