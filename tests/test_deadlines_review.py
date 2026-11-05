"""Regression tests contributed by sn56-codex in review image-v1-f9982f9 (shipped delta_average=2 path and cutoff precedence)."""
from test_engine_deadlines import BATCHES, clock, member_trainer, make_trainer


def test_shipped_delta_average_path(clock, monkeypatch):
    t = member_trainer(clock, monkeypatch, prep_s=3.0, opt_s=1.0,
                       eval_cost=lambda i: 10.0 / BATCHES)
    t.cfg.delta_average = 2
    t.holdout_time = 10.0
    t.train_member(None, None, 0, budget_end=600.0)
    s = t.summary['schedules'][0]
    assert s['steps'] == s['horizon']
    assert clock() <= 600.0
    assert any(e['tag'].startswith('m0:deltaavg2@') for e in t.summary['evals'])
    assert all(e['tag'] in t.vectors for e in t.summary['evals'])


def test_aborted_slow_path_with_delta_average_enabled(clock, monkeypatch):
    t = member_trainer(clock, monkeypatch, prep_s=0.0, opt_s=1.0,
                       eval_cost=lambda i: 300.0 / BATCHES)
    t.cfg.delta_average = 2
    t.holdout_time = 10.0
    t.train_member(None, None, 0, budget_end=600.0)
    assert clock() <= 600.0
    assert t.summary.get('skipped_evals')
    assert not any(e['tag'].startswith('m0:deltaavg2@') for e in t.summary['evals'])
    assert all(e['tag'] not in t.vectors for e in t.summary['skipped_evals'])


def test_base_retry_does_not_start_past_task_cutoff(clock):
    t = make_trainer(clock, lambda i: 1.0, deadline=100.0)
    clock.t = 11.0
    assert t._base_eval(None, None) is None
    assert t.batches == 0


def test_member_limit_cannot_extend_task_limit(clock):
    t = make_trainer(clock, lambda i: 1.0, deadline=100.0)
    clock.t = 11.0
    t.evaluate_candidates(None, None, [('m0:raw@1', None, None)], 1, limit=1000.0)
    assert t.batches == 0
