"""End-to-end D2 regressions contributed by sn56-codex in review image-v1-62a424a.
Adapted: the member-0 fallback is now saved as crossfit-single:m0:... (same tag as summary.final.best)."""
import math

import pytest

from test_engine_deadlines import BATCHES, clock, member_trainer


def setup_crossfit(clock, monkeypatch, tmp_path):
    t = member_trainer(clock, monkeypatch, prep_s=1.0, opt_s=1.0,
                       eval_cost=lambda i: 10.0 / BATCHES, deadline=2090.0)
    t.cfg.__dict__.update(crossfit=True, crossfit_min_images=8, crossfit_min_member_s=900.0,
                          crossfit_eval_max=4, delta_average=2)
    t.items = t.train_items[:12]
    for i, item in enumerate(t.items):
        item.update(sha256=str(i), aug=False, holdout=False)
    t.out_dir = str(tmp_path)
    t.best = None
    t.lora.scale = 1.0
    saved = []
    t.save = lambda states, scale, score, tag: saved.append((len(states), tag, score, clock()))
    original_better = t._better
    comparisons = []
    def same_fold_only(score, tag, best_score, best_tag):
        assert tag.split(':', 1)[0] == best_tag.split(':', 1)[0]
        comparisons.append((tag, best_tag))
        return original_better(score, tag, best_score, best_tag)
    t._better = same_fold_only
    return t, saved, comparisons


def test_real_two_fold_run_has_local_comparisons_and_ordered_saves(clock, monkeypatch, tmp_path):
    t, saved, comparisons = setup_crossfit(clock, monkeypatch, tmp_path)
    t.run(None, None)
    assert comparisons
    assert {a.split(':', 1)[0] for a, _ in comparisons} == {'m0', 'm1'}
    assert [n for n, _, _, _ in saved] == [1, 1, 2]
    assert saved[0][1] == 'identity'
    assert saved[1][1].startswith('crossfit-single:m0:')
    assert saved[2][1].startswith('crossfit-soup:')
    assert saved[1][3] < 1000.0
    assert clock() <= 2000.0
    assert t.best is None
    assert {'m0:base', 'm1:base'} <= t.vectors.keys()
    assert all(math.isnan(score) for _, _, score, _ in saved)
    assert t.summary['final']['score'] is None
    assert len(t.summary['crossfit']) == 2


@pytest.mark.parametrize('failure', ['base', 'member'])
def test_second_member_failure_keeps_first_fallback(clock, monkeypatch, tmp_path, failure):
    t, saved, _ = setup_crossfit(clock, monkeypatch, tmp_path)
    if failure == 'base':
        original = t._base_eval
        t._base_eval = lambda *args, **kwargs: None if t.base_tag == 'm1:base' else original(*args, **kwargs)
    else:
        original = t.train_member
        t.train_member = lambda guider, extra, mi, end: None if mi == 1 else original(guider, extra, mi, end)
    t.run(None, None)
    assert [n for n, _, _, _ in saved] == [1, 1]
    assert saved[-1][1].startswith('crossfit-single:m0:')
    assert t.summary['final']['best'].startswith('crossfit-single:m0:')
    assert t.summary['final']['score'] is None
    assert clock() <= 2000.0


def test_first_base_failure_keeps_identity(clock, monkeypatch, tmp_path):
    t, saved, _ = setup_crossfit(clock, monkeypatch, tmp_path)
    t._base_eval = lambda *args, **kwargs: None
    t.run(None, None)
    assert [tag for _, tag, _, _ in saved] == ['identity']
    assert t.summary['final']['best'] == 'identity'
