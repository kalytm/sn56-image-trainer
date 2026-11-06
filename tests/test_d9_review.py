# Adopted verbatim from sn56-codex review image-v1-debe968 (reviews/image-v1-debe968-tests.py).
import importlib.util
import pathlib
import sys

from godtrainer import engine
from test_engine_deadlines import clock, member_trainer, BATCHES
from test_engine_selection import recipe_args


def parse_family(monkeypatch, family):
    source = pathlib.Path(__file__).parents[1] / 'trainer' / 'train.py'
    spec = importlib.util.spec_from_file_location('review_train', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, 'argv', ['train.py', '--family', family, '--model-dir', '/tmp',
                        '--dataset', '/tmp', '--out', '/tmp', '--minutes', '10'] + recipe_args(family))
    return module.parse()


def test_actual_parser_applies_family_overrides(monkeypatch):
    for family in ['krea2', 'ideogram4', 'z-image', 'flux', 'qwen-image']:
        c = parse_family(monkeypatch, family)
        assert c.schedule == 'steps'
        assert (c.epoch_cap, c.extra_members, c.continue_lr_frac) == (
            (0., 0, 0.) if family == 'qwen-image' else (5., 1, .3))
        assert c.soup == 1 and not c.retrain_all and not c.crossfit


def test_real_qwen_loop_is_uncapped_and_single_member(clock, tmp_path, monkeypatch):
    c = parse_family(monkeypatch, 'qwen-image')
    t = member_trainer(clock, monkeypatch, prep_s=0, opt_s=1,
                       eval_cost=lambda i: 2/BATCHES, deadline=600)
    for k in ['schedule','epoch_cap','extra_members','continue_lr_frac','soup','retrain_all',
              'crossfit','eval_count','eval_grid_gamma','delta_average']:
        setattr(t.cfg, k, getattr(c, k))
    t.train_items = t.train_items[:1]  # 16 cases: five epochs would stop at 80 steps
    t.items = []
    t.out_dir = str(tmp_path)
    t.summary['members'] = []
    t.lora.scale = 1.
    t.remaining = lambda: t.cfg.deadline_ts-clock()
    monkeypatch.setattr(engine, 'soup_tensors', lambda states, dev: {})
    t.run(None, None)
    assert len(t.summary['members']) == len(t.summary['schedules']) == 1
    assert t.summary['steps'] > 80
    assert not t.summary['members'][0]['continuation']
    assert 'continuation' not in t.summary and 'extra_members' not in t.summary
    assert not t.lora.reinit.called and not t.lora.load_state.called
    assert clock() <= 510  # save reserve preserved
