import ast
import os
import subprocess
import sys
import time
import types
from unittest import mock

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "trainer"))
# torch/safetensors are stubbed in conftest.py
from godtrainer.engine import HoldoutAborted, Trainer  # noqa: E402


def make(rule, **cfg):
    t = object.__new__(Trainer)
    t.cfg = types.SimpleNamespace(select_rule=rule, select_k=1.0, deadline_ts=time.time() + 2700,
                                  save_reserve=90.0, base_eval_frac=0.15, holdout_noises=4, **cfg)
    t.vectors = {
        "base": [0.30, 0.40, 0.50, 0.60],
        "m0:ema@600": [0.20, 0.30, 0.40, 0.50],
        "m0:raw@300": [0.199, 0.302, 0.398, 0.500],
    }
    t.summary = {"evals": []}
    t.base_score = 0.45
    t.lora = mock.MagicMock()
    return t


def test_key_keeps_member_identity_and_maps_identity():
    assert Trainer._key("m1:ema@600") == "m1:ema@600"
    assert Trainer._key("identity") == "base"


def test_min_rule_matches_champion_behaviour():
    t = make("min")
    assert t._better(0.34975, "m0:raw@300", 0.35, "m0:ema@600")


def test_paired_rule_rejects_noise_level_gain_but_accepts_real_gain():
    t = make("paired")
    assert not t._better(0.34975, "m0:raw@300", 0.35, "m0:ema@600")
    assert t._better(0.35, "m0:ema@600", 0.45, "identity")


def fake_holdout(t, results):
    """Install a holdout_score that returns preset per-image vectors in order."""
    it = iter(results)

    def score(guider, extra, limit=None):
        vec = next(it)
        t.last_holdout_unit = list(vec)
        t.holdout_time = 1.0
        return sum(vec) / len(vec)
    t.holdout_score = score


def test_two_members_with_equal_steps_do_not_share_vectors():
    # codex 4c009d0 P2: member 1's rejected ema@300 must not replace member 0's incumbent vector
    t = make("paired")
    t.vectors = {"base": [0.9, 0.9, 0.9, 0.9]}
    fake_holdout(t, [[0.3, 0.3, 0.3, 0.3], [0.6, 0.6, 0.6, 0.6], [0.5, 0.5, 0.5, 0.5]])
    (s0, tag0), = t.evaluate_candidates(None, None, [("m0:ema@300", None, None)], 300)
    best = (s0, tag0)
    (s1, tag1), = t.evaluate_candidates(None, None, [("m1:ema@300", None, None)], 300)
    assert not t._better(s1, tag1, *best)
    (s2, tag2), = t.evaluate_candidates(None, None, [("m1:raw@600", None, None)], 600)
    assert t.vectors["m0:ema@300"] == [0.3] * 4
    assert not t._better(s2, tag2, *best)


def test_skipped_evaluation_scores_nan_and_never_replaces():
    t = make("paired")

    def abort(guider, extra, limit=None):
        raise HoldoutAborted(5000.0, 1, 6)
    t.holdout_score = abort
    (sc, tag), = t.evaluate_candidates(None, None, [("m0:raw@900", None, None)], 900)
    assert sc != sc and tag == "m0:raw@900"
    assert not t._better(sc, tag, 0.45, "identity")
    t.lora.restore.assert_called()
    assert "m0:raw@900" not in t.vectors


def slow_holdout(t, seconds_per_noise):
    """Holdout whose cost scales with the noise count; aborts like the real one when projected late."""
    calls = []

    def score(guider, extra, limit=None):
        projected = seconds_per_noise * t.cfg.holdout_noises
        calls.append((t.cfg.holdout_noises, limit))
        if limit is not None and time.time() + projected > limit:
            raise HoldoutAborted(projected, 1, 6)
        t.last_holdout_unit = [0.5, 0.5, 0.5]
        t.holdout_time = projected
        return 0.5
    t.holdout_score = score
    t.remaining = lambda: t.cfg.deadline_ts - time.time()
    return calls


def test_slow_first_holdout_reduces_noises_instead_of_overrunning():
    # codex 4c009d0 P1: a base holdout projected at 2600 s must not run to completion in a 45-min task
    t = make("paired")
    calls = slow_holdout(t, 650.0)
    assert t._base_eval(None, None) == 0.5
    assert t.cfg.holdout_noises == 1 and len(calls) == 2
    assert all(limit <= t.eval_limit() for _, limit in calls)


def test_base_eval_that_cannot_finish_even_at_one_noise_returns_none():
    t = make("paired")
    slow_holdout(t, 5000.0)
    assert t._base_eval(None, None) is None


def test_run_saves_identity_before_base_eval_and_stops_if_it_cannot_finish(tmp_path):
    t = make("paired")
    t.items, t.out_dir, t.summary = [], str(tmp_path), {"evals": []}
    order = []
    t.save = lambda states, scale, score, tag: order.append(("save", tag))

    def base(guider, extra):
        order.append(("base_eval", None))
        return None
    t._base_eval = base
    t.remaining = lambda: 100.0
    t.run(None, None)
    assert order == [("save", "identity"), ("base_eval", None)]
    assert t.summary["final"]["reason"] == "base evaluation cannot finish"


def recipe_args(family):
    src = open(os.path.join(ROOT, "scripts", "image_trainer.py")).read()
    tree = ast.parse(src)
    env = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id in ("RECIPES", "COMMON", "FAMILY_EXTRAS"):
            env[node.targets[0].id] = ast.literal_eval(node.value)
    return env["RECIPES"][family] + env["COMMON"] + env["FAMILY_EXTRAS"].get(family, [])


def run_train_validation(args, tmp_path):
    cmd = [sys.executable, os.path.join(ROOT, "trainer", "train.py"), "--family", "ideogram4",
           "--model-dir", str(tmp_path), "--dataset", str(tmp_path), "--out", str(tmp_path / "out"),
           "--minutes", "1"] + args
    # torch is not installed in the CPU test env: reaching the import means validation passed
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("family", ["krea2", "ideogram4", "z-image", "flux", "qwen-image"])
def test_shipped_recipes_use_mean_with_the_paired_rule(family):
    args = recipe_args(family)
    metrics = [args[i + 1] for i, a in enumerate(args) if a == "--select-metric"]
    assert "paired" in args and all(m == "mean" for m in metrics)


def test_train_rejects_paired_rule_with_non_mean_metric(tmp_path):
    r = run_train_validation(["--select-rule", "paired", "--select-metric", "p75"], tmp_path)
    assert r.returncode != 0 and "arithmetic means" in r.stderr
    r = run_train_validation(["--select-rule", "paired", "--select-k", "-1"], tmp_path)
    assert r.returncode != 0 and "--select-k" in r.stderr
    r = run_train_validation(recipe_args("ideogram4"), tmp_path)
    assert "No module named 'torch'" in r.stderr  # validation passed; stopped at the GPU import
