import os
import sys
import types
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trainer"))
# engine.py imports torch/safetensors at module level; the selection wiring does not need them.
sys.modules.setdefault("torch", mock.MagicMock(name="torch"))
for name in ("safetensors", "safetensors.torch"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["safetensors.torch"].save_file = lambda *a, **k: None
sys.modules["safetensors.torch"].load_file = lambda *a, **k: {}

from godtrainer.engine import Trainer  # noqa: E402


def make(rule):
    t = object.__new__(Trainer)
    t.cfg = types.SimpleNamespace(select_rule=rule, select_k=1.0)
    t.vectors = {
        "base": [0.30, 0.40, 0.50, 0.60],
        "ema@600": [0.20, 0.30, 0.40, 0.50],
        "raw@300": [0.199, 0.302, 0.398, 0.500],
    }
    return t


def test_key_normalises_member_prefix_and_identity():
    assert Trainer._key("m1:ema@600") == "ema@600"
    assert Trainer._key("identity") == "base"


def test_min_rule_matches_champion_behaviour():
    t = make("min")
    assert t._better(0.34975, "raw@300", 0.35, "m0:ema@600")


def test_paired_rule_rejects_noise_level_gain_but_accepts_real_gain():
    t = make("paired")
    assert not t._better(0.34975, "raw@300", 0.35, "m0:ema@600")
    assert t._better(0.35, "ema@600", 0.45, "identity")
