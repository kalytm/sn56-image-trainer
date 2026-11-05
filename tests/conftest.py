import os
import sys
import types
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trainer"))

# engine.py imports torch/safetensors at module level. The CPU tests exercise control flow only, so
# torch is a MagicMock whose no_grad() is a pass-through decorator (the real methods stay callable).
class _NoGrad:
    """Usable as ``@torch.no_grad()`` (pass-through) and as ``with torch.no_grad():``."""

    def __call__(self, f):
        return f

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_torch = mock.MagicMock(name="torch")
_torch.no_grad = _NoGrad
sys.modules.setdefault("torch", _torch)
for name in ("safetensors", "safetensors.torch"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["safetensors.torch"].save_file = lambda *a, **k: None
sys.modules["safetensors.torch"].load_file = lambda *a, **k: {}
