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
# Real attributes (not auto-created children) so monkeypatching them restores cleanly between tests.
_torch.enable_grad = _NoGrad


class _OutOfMemoryError(RuntimeError):
    pass


_torch.cuda.OutOfMemoryError = _OutOfMemoryError
_torch.cuda.is_available = lambda: True  # the existing tests take the CUDA-sync path (mocked)
_torch.cuda.empty_cache = lambda: None


class _AdamW:
    """Stand-in optimizer with the old mock's behaviour (no param groups; step/zero_grad no-ops)."""

    def __init__(self, *a, **k):
        self.param_groups = []

    def step(self):
        pass

    def zero_grad(self, set_to_none=True):
        pass


_torch.optim.AdamW = _AdamW
sys.modules.setdefault("torch", _torch)
for name in ("safetensors", "safetensors.torch"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["safetensors.torch"].save_file = lambda *a, **k: None
sys.modules["safetensors.torch"].load_file = lambda *a, **k: {}
