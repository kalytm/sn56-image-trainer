"""D11/D12 tests that need real torch/safetensors/yaml (no torch mock): run from the repo root with
`python -m pytest -q -p no:cacheprovider tests_torch` in an environment that has torch (CPU is enough)."""
import importlib.util
import os
import sys

import pytest
import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import objective  # noqa: E402


def _load_entry():
    spec = importlib.util.spec_from_file_location("toolkit_entry", os.path.join(ROOT, "scripts", "toolkit_entry.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- D11 objective alignment

def _template(name="base_diffusion_flux2_klein_4b.yaml"):
    with open(os.path.join(ROOT, "templates", name)) as f:
        return yaml.safe_load(f)["config"]["process"][0]


def test_align_sets_the_declared_objective():
    p = _template()
    steps = objective.align_process(p, "flux2-klein-4b", 24)
    ds, tr = p["datasets"][0], p["train"]
    assert ds["caption_dropout_rate"] == 0.5 and ds["resolution"] == [1024]
    assert (tr["min_denoising_steps"], tr["max_denoising_steps"], tr["timestep_type"]) == (0, 250, "linear")
    assert steps == tr["steps"] == 912


def test_hidream_band_and_step_clamps():
    p = _template("base_diffusion_hidream_o1.yaml")
    objective.align_process(p, "hidream-o1", 5)
    assert p["train"]["max_denoising_steps"] == 50
    assert objective.member_steps("moss-image-base", 1) == 378
    assert objective.member_steps("moss-image-base", 10) == 420
    assert objective.member_steps("moss-image-base", 100) == 1134
    assert objective.member_steps("flux2-klein-4b", 0) == 250
    assert objective.member_steps("qwen-image-edit-2511", 50) == 1000


def test_next_member_fits():
    assert not objective.next_member_fits([], 0, 10_000)
    assert objective.next_member_fits([100, 300], 0, 1.2 * 300 + objective.MERGE_RESERVE_S)
    assert not objective.next_member_fits([100, 300], 1, 1.2 * 300 + objective.MERGE_RESERVE_S)
    assert not objective.next_member_fits([1] * objective.MAX_MEMBERS, 0, 10_000)


# ---- D12 exact averaging

def _lora(seed, r=4, d_in=6, d_out=5, alpha=None, names=("a.b", "c.d"), style=("lora_A", "lora_B")):
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for n in names:
        sd[f"{n}.{style[0]}.weight"] = torch.randn(r, d_in, generator=g)
        sd[f"{n}.{style[1]}.weight"] = torch.randn(d_out, r, generator=g)
        if alpha is not None:
            sd[f"{n}.alpha"] = torch.tensor(float(alpha))
    return sd


def _delta(sd, n, style=("lora_A", "lora_B")):
    a, b = sd[f"{n}.{style[0]}.weight"], sd[f"{n}.{style[1]}.weight"]
    r = a.shape[0]
    alpha = float(sd[f"{n}.alpha"]) if f"{n}.alpha" in sd else r
    return (alpha / r) * (b @ a)


@pytest.mark.parametrize("alpha,style", [(None, ("lora_A", "lora_B")), (8.0, ("lora_A", "lora_B")),
                                         (2.0, ("lora_down", "lora_up"))])
def test_average_is_the_exact_mean_delta(alpha, style):
    sds = [_lora(s, alpha=alpha, style=style) for s in (1, 2, 3)]
    avg = objective.average_loras(sds)
    for n in ("a.b", "c.d"):
        want = sum(_delta(sd, n, style) for sd in sds) / 3
        assert torch.allclose(_delta(avg, n, style), want, atol=1e-5)
        assert avg[f"{n}.{style[0]}.weight"].shape[0] == 12
    if alpha is not None:
        assert float(avg["a.b.alpha"]) == pytest.approx(3 * alpha)


def test_average_of_one_is_identity_and_mismatches_raise():
    sd = _lora(1)
    assert objective.average_loras([sd]).keys() == sd.keys()
    with pytest.raises(ValueError):
        objective.average_loras([_lora(1), _lora(2, names=("a.b",))])
    with pytest.raises(ValueError):
        objective.average_loras([_lora(1, r=4), _lora(2, r=8)])
    with pytest.raises(ValueError):
        objective.average_loras([_lora(1, alpha=4), _lora(2, alpha=8)])
    with pytest.raises(ValueError):
        objective.average_loras([])


def test_non_lora_tensors_are_averaged_and_dtype_kept():
    a, b = _lora(1), _lora(2)
    a["x.diff"], b["x.diff"] = torch.ones(3, dtype=torch.bfloat16), 3 * torch.ones(3, dtype=torch.bfloat16)
    for sd in (a, b):
        for k in list(sd):
            sd[k] = sd[k].to(torch.bfloat16)
    avg = objective.average_loras([a, b])
    assert avg["x.diff"].dtype == torch.bfloat16 and torch.equal(avg["x.diff"].float(), 2 * torch.ones(3))
    assert avg["a.b.lora_B.weight"].dtype == torch.bfloat16


def test_write_average_is_atomic_and_keeps_metadata(tmp_path):
    paths = []
    for s in (1, 2):
        p = tmp_path / f"m{s}.safetensors"
        save_file(_lora(s), str(p), metadata={"format": "pt", "software": "ai-toolkit"})
        paths.append(p)
    dst = tmp_path / "final.safetensors"
    assert objective.write_average(paths, dst) == 2
    with safe_open(str(dst), framework="pt") as f:
        assert f.metadata()["software"] == "ai-toolkit" and f.metadata()["g56_members"] == "2"
    assert not (tmp_path / "final.safetensors.tmp").exists()


# ---- D12 member loop (subprocess faked; the trainer is never run)

def _setup_entry(tmp_path, monkeypatch, outcomes, clock):
    te = _load_entry()
    monkeypatch.setattr(te, "CHECKPOINTS_ROOT", tmp_path / "ckpt")
    monkeypatch.setattr(te, "CONFIG_OUTPUT_ROOT", tmp_path / "cfg")
    (tmp_path / "cfg").mkdir()
    cfg = {"config": {"name": "job", "process": [_template()]}}
    cfg["config"]["process"][0]["training_folder"] = str(tmp_path / "ckpt" / "t1")
    path = tmp_path / "cfg" / "t1.yaml"
    path.write_text(yaml.safe_dump(cfg))
    calls = []

    class R:
        def __init__(self, rc):
            self.returncode = rc

    def fake_run(cmd, cwd=None, env=None, **k):
        c = yaml.safe_load(open(cmd[-1]))["config"]["process"][0]
        seed = int(env["G56_SEED"])
        calls.append((seed, c["training_folder"], c["save"]["save_every"]))
        clock[0] += 300
        rc = outcomes[len(calls) - 1]
        if rc == 0:
            out = os.path.join(c["training_folder"], "job")
            os.makedirs(out, exist_ok=True)
            save_file(_lora(seed), os.path.join(out, "job.safetensors"))
        return R(rc)

    monkeypatch.setattr(te.subprocess, "run", fake_run)
    monkeypatch.setattr(te.time, "time", lambda: clock[0])
    return te, path, calls


class _Args:
    task_id = "t1"


def test_members_until_the_next_does_not_fit_and_final_is_their_average(tmp_path, monkeypatch):
    clock = [0.0]
    deadline = 2 * 300 + 1.2 * 300 + objective.MERGE_RESERVE_S + 100  # member 2 starts at 600 s, member 3 cannot
    te, path, calls = _setup_entry(tmp_path, monkeypatch, [0, 0, 0, 0], clock)
    te.run_members(_Args, path, deadline)
    assert [c[0] for c in calls] == [1000, 1001, 1002]
    assert calls[0][1] == str(tmp_path / "ckpt" / "t1") and calls[0][2] == 250          # member 0 unchanged
    assert calls[1][1].endswith("members/m1") and calls[1][2] == 1001                  # later members: final save only
    final = load_file(str(tmp_path / "ckpt" / "t1" / "job" / "job.safetensors"))
    want = sum(_delta(_lora(s), "a.b") for s in (1000, 1001, 1002)) / 3
    assert torch.allclose(_delta(final, "a.b"), want, atol=1e-5)


def test_failed_later_member_keeps_previous_average_and_member0_failure_raises(tmp_path, monkeypatch):
    clock = [0.0]
    te, path, calls = _setup_entry(tmp_path, monkeypatch, [0, 0, 1], clock)
    te.run_members(_Args, path, 10_000)
    final = load_file(str(tmp_path / "ckpt" / "t1" / "job" / "job.safetensors"))
    want = sum(_delta(_lora(s), "a.b") for s in (1000, 1001)) / 2
    assert len(calls) == 3 and torch.allclose(_delta(final, "a.b"), want, atol=1e-5)

    clock = [0.0]
    (tmp_path / "x").mkdir()
    te, path, calls = _setup_entry(tmp_path / "x", monkeypatch, [1], clock)
    with pytest.raises(RuntimeError):
        te.run_members(_Args, path, 10_000)


def test_single_member_output_is_byte_identical_to_a_plain_run(tmp_path, monkeypatch):
    clock = [0.0]
    te, path, calls = _setup_entry(tmp_path, monkeypatch, [0], clock)
    te.run_members(_Args, path, 300 + objective.MERGE_RESERVE_S)          # member 1 cannot fit
    final = tmp_path / "ckpt" / "t1" / "job" / "job.safetensors"
    keep = tmp_path / "ckpt" / "t1" / "members" / "m0.safetensors"
    assert len(calls) == 1 and final.read_bytes() == keep.read_bytes()
