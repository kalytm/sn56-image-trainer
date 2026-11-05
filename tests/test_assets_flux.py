"""D5: standalone FLUX checkpoints (bare transformer file) load their text encoders and VAE from the
validator-provided FLUX_* artifacts; bundled checkpoints keep the champion's path."""
import json
import struct

import pytest

from godtrainer import assets as A


def write_header(path, keys):
    header = {k: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for k in keys}
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0\0")
    return str(path)


@pytest.fixture
def env(tmp_path, monkeypatch):
    loaded = []
    monkeypatch.setattr(A, "load_file", lambda p, device=None: loaded.append(p) or {"from": p})
    monkeypatch.setattr(A, "to_bf16", lambda sd: sd)
    for part, (var, name) in A.FLUX_ARTIFACTS.items():
        f = tmp_path / f"env-{name}"
        f.write_bytes(b"x")
        monkeypatch.setenv(var, str(f))
    return tmp_path, loaded, monkeypatch


def flux_assets(tmp_path, monkeypatch, keys, sd):
    ckpt = write_header(tmp_path / "flux.safetensors", keys)
    monkeypatch.setattr(A, "_root_safetensors", lambda d, min_bytes=0: [ckpt])
    monkeypatch.setattr(A, "load_sharded", lambda files: dict(sd))
    return A.Assets("flux", str(tmp_path), hf_cache=str(tmp_path / "hf"), baked_dir=str(tmp_path / "baked"))


def test_safetensors_keys_reads_header_only(tmp_path):
    p = write_header(tmp_path / "a.safetensors", ["double_blocks.0.img_attn.qkv.weight", "img_in.weight"])
    assert sorted(A.safetensors_keys(p)) == ["double_blocks.0.img_attn.qkv.weight", "img_in.weight"]


def test_standalone_flux_uses_validator_artifacts(env):
    tmp_path, loaded, mp = env
    bare = {"double_blocks.0.img_attn.qkv.weight": 1, "img_in.weight": 2}
    a = flux_assets(tmp_path, mp, list(bare), bare)
    assert a.flux_standalone()
    assert a.diffusion_sd() == bare  # whole bare file, not an empty prefix sub-dict
    sds = a.text_encoder_sds()
    assert [d["from"] for d in sds] == [str(tmp_path / "env-clip_l.safetensors"), str(tmp_path / "env-t5xxl_fp16.safetensors")]
    kind, vae = a.vae()
    assert kind == "comfy" and vae["from"] == str(tmp_path / "env-ae.safetensors")


def test_bundled_flux_keeps_champion_path(env):
    tmp_path, loaded, mp = env
    full = {"model.diffusion_model.img_in.weight": 1, "vae.decoder.conv_in.weight": 2,
            "text_encoders.clip_l.transformer.text_model.x": 3, "text_encoders.t5xxl.transformer.encoder.y": 4}
    a = flux_assets(tmp_path, mp, list(full), full)
    assert not a.flux_standalone()
    assert a.diffusion_sd() == {"img_in.weight": 1}
    assert a.vae() == ("comfy", {"decoder.conv_in.weight": 2})
    assert a.text_encoder_sds() == [{"text_model.x": 3}, {"encoder.y": 4}]
    assert loaded == []  # no external artifact touched


def test_missing_artifact_is_a_clear_error(env):
    tmp_path, loaded, mp = env
    mp.delenv("FLUX_VAE_PATH")
    bare = {"img_in.weight": 1}
    a = flux_assets(tmp_path, mp, list(bare), bare)
    with pytest.raises(FileNotFoundError, match="ae.safetensors"):
        a.vae()


def test_hf_cache_fallback_when_env_unset(env):
    tmp_path, loaded, mp = env
    mp.delenv("FLUX_CLIP_PATH")
    (tmp_path / "hf").mkdir()
    (tmp_path / "hf" / "clip_l.safetensors").write_bytes(b"x")
    bare = {"img_in.weight": 1}
    a = flux_assets(tmp_path, mp, list(bare), bare)
    assert a.flux_artifact("clip_l") == str(tmp_path / "hf" / "clip_l.safetensors")


def test_other_families_are_never_standalone_flux(env):
    tmp_path, _, _ = env
    assert not A.Assets("krea2", str(tmp_path)).flux_standalone()
