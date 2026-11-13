"""image-v2: dispatch between our engine and the ported AI Toolkit / Musubi path (#1420 families and edit mode)."""
import importlib.util
import os
import sys
import time

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_trainer():
    spec = importlib.util.spec_from_file_location("image_trainer_v2", os.path.join(ROOT, "scripts", "image_trainer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BASE_ARGS = ["--task-id", "t1", "--model", "org/m", "--dataset-zip", "/x.zip", "--expected-repo-name", "repo1",
             "--hours-to-complete", "1"]


def run_main(monkeypatch, extra):
    it = load_trainer()
    seen = {}
    monkeypatch.setattr(it, "run_toolkit", lambda a, mode, deadline: seen.update(path="toolkit", mode=mode, family=a.model_type))

    def engine_call(cmd, env=None):
        seen.update(path="engine", cmd=cmd, env=env)
        raise SystemExit(0)

    monkeypatch.setattr(it.subprocess, "call", engine_call)
    monkeypatch.setattr(it, "resolve_model_dir", lambda m: "/cache/models/org--m")
    monkeypatch.setattr(sys, "argv", ["image_trainer.py"] + BASE_ARGS + extra)
    try:
        it.main()
    except SystemExit:
        pass
    return it, seen


@pytest.mark.parametrize("family,mode", [
    ("flux2-klein-4b", None), ("flux2-klein-4b", "edit"), ("qwen-image-edit-2511", "edit"),
    ("hidream-o1", None), ("hidream-o1", "edit"), ("moss-image-base", None), ("moss-image-base", "edit")])
def test_new_families_and_edit_go_to_toolkit(monkeypatch, family, mode):
    extra = ["--model-type", family] + (["--training-mode", mode] if mode else [])
    _, seen = run_main(monkeypatch, extra)
    assert seen["path"] == "toolkit"
    assert seen["mode"] == (mode or "text-to-image")
    assert seen["family"] == family


@pytest.mark.parametrize("family", ["z-image", "qwen-image", "krea2"])
def test_legacy_text_to_image_stays_on_our_engine(monkeypatch, family):
    it, seen = run_main(monkeypatch, ["--model-type", family])
    assert seen["path"] == "engine"
    assert seen["cmd"][0] == it.ENGINE_PYTHON and seen["cmd"][1] == "/app/trainer/train.py"
    assert seen["env"]["PYTHONPATH"] == "/app/trainer"


def test_model_type_underscores_are_normalised_for_toolkit(monkeypatch):
    _, seen = run_main(monkeypatch, ["--model-type", "MOSS_IMAGE_BASE"])
    assert seen == {"path": "toolkit", "mode": "text-to-image", "family": "moss-image-base"}


@pytest.mark.parametrize("extra", [["--model-type", "z-image", "--training-mode", "edit"],
                                   ["--model-type", "qwen-image-edit-2511"],
                                   ["--model-type", "unknown-family"]])
def test_unsupported_combinations_fail_loudly(monkeypatch, extra):
    it = load_trainer()
    monkeypatch.setattr(sys, "argv", ["image_trainer.py"] + BASE_ARGS + extra)
    with pytest.raises(SystemExit) as e:
        it.main()
    assert "unsupported" in str(e.value.code)


def test_every_toolkit_combo_has_a_template_or_musubi_path():
    it = load_trainer()
    spec = importlib.util.spec_from_file_location("toolkit_entry", os.path.join(ROOT, "scripts", "toolkit_entry.py"))
    src = open(spec.origin).read()
    for family, mode in it.TOOLKIT_COMBOS:
        if (family, mode) == ("hidream-o1", "edit"):
            assert "run_hidream_edit" in src
            continue
        name = f'("{family}", {"TEXT_TO_IMAGE" if mode == "text-to-image" else "EDIT"}): "'
        assert name in src, (family, mode)
        template = src.split(name, 1)[1].split('"', 1)[0]
        assert os.path.isfile(os.path.join(ROOT, "templates", template)), template


def touch(d, name):
    p = os.path.join(d, name)
    open(p, "wb").write(name.encode())
    return p


def test_finalize_prefers_the_final_save(tmp_path):
    it = load_trainer()
    d = str(tmp_path)
    for n in ("repo1_000000250.safetensors", "repo1_000000750.safetensors", "repo1.safetensors"):
        touch(d, n)
    touch(d, "optimizer.pt")
    chosen = it.finalize_toolkit_output(d, "repo1")
    assert os.path.basename(chosen) == "repo1.safetensors"
    assert sorted(os.listdir(d)) == ["last.safetensors", "optimizer.pt"]
    assert open(os.path.join(d, "last.safetensors"), "rb").read() == b"repo1.safetensors"


def test_finalize_falls_back_to_highest_step_when_cut_short(tmp_path):
    it = load_trainer()
    d = str(tmp_path)
    for n in ("repo1_000000250.safetensors", "repo1_000000750.safetensors", "repo1-000002.safetensors"):
        touch(d, n)
    chosen = it.finalize_toolkit_output(d, "repo1")
    assert os.path.basename(chosen) == "repo1_000000750.safetensors"
    assert os.listdir(d) == ["last.safetensors"]


def test_finalize_with_nothing_saved_returns_none(tmp_path):
    it = load_trainer()
    assert it.finalize_toolkit_output(str(tmp_path), "repo1") is None


def test_run_until_stops_a_trainer_at_the_deadline():
    it = load_trainer()
    t0 = time.time()
    rc = it.run_until([sys.executable, "-c", "import time; time.sleep(60)"], time.time() + 1.5)
    assert rc != 0 and time.time() - t0 < 30


def test_run_toolkit_ships_last_and_exits_zero(monkeypatch, tmp_path):
    it = load_trainer()
    out = tmp_path / "t1" / "repo1"
    out.mkdir(parents=True)

    def fake_run(cmd, deadline):
        assert cmd[1] == it.TOOLKIT_ENTRY and "--training-mode" in cmd
        touch(str(out), "repo1.safetensors")
        return 0

    monkeypatch.setattr(it, "run_until", fake_run)
    real_join = os.path.join
    monkeypatch.setattr(it.os.path, "join", lambda *p: real_join(str(tmp_path), *p[1:]) if p[0] == "/app/checkpoints" else real_join(*p))
    a = type("A", (), dict(task_id="t1", model="org/m", dataset_zip="/x.zip", model_type="moss-image-base",
                           expected_repo_name="repo1", hours_to_complete=1.0, trigger_word=None))()
    with pytest.raises(SystemExit) as e:
        it.run_toolkit(a, "edit", time.time() + 100)
    assert e.value.code == 0
    assert os.listdir(out) == ["last.safetensors"]
