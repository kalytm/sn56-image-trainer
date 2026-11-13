"""D10 review checks. Copy beside test_toolkit_dispatch.py; no GPU/network/model loads."""
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time

import pytest
from test_toolkit_dispatch import load_trainer


def valid_checkpoint():
    header = json.dumps({"lora.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
                        separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    return struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0)


@pytest.mark.parametrize("bad_name", ["repo.safetensors", "repo_000000750.safetensors"])
def test_finalization_skips_interrupted_save(tmp_path, bad_name):
    it = load_trainer()
    good = valid_checkpoint()
    (tmp_path / "repo_000000250.safetensors").write_bytes(good)
    # Correct header, missing the tensor payload: an interrupted direct write.
    (tmp_path / bad_name).write_bytes(good[:-4])
    chosen = it.finalize_toolkit_output(str(tmp_path), "repo")
    assert Path(chosen).name == "repo_000000250.safetensors"
    assert (tmp_path / "last.safetensors").read_bytes() == good


def test_finalization_does_not_ship_only_corrupt_save(tmp_path):
    it = load_trainer()
    (tmp_path / "repo.safetensors").write_bytes(valid_checkpoint()[:-4])
    try:
        chosen = it.finalize_toolkit_output(str(tmp_path), "repo")
    except (ValueError, RuntimeError):
        chosen = None
    assert chosen is None
    assert not (tmp_path / "last.safetensors").exists()


def test_deadline_stops_group_even_when_wrapper_exits_first(tmp_path, monkeypatch):
    it = load_trainer()
    child_file = tmp_path / "child.pid"
    code = r'''
import os, signal, sys, time
from pathlib import Path
child = os.fork()
if child == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path(sys.argv[1]).write_text(str(os.getpid()))
    time.sleep(30)
else:
    time.sleep(30)
'''
    groups = []
    original = subprocess.Popen

    def spawn(*args, **kwargs):
        proc = original(*args, **kwargs)
        groups.append(proc.pid)
        return proc

    monkeypatch.setattr(it.subprocess, "Popen", spawn)
    try:
        it.run_until([sys.executable, "-c", code, str(child_file)], time.time() + 0.75)
        assert child_file.exists(), "test child must be ready before the cutoff"
        pid = int(child_file.read_text())
        status = Path(f"/proc/{pid}/stat")
        alive = status.exists() and status.read_text().split(") ", 1)[1][0] != "Z"
        assert not alive, "wrapper exited, but trainer child still runs after run_until returned"
    finally:
        # Only process groups created by this test; never leave its child running.
        for pgid in groups:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_musubi_epoch_and_final_names_are_supported(tmp_path):
    it = load_trainer()
    good = valid_checkpoint()
    for name in ("repo-000001.safetensors", "repo-000002.safetensors"):
        (tmp_path / name).write_bytes(good)
    assert Path(it.finalize_toolkit_output(str(tmp_path), "repo")).name == "repo-000002.safetensors"
    (tmp_path / "repo.safetensors").write_bytes(good)
    assert Path(it.finalize_toolkit_output(str(tmp_path), "repo")).name == "repo.safetensors"
