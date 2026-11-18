import argparse
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

START = time.time()
CACHE = "/cache"
KILL_MARGIN_S = 180
TEXT_TO_IMAGE = "text-to-image"
EDIT = "edit"
# Our engine trains these families (text-to-image only); it runs in its own venv in the image.
ENGINE_PYTHON = os.environ.get("ENGINE_PYTHON", sys.executable)
# Everything else (#1420 families and edit mode) runs the pinned AI Toolkit / Musubi path (scripts/toolkit_entry.py).
TOOLKIT_COMBOS = {
    ("flux2-klein-4b", TEXT_TO_IMAGE), ("flux2-klein-4b", EDIT),
    ("qwen-image-edit-2511", EDIT),
    ("hidream-o1", TEXT_TO_IMAGE), ("hidream-o1", EDIT),
    ("moss-image-base", TEXT_TO_IMAGE), ("moss-image-base", EDIT),
}
TOOLKIT_ENTRY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "toolkit_entry.py")

RECIPES = {
    "krea2": ["--rank", "16", "--lr", "2e-4"],
    "ideogram4": ["--rank", "16", "--lr", "2e-4", "--include", r"layers\.\d+\.(attention|adaln_modulation|feed_forward)", "--empty-prompt-frac", "0.3"],
    "z-image": ["--rank", "16", "--lr", "3e-4"],
    "flux": ["--rank", "16", "--lr", "3e-4", "--no-ckpt", "--include", r"(double_blocks|single_blocks)\.\d+\.(img_attn|txt_attn|img_mlp|txt_mlp|linear1|linear2)"],
    "qwen-image": ["--rank", "8", "--lr", "1e-4", "--include", r"transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v|to_out)"],
}
COMMON = ["--holdout-frac", "0.1", "--holdout-min", "3", "--holdout-noises", "4", "--eval-every", "300",
          "--select-rule", "paired", "--select-k", "1.0", "--schedule", "steps", "--eval-count", "6",
          "--epoch-cap", "5", "--eval-grid-gamma", "1.0", "--extra-members", "1",
          "--continue-lr-frac", "0.3", "--continue-min-frac", "0.25",
          "--lr-final-frac", "0.1", "--ema", "0.99", "--case-cycle", "--balanced-prompts", "--delta-average", "2"]
FAMILY_EXTRAS = {
    "krea2": ["--select-metric", "mean", "--eval-every", "200", "--edge-band-count", "2", "--edge-band-repeat", "3"],
    "ideogram4": ["--no-ckpt", "--eval-every", "200"],
    # D9: qwen-image keeps improving on the holdout to ~13 passes (dev 3d5f45ab, B2 and C), so no epoch cap;
    # without the cap the adaptive member loop (D6/D8) has nothing to fill. Later flags override COMMON.
    "qwen-image": ["--epoch-cap", "0", "--extra-members", "0", "--continue-lr-frac", "0"],
}


def logs_enabled():
    return os.environ.get("GOD_TRAIN_LOGS", "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_model_dir(model):
    cand = os.path.join(CACHE, "models", model.replace("/", "--"))
    if os.path.isdir(cand):
        return cand
    dirs = [d for d in glob.glob(os.path.join(CACHE, "models", "*")) if os.path.isdir(d) and not os.path.basename(d).startswith(".")]
    if len(dirs) == 1:
        return dirs[0]
    raise FileNotFoundError(f"model dir for {model!r} not found under {CACHE}/models")


def safetensors_complete(path):
    """Structural check of a safetensors file without loading tensors: an 8-byte header length, a JSON header whose
    tensors all lie inside the data section, at least one tensor, and a file length that ends exactly at the data
    section's end. A save interrupted at the deadline fails this."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                return False
            n = int.from_bytes(raw, "little")
            if n <= 0 or 8 + n > size or n > 100 * 1024 * 1024:
                return False
            header = json.loads(f.read(n))
    except (OSError, ValueError):
        return False
    if not isinstance(header, dict):
        return False
    end = 0
    tensors = 0
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        offs = meta.get("data_offsets") if isinstance(meta, dict) else None
        if not (isinstance(offs, list) and len(offs) == 2 and all(isinstance(o, int) for o in offs) and 0 <= offs[0] <= offs[1]):
            return False
        end = max(end, offs[1])
        tensors += 1
    return tensors > 0 and size == 8 + n + end


def finalize_toolkit_output(out_dir, name):
    """Leave exactly one complete LoRA for the evaluator as last.safetensors, which select_lora() prefers.
    Candidates in order: the trainer's final save (AI Toolkit and Musubi both write `<name>.safetensors`), then the
    numbered intermediates from the highest step/epoch down, then a single unnamed file. The first candidate that
    passes safetensors_complete() is published through a temporary file and an atomic rename; only then are the
    original .safetensors files removed. Returns the chosen path, or None (nothing published, nothing removed).
    Call only after the trainer's whole process group has stopped (run_until)."""
    files = [f for f in glob.glob(os.path.join(out_dir, "*.safetensors")) if os.path.basename(f) != "last.safetensors"]
    final = os.path.join(out_dir, f"{name}.safetensors")
    numbered = sorted(((int(m.group(1)), f) for f in files if f != final
                       and (m := re.search(r"[-_](\d+)\.safetensors$", f))), reverse=True)
    order = ([final] if final in files else []) + [f for _, f in numbered]
    if not order and len(files) == 1:
        order = files
    chosen = next((f for f in order if safetensors_complete(f)), None)
    if chosen is None:
        return None
    dst = os.path.join(out_dir, "last.safetensors")
    tmp = dst + ".tmp"
    shutil.copyfile(chosen, tmp)
    if not safetensors_complete(tmp):
        os.remove(tmp)
        return None
    os.replace(tmp, dst)
    for f in files:
        os.remove(f)
    return chosen


def _reap_children(leader=None):
    # In the container this process is PID 1, so orphaned trainer processes are re-parented to it: reap them, or
    # their zombies would keep the process group "alive" forever. The leader's own status is kept on its Popen.
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return
        if leader is not None and pid == leader.pid and leader.returncode is None:
            leader.returncode = os.waitstatus_to_exitcode(status)


def _group_alive(pgid, leader=None):
    _reap_children(leader)
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def stop_group(pgid, term_s=60, kill_s=30, leader=None):
    """SIGTERM the whole process group, then SIGKILL whatever is left (including descendants whose parent already
    exited). Every wait is bounded (term_s + kill_s, inside the 180 s kill margin). Returns True once no process of
    the group remains."""
    for sig, wait_s in ((signal.SIGTERM, term_s), (signal.SIGKILL, kill_s)):
        if not _group_alive(pgid, leader):
            return True
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return True
        t_end = time.time() + wait_s
        while time.time() < t_end:
            if not _group_alive(pgid, leader):
                return True
            time.sleep(0.2)
    return not _group_alive(pgid, leader)


def run_until(cmd, deadline, term_s=60, kill_s=30):
    """Run cmd in its own process group (session). At the deadline, or as soon as the leader exits, make sure no
    process of that group is still running before returning, so checkpoint finalization never races a writer.
    Returns (rc, stopped): rc is the leader's exit code (or -9 if it had to be killed), stopped is False only if
    the group could not be confirmed gone within the bounded waits."""
    proc = subprocess.Popen(cmd, start_new_session=True)
    rc = None
    while rc is None:
        try:
            rc = proc.wait(timeout=max(0.2, min(30.0, deadline - time.time())))
        except subprocess.TimeoutExpired:
            if time.time() >= deadline:
                print("deadline reached: stopping trainer", flush=True)
                break
    if rc is None or _group_alive(proc.pid, proc):
        stopped = stop_group(proc.pid, term_s, kill_s, leader=proc)
        if rc is None:
            if proc.returncode is not None:
                rc = proc.returncode
            else:
                try:
                    rc = proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    rc = -9
        return rc, stopped
    return rc, True


def run_toolkit(a, mode, deadline):
    cmd = [sys.executable, TOOLKIT_ENTRY, "--task-id", a.task_id, "--model", a.model, "--dataset-zip", a.dataset_zip,
           "--model-type", a.model_type, "--training-mode", mode, "--expected-repo-name", a.expected_repo_name,
           "--hours-to-complete", str(a.hours_to_complete), "--deadline-epoch", f"{deadline:.3f}"]
    if a.trigger_word:
        cmd += ["--trigger-word", a.trigger_word]
    if logs_enabled():
        print("running:", " ".join(cmd), flush=True)
    rc, stopped = run_until(cmd, deadline)
    out_dir = os.path.join("/app/checkpoints", a.task_id, a.expected_repo_name)
    if not stopped:
        print("trainer processes could not be confirmed stopped; not finalizing", flush=True)
        sys.exit(rc or 1)
    chosen = finalize_toolkit_output(out_dir, a.expected_repo_name) if os.path.isdir(out_dir) else None
    print(f"toolkit exit code {rc}; shipped {chosen or 'nothing'}", flush=True)
    sys.exit(0 if chosen else (rc or 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset-zip", required=True)
    p.add_argument("--model-type", required=True)
    p.add_argument("--expected-repo-name", required=True)
    p.add_argument("--hours-to-complete", type=float, required=True)
    p.add_argument("--trigger-word", default=None)
    p.add_argument("--training-mode", default=TEXT_TO_IMAGE)
    a = p.parse_args()

    family = a.model_type.lower().replace("_", "-")
    mode = a.training_mode.lower()
    deadline = START + a.hours_to_complete * 3600 - KILL_MARGIN_S
    if (family, mode) in TOOLKIT_COMBOS:
        a.model_type = family
        return run_toolkit(a, mode, deadline)
    if family not in RECIPES or mode != TEXT_TO_IMAGE:
        raise SystemExit(f"unsupported model type / training mode {a.model_type} / {a.training_mode}")
    dataset = os.path.join(CACHE, "datasets", f"{a.task_id}_tourn.zip")
    if not os.path.exists(dataset):
        dataset = a.dataset_zip if os.path.exists(a.dataset_zip) else dataset
    model_dir = resolve_model_dir(a.model)
    out_final = os.path.join("/app/checkpoints", a.task_id, a.expected_repo_name)
    work = "/tmp/godtrainer-out"
    extra = ["--text-enc-dtype", "bf16"] if family == "qwen-image" else []
    cmd = [ENGINE_PYTHON, "/app/trainer/train.py", "--family", family, "--model-dir", model_dir, "--hf-cache", os.path.join(CACHE, "hf_cache"),
           "--baked-dir", "/opt/godtrainer/assets", "--comfy-root", os.environ.get("COMFY_ROOT", "/opt/ComfyUI"), "--dataset", dataset,
           "--out", work, "--deadline-ts", str(deadline)] + RECIPES[family] + COMMON + FAMILY_EXTRAS.get(family, [])\
          + extra
    if a.trigger_word:
        cmd += ["--trigger-word", a.trigger_word]
    if logs_enabled():
        print("running:", " ".join(cmd), flush=True)
    # The engine imports godtrainer from /app/trainer; set it only here so the AI Toolkit stack never sees it.
    rc = subprocess.call(cmd, env={**os.environ, "PYTHONPATH": "/app/trainer"})
    if rc or logs_enabled():
        print(f"trainer exit code {rc}", flush=True)
    os.makedirs(out_final, exist_ok=True)
    src = os.path.join(work, "last.safetensors")
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(out_final, "last.safetensors"))
        for f in ("summary.json", "config.json"):
            if os.path.exists(os.path.join(work, f)):
                shutil.copy2(os.path.join(work, f), os.path.join(out_final, f))
        if logs_enabled():
            print("saved", os.path.join(out_final, "last.safetensors"), flush=True)
        if rc:
            sys.exit(rc)
    else:
        print("no checkpoint produced", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
