import argparse
import glob
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


def finalize_toolkit_output(out_dir, name):
    """Leave exactly one unambiguous LoRA for the evaluator: the final save (AI Toolkit `<name>.safetensors`,
    Musubi `<name>.safetensors`), else the highest-step intermediate if training was cut short. It is copied to
    last.safetensors, which select_lora() prefers; the other .safetensors files are removed from the upload."""
    files = [f for f in glob.glob(os.path.join(out_dir, "*.safetensors")) if os.path.basename(f) != "last.safetensors"]
    final = os.path.join(out_dir, f"{name}.safetensors")
    chosen = final if final in files else None
    if chosen is None:
        numbered = [(int(m.group(1)), f) for f in files if (m := re.search(r"[-_](\d+)\.safetensors$", f))]
        chosen = max(numbered)[1] if numbered else (files[0] if len(files) == 1 else None)
    if chosen is None:
        return None
    dst = os.path.join(out_dir, "last.safetensors")
    shutil.copy2(chosen, dst)
    for f in files:
        os.remove(f)
    return chosen


def run_until(cmd, deadline):
    """Run cmd in its own process group; stop it at the deadline so the saves made so far can still be shipped."""
    proc = subprocess.Popen(cmd, start_new_session=True)
    while True:
        try:
            return proc.wait(timeout=max(1.0, min(30.0, deadline - time.time())))
        except subprocess.TimeoutExpired:
            if time.time() < deadline:
                continue
            print("deadline reached: stopping trainer", flush=True)
            for sig, wait_s in ((signal.SIGTERM, 60), (signal.SIGKILL, 30)):
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    return proc.wait(timeout=wait_s)
                except subprocess.TimeoutExpired:
                    continue
            return proc.wait()


def run_toolkit(a, mode, deadline):
    cmd = [sys.executable, TOOLKIT_ENTRY, "--task-id", a.task_id, "--model", a.model, "--dataset-zip", a.dataset_zip,
           "--model-type", a.model_type, "--training-mode", mode, "--expected-repo-name", a.expected_repo_name,
           "--hours-to-complete", str(a.hours_to_complete)]
    if a.trigger_word:
        cmd += ["--trigger-word", a.trigger_word]
    if logs_enabled():
        print("running:", " ".join(cmd), flush=True)
    rc = run_until(cmd, deadline)
    out_dir = os.path.join("/app/checkpoints", a.task_id, a.expected_repo_name)
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
