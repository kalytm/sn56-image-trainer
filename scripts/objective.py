"""D11/D12: align the AI Toolkit families with the published evaluation objective, and fill the time budget with
independently seeded members whose LoRAs are averaged exactly.

The objective is public (G.O.D validator/evaluation/evaluators/diffusion_objective.py): the task's own training
objective scored only in the noisiest quartile of AI Toolkit's flow-match timesteps (index 0-250 of
linspace(1000, 1); HiDream O1 0-50), at resolution 1024, and the test loss is half captioned and half blank-caption.
Everything here depends only on the task's declared model type, its image count and its time budget.
"""
import os

# Declared-objective constants, mirrored from the public evaluator.
EVAL_MIN_DENOISING_STEPS = 0
EVAL_MAX_DENOISING_STEPS = 250
EVAL_HIDREAM_MAX_DENOISING_STEPS = 50
EVAL_RESOLUTION = 1024
BLANK_CAPTION_SHARE = 0.5           # test_loss = 0.5 * captioned + 0.5 * blank

# Steps per member = clamp(per_image * N, lo, hi). Starting points to be fitted by the S2 A/B.
STEPS_BY_FAMILY = {
    "flux2-klein-4b": (38, 250, 1000),
    "moss-image-base": (42, 378, 1134),
}
STEPS_DEFAULT = (40, 300, 1000)

MAX_MEMBERS = 8
MEMBER_SAFETY = 1.2                 # the next member must fit 1.2x the slowest member so far
MERGE_RESERVE_S = 90                # averaging + atomic publish after the last member


def member_steps(model_type, n_images):
    per, lo, hi = STEPS_BY_FAMILY.get(model_type, STEPS_DEFAULT)
    return int(min(hi, max(lo, per * max(1, n_images))))


def align_process(process, model_type, n_images):
    """Edit one AI Toolkit `process` dict in place to train the declared objective. Returns the step count."""
    ds = process.setdefault("datasets", [{}])[0]
    ds["caption_dropout_rate"] = BLANK_CAPTION_SHARE
    ds["resolution"] = [EVAL_RESOLUTION]
    train = process.setdefault("train", {})
    train["min_denoising_steps"] = EVAL_MIN_DENOISING_STEPS
    train["max_denoising_steps"] = (EVAL_HIDREAM_MAX_DENOISING_STEPS if model_type == "hidream-o1"
                                    else EVAL_MAX_DENOISING_STEPS)
    train["timestep_type"] = "linear"
    steps = member_steps(model_type, n_images)
    train["steps"] = steps
    return steps


def next_member_fits(member_walls, now, deadline):
    """Start another member only if 1.2x the slowest member so far plus the merge reserve ends before the deadline."""
    if not member_walls or len(member_walls) >= MAX_MEMBERS:
        return False
    return now + MEMBER_SAFETY * max(member_walls) + MERGE_RESERVE_S <= deadline


def enabled(name, default="1"):
    return os.environ.get(name, default) not in ("0", "", "false", "no")


# ---- exact LoRA averaging (rank concatenation)

_PAIRS = ((".lora_A.weight", ".lora_B.weight"), (".lora_down.weight", ".lora_up.weight"))


def _module_of(key):
    for down, up in _PAIRS:
        if key.endswith(down):
            return key[: -len(down)], down, up
    return None


def average_loras(state_dicts):
    """Exact mean of K LoRAs' weight deltas, as one LoRA of rank K*r per module.

    For each module, delta_i = s * B_i @ A_i with s = alpha / r (alpha defaults to r when absent). The result stacks
    A_i along the rank axis and B_i / K along it, so B' @ A' = mean_i(B_i @ A_i). If the module stores an alpha, it is
    multiplied by K so that alpha' / (K*r) == alpha / r keeps the per-rank scale. Every other tensor (e.g. a direct
    delta) is averaged element-wise. All inputs must have identical key sets, shapes and alphas."""
    import torch

    k = len(state_dicts)
    if k == 0:
        raise ValueError("no LoRAs to average")
    keys = set(state_dicts[0])
    for sd in state_dicts[1:]:
        if set(sd) != keys:
            raise ValueError("LoRA key sets differ between members")
    if k == 1:
        return dict(state_dicts[0])
    out, done = {}, set()
    for key in sorted(keys):
        m = _module_of(key)
        if m is None:
            continue
        base, down, up = m
        ka, kb, kalpha = base + down, base + up, base + ".alpha"
        if kb not in keys:
            raise ValueError(f"LoRA pair incomplete: {ka}")
        a_list = [sd[ka] for sd in state_dicts]
        b_list = [sd[kb] for sd in state_dicts]
        if any(t.shape != a_list[0].shape for t in a_list) or any(t.shape != b_list[0].shape for t in b_list):
            raise ValueError(f"LoRA shapes differ between members: {base}")
        dtype = b_list[0].dtype
        out[ka] = torch.cat(a_list, dim=0)
        out[kb] = torch.cat([b.float() / k for b in b_list], dim=1).to(dtype)
        done |= {ka, kb}
        if kalpha in keys:
            alphas = [float(sd[kalpha]) for sd in state_dicts]
            if any(abs(x - alphas[0]) > 1e-6 * max(1.0, abs(alphas[0])) for x in alphas):
                raise ValueError(f"LoRA alphas differ between members: {base}")
            ref = state_dicts[0][kalpha]
            out[kalpha] = (ref.float() * k).to(ref.dtype)
            done.add(kalpha)
    for key in keys - done:
        ref = state_dicts[0][key]
        if not ref.is_floating_point():
            raise ValueError(f"cannot average non-float tensor {key}")
        out[key] = (sum(sd[key].float() for sd in state_dicts) / k).to(ref.dtype)
    return out


def write_average(paths, dst):
    """Average the LoRA files `paths` into `dst` through a temp file and an atomic rename."""
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    sds = [load_file(str(p)) for p in paths]
    with safe_open(str(paths[0]), framework="pt") as f:
        meta = dict(f.metadata() or {})
    meta["g56_members"] = str(len(sds))
    avg = average_loras(sds)
    tmp = str(dst) + ".tmp"
    save_file(avg, tmp, metadata=meta)
    os.replace(tmp, dst)
    return len(sds)
