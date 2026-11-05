"""Noise-aware checkpoint selection (image-v1, D1).

The champion replaced its best checkpoint whenever a candidate's holdout mean was lower. With a
holdout of a few images, taking the minimum over many noisy evaluations favours whichever
checkpoint happened to fit the holdout draw best (winner's curse). Here a candidate replaces the
incumbent only when its paired improvement is larger than ``k`` standard errors, or when it is
not worse on the mean and is a smoother/later state (delta-average > EMA > raw > base).

All candidates are scored on identical cases (same holdout images, bands and seeded noises), so
differences are paired and much of the noise cancels.
"""

import math
import re

import numpy as np

KIND_RANK = {"base": 0, "identity": 0, "raw": 1, "ema": 2, "deltaavg": 3, "soup": 3}
_MEMBER_PREFIX = re.compile(r"^m\d+:")
_TAG = re.compile(r"^(?P<kind>[a-z]+?)(?:\d+)?(?:@(?P<step>\d+))?$")


def parse_tag(tag):
    """Return (kind_rank, step) for tags such as 'raw@300', 'm0:ema@600', 'deltaavg2@900', 'soup'."""
    tag = _MEMBER_PREFIX.sub("", tag or "")
    if tag.startswith("retrain-all:"):
        tag = tag[len("retrain-all:"):]
    m = _TAG.match(tag)
    if not m:
        return 0, -1
    return KIND_RANK.get(m.group("kind"), 0), int(m.group("step") or -1)


def paired_stats(candidate, incumbent):
    """Mean and standard error of the paired improvement (incumbent - candidate; >0 = candidate better)."""
    c = np.asarray(candidate, dtype=np.float64)
    b = np.asarray(incumbent, dtype=np.float64)
    if c.shape != b.shape or c.size == 0:
        raise ValueError("paired vectors must be non-empty and the same shape")
    d = b - c
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(d.size)) if d.size > 1 else float("inf")
    return mean, se


def unit_vector(per_image, cells):
    """Pick the comparison unit: per-image losses when there are enough images, else the
    (image, band) cells. With < 3 images an image-level standard error is meaningless."""
    if per_image is not None and len(per_image) >= 3:
        return list(per_image)
    return list(cells) if cells is not None else list(per_image or [])


def should_replace(cand_score, cand_tag, cand_vec, best_score, best_tag, best_vec, k=1.0):
    """Decide whether the candidate replaces the incumbent best checkpoint.

    Falls back to the plain mean comparison when either side has no paired vector (for example,
    a state restored from an older run) or when the vectors are not comparable.
    """
    if not math.isfinite(cand_score):
        return False
    if not math.isfinite(best_score):
        return True
    if cand_vec is None or best_vec is None or len(cand_vec) != len(best_vec) or len(cand_vec) < 2:
        return cand_score < best_score
    mean, se = paired_stats(cand_vec, best_vec)
    if mean > k * se:
        return True
    smoother_or_later = parse_tag(cand_tag) >= parse_tag(best_tag)
    return smoother_or_later and cand_score <= best_score
