import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trainer"))

from godtrainer.selection import paired_stats, parse_tag, should_replace, unit_vector  # noqa: E402


def test_parse_tag_orders_kinds_and_steps():
    assert parse_tag("base") < parse_tag("raw@300") < parse_tag("ema@300") < parse_tag("deltaavg2@300")
    assert parse_tag("m1:ema@600") == parse_tag("ema@600")
    assert parse_tag("raw@300") < parse_tag("raw@600")
    assert parse_tag("retrain-all:ema@120")[0] == parse_tag("ema@1")[0]
    assert parse_tag("identity") == (0, -1)


def test_paired_stats_sign_and_shape():
    mean, se = paired_stats([1.0, 2.0, 3.0], [1.5, 2.5, 3.5])
    assert mean == pytest.approx(0.5) and se == pytest.approx(0.0)
    with pytest.raises(ValueError):
        paired_stats([1.0], [1.0, 2.0])


def test_noise_level_gain_on_earlier_raw_does_not_replace_smoother_state():
    best = [0.100, 0.200, 0.300, 0.400]
    cand = [0.099, 0.203, 0.296, 0.401]  # mean slightly lower, differences are noise
    assert sum(cand) < sum(best)
    assert not should_replace(sum(cand) / 4, "raw@300", cand, sum(best) / 4, "ema@600", best, k=1.0)


def test_consistent_gain_replaces():
    best = [0.100, 0.200, 0.300, 0.400]
    cand = [0.095, 0.192, 0.291, 0.389]
    assert should_replace(sum(cand) / 4, "raw@300", cand, sum(best) / 4, "ema@600", best, k=1.0)


def test_smoother_or_later_wins_ties_but_not_regressions():
    best = [0.100, 0.200, 0.300, 0.400]
    tie = list(best)
    worse = [0.101, 0.201, 0.301, 0.401]
    assert should_replace(0.25, "deltaavg2@900", tie, 0.25, "raw@900", best)
    assert not should_replace(0.2515, "deltaavg2@900", worse, 0.25, "raw@900", best)


def test_falls_back_to_mean_without_vectors():
    assert should_replace(0.1, "raw@1", None, 0.2, "base", [0.2, 0.2])
    assert not should_replace(0.3, "raw@1", None, 0.2, "base", None)
    assert not should_replace(float("nan"), "raw@1", [0.1, 0.1], 0.2, "base", [0.2, 0.2])
    assert should_replace(0.3, "raw@1", None, float("inf"), "base", None)


def test_unit_vector_is_per_image_only():
    assert unit_vector([1, 2, 3], [9] * 48) == [1, 2, 3]
    assert unit_vector([1, 2], [9] * 32) == [1, 2]
    assert unit_vector(None, [9] * 32) is None


def test_fewer_than_three_images_uses_champion_rule():
    best, cand = [0.20, 0.30], [0.199, 0.302]
    # A lower mean wins outright with only two images, even for an earlier raw state.
    assert should_replace(0.2499, "raw@300", cand, 0.25, "ema@600", best)
    assert not should_replace(0.2501, "deltaavg2@900", cand, 0.25, "raw@300", best)


def test_tie_replacements_never_raise_the_incumbent_mean():
    best = [0.100, 0.200, 0.300, 0.400]
    slightly_worse = [0.1001, 0.2001, 0.3001, 0.4001]
    assert not should_replace(0.25001, "deltaavg2@900", slightly_worse, 0.25, "raw@300", best)
