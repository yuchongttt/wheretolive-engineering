"""Calibration properties: raw -> PCHIP percentile -> N(65, 15).

Added for this copy of the module: it pins the properties the scale relies on,
not specific breakpoint values (those are re-derived by recalibrate_all.py).
"""
import pytest

from simple_scorer import SimpleScorer

DIMS = ["transport", "community", "environment", "price", "schools", "total"]


@pytest.fixture(scope="module")
def scorer():
    return SimpleScorer()


@pytest.mark.parametrize("dim", DIMS)
def test_breakpoints_are_strictly_increasing(dim):
    bps = SimpleScorer.CALIBRATION_BREAKPOINTS[dim]
    xs = [x for x, _ in bps]
    ps = [p for _, p in bps]
    assert all(a < b for a, b in zip(xs, xs[1:])), "PCHIP needs strictly increasing x"
    assert all(a < b for a, b in zip(ps, ps[1:]))


@pytest.mark.parametrize("dim", DIMS)
def test_median_maps_to_65_and_breakpoints_reproduce_their_percentile(scorer, dim):
    for raw, pct in SimpleScorer.CALIBRATION_BREAKPOINTS[dim]:
        _, got_pct = scorer._calibrate_score(raw, dim)
        assert got_pct == pytest.approx(pct * 100, abs=0.11)
    median_raw = dict((p, x) for x, p in SimpleScorer.CALIBRATION_BREAKPOINTS[dim])[0.50]
    score, _ = scorer._calibrate_score(median_raw, dim)
    assert score == pytest.approx(65.0, abs=0.2)


@pytest.mark.parametrize("dim", DIMS)
def test_monotonic_and_clamped_over_the_whole_range(scorer, dim):
    prev = -1.0
    for tenth in range(-200, 1201):          # raw -20.0 .. 120.0, incl. extrapolation
        score, pct = scorer._calibrate_score(tenth / 10, dim)
        assert 5.0 <= score <= 98.0
        assert 0.1 <= pct <= 99.9
        assert score >= prev
        prev = score


def test_one_sd_above_the_median_is_about_80(scorer):
    # p84 ~ +1 SD: interpolate between the p80 and p85 breakpoints of one dimension
    bps = dict((p, x) for x, p in SimpleScorer.CALIBRATION_BREAKPOINTS["price"])
    raw_p84 = bps[0.80] + (bps[0.85] - bps[0.80]) * (0.8413 - 0.80) / 0.05
    score, _ = scorer._calibrate_score(raw_p84, "price")
    assert score == pytest.approx(80.0, abs=0.6)


def test_rating_is_a_single_positive_frame(scorer):
    assert scorer._get_rating(65.0) == "Better than 50% of postcodes"
    assert scorer._get_rating(95.0) == "Better than 98% of postcodes"
    assert scorer._get_rating(5.0) == "Better than 1% of postcodes"
    assert scorer._get_rating(None) == "Insufficient data"


def test_total_needs_three_dimensions(scorer):
    two = scorer.calculate_scores({
        "address": "",
        "schools": {"score": 85.6},
        "flood_risk": {"score": 100}, "noise": {"score": 70},
    })
    assert two["total_score"] is None and two["rating"] == "Insufficient data"
    assert two["percentiles"]["transport"] is None     # missing, not a fake median
    three = scorer.calculate_scores({
        "address": "",
        "schools": {"score": 85.6},
        "flood_risk": {"score": 100}, "noise": {"score": 70},
        "transit": {"score": 60, "lines_count": 2, "lines": []},
    })
    assert three["total_score"] is not None
    assert three["data_completeness"] == "3/5"
