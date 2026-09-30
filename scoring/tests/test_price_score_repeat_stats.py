# tests/test_price_score_repeat_stats.py
"""Tests for the data source of the price dimension's "growth stability / market activity" sub-scores (SCORER_VERSION v8).

Before the fix, both read price_data["repeat_sales"] — the curated ≤5-item display sample from
_select_repeat_sales — so the activity score was almost constant at ~7/100 for every postcode.
After the fix they read only the full-population stats repeat_sales_count / repeat_sales_return_std;
a missing field (should not happen under CACHE_VERSION isolation) gets a neutral 50 per the insufficient-data convention,
never falling back to the biased display sample.
"""
import math

from simple_scorer import SimpleScorer


def _price_data(**extra):
    base = {
        "average_price": 500_000,
        "cagr_10y": 4.0,
        "cagr_3y": 2.0,
        # Curated display sample (fixed 5 items, simulating all the data the pre-fix scorer could see)
        "repeat_sales": [
            {"annualized_return": r} for r in (3.0, 4.0, 5.0, 6.0, 7.0)
        ],
    }
    base.update(extra)
    return base


def _activity(txns):
    return 100.0 / (1.0 + math.exp(-0.1 * (txns - 30)))


class TestActivityScore:
    def test_uses_full_pair_count_when_present(self):
        result = SimpleScorer()._score_price(_price_data(repeat_sales_count=60))
        assert result["activity_score"] == round(_activity(60), 1)  # ~95, not the ~7.6 from len=5

    def test_active_area_now_distinguishable_from_quiet_area(self):
        active = SimpleScorer()._score_price(_price_data(repeat_sales_count=60))
        quiet = SimpleScorer()._score_price(_price_data(repeat_sales_count=6))
        assert active["activity_score"] - quiet["activity_score"] > 50

    def test_missing_count_is_neutral_not_biased_sample_len(self):
        # Missing repeat_sales_count (should not happen under CACHE_VERSION isolation) → neutral 50;
        # never use the len of the display sample (that is exactly the bias v8 fixed)
        result = SimpleScorer()._score_price(_price_data())
        assert result["activity_score"] == 50.0

    def test_zero_pairs_is_neutral(self):
        result = SimpleScorer()._score_price(
            _price_data(repeat_sales_count=0, repeat_sales=[])
        )
        assert result["activity_score"] == 50.0


class TestStabilityScore:
    # Curve midpoint 10, slope 0.15: calibrated on the full-population pair std distribution (simulated n=985, median ~10)
    def test_uses_full_distribution_std_when_present(self):
        result = SimpleScorer()._score_price(
            _price_data(repeat_sales_return_std=20.0, repeat_sales_count=60)
        )
        expected = 100.0 / (1.0 + math.exp(0.15 * (20.0 - 10)))
        assert result["stability_score"] == round(expected, 1)

    def test_median_std_scores_neutral(self):
        # std=10 (median of the full distribution) should land on the curve midpoint = 50 points
        result = SimpleScorer()._score_price(
            _price_data(repeat_sales_return_std=10.0, repeat_sales_count=60)
        )
        assert result["stability_score"] == 50.0

    def test_missing_std_is_neutral_not_biased_sample_stdev(self):
        # Missing repeat_sales_return_std → neutral 50; never compute a stdev from the ≤5-item display sample
        result = SimpleScorer()._score_price(_price_data())
        assert result["stability_score"] == 50.0

    def test_none_std_with_sparse_pairs_is_neutral(self):
        # The analyser outputs repeat_sales_return_std=None for postcodes with <2 pairs → neutral 50
        result = SimpleScorer()._score_price(
            _price_data(repeat_sales_return_std=None, repeat_sales_count=1)
        )
        assert result["stability_score"] == 50.0

    def test_too_few_pairs_is_neutral_even_with_std(self):
        # A std estimate from <5 pairs is too noisy → neutral score even when std is present (min-sample guardrail)
        result = SimpleScorer()._score_price(
            _price_data(repeat_sales_return_std=2.0, repeat_sales_count=4)
        )
        assert result["stability_score"] == 50.0
