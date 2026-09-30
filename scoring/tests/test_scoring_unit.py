"""Unit tests for SimpleScorer core functions."""
import math
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simple_scorer import SimpleScorer


class TestInvNormalCdf:
    """Tests for the inverse normal CDF approximation."""

    def test_median(self):
        """p=0.5 should return 0 (median of standard normal)."""
        assert abs(SimpleScorer._inv_normal_cdf(0.5)) < 0.01

    def test_upper_tail(self):
        """p=0.975 should return ~1.96."""
        result = SimpleScorer._inv_normal_cdf(0.975)
        assert abs(result - 1.96) < 0.02

    def test_lower_tail(self):
        """p=0.025 should return ~-1.96."""
        result = SimpleScorer._inv_normal_cdf(0.025)
        assert abs(result - (-1.96)) < 0.02

    def test_symmetry(self):
        """inv_normal_cdf(p) should be roughly -inv_normal_cdf(1-p)."""
        for p in [0.1, 0.25, 0.75, 0.9]:
            assert abs(SimpleScorer._inv_normal_cdf(p) + SimpleScorer._inv_normal_cdf(1 - p)) < 0.02

    def test_clamping(self):
        """Extreme values should be clamped, not error."""
        SimpleScorer._inv_normal_cdf(0.0)  # should not raise
        SimpleScorer._inv_normal_cdf(1.0)  # should not raise


class TestScoreCommunity:
    """Tests for the _score_community method."""

    @pytest.fixture
    def scorer(self):
        return SimpleScorer.__new__(SimpleScorer)

    def test_low_crime_high_imd(self, scorer, monkeypatch):
        """Low crime + high IMD decile = high score (v9 offline-LSOA path)."""
        # Pin the crime lookup. v9 scores crime from area_intel's London
        # percentile and IGNORES the passed-in total_crimes, so this used to read
        # SW1A1AA's REAL rate — Buckingham Palace, i.e. central Westminster,
        # which is one of the worst — and got 54.8 while asserting > 80. The
        # synthetic "low crime" input had stopped meaning anything.
        import apis.area_intel as area_intel
        monkeypatch.setattr(area_intel, "get_crime_profile",
                            lambda pc: {"resi_rate_pctile": 5, "resi_per_1000": 1.2})
        safety = {"total_crimes": 30, "daytime_pop": 10000}
        demographics = {"imd_decile": 9}
        result = scorer._score_community(safety, "SW1A1AA", demographics)
        assert result["crime_score"] > 90  # 100 - 5th percentile = 95
        assert result["score"] > 80
        assert result["imd_score"] == 90  # decile 9 → 10 + 8*10 = 90

    def test_low_crime_high_imd_on_the_api_fallback(self, scorer, monkeypatch):
        """Same brief when area_intel has no row — the legacy per-capita formula.

        Kept because the fallback is live code: _score_community drops to it for
        non-London postcodes and whenever the offline lookup raises.
        """
        import apis.area_intel as area_intel
        monkeypatch.setattr(area_intel, "get_crime_profile", lambda pc: None)
        safety = {"total_crimes": 30, "daytime_pop": 10000}
        result = scorer._score_community(safety, "SW1A1AA", {"imd_decile": 9})
        assert result["crime_score"] > 90  # rate 3.0/1000 → 95
        assert result["score"] > 80

    def test_high_crime_low_imd(self, scorer):
        """High crime + low IMD decile = low score."""
        # Scorer uses its own area-type based daytime_pop (~45k for urban),
        # so need many crimes to push crime_rate high
        safety = {"total_crimes": 800}
        demographics = {"imd_decile": 2}
        result = scorer._score_community(safety, "E1 6AN", demographics)
        assert result["score"] < 40

    def test_no_imd_data(self, scorer):
        """Missing IMD should redistribute weight to crime only."""
        safety = {"total_crimes": 30, "daytime_pop": 10000}
        result = scorer._score_community(safety, "SW1A1AA", None)
        assert result["imd_score"] is None
        # Score should equal crime_score when IMD is missing
        assert abs(result["score"] - result["crime_score"]) < 0.2

    def test_imd_decile_mapping(self, scorer):
        """IMD decile 1 → 10, decile 10 → 100."""
        safety = {"total_crimes": 50, "daytime_pop": 10000}
        r1 = scorer._score_community(safety, "E1 6AN", {"imd_decile": 1})
        r10 = scorer._score_community(safety, "E1 6AN", {"imd_decile": 10})
        assert r1["imd_score"] == 10
        assert r10["imd_score"] == 100

    def test_weights(self, scorer):
        """Crime 40% + IMD 60% weighted average."""
        safety = {"total_crimes": 30, "daytime_pop": 10000}
        demographics = {"imd_decile": 5}
        result = scorer._score_community(safety, "SW1A1AA", demographics)
        expected = result["crime_score"] * 0.4 + result["imd_score"] * 0.6
        assert abs(result["score"] - round(expected, 1)) < 0.2


class TestScoreEnvironment:
    """Tests for the _score_environment method."""

    @pytest.fixture
    def scorer(self):
        return SimpleScorer.__new__(SimpleScorer)

    def test_all_components(self, scorer):
        """All 4 components present → equal 25% weights."""
        result = scorer._score_environment(
            {"score": 80}, {"score": 60}, {"score": 40}, {"green_score": 100}
        )
        assert result is not None
        expected = (80 + 60 + 40 + 100) / 4  # 70
        assert abs(result["score"] - expected) < 0.2
        assert result["noise_score"] == 80
        assert result["flood_score"] == 60
        assert result["air_score"] == 40
        assert result["parks_score"] == 100

    def test_missing_component(self, scorer):
        """Missing component redistributes weight to remaining."""
        result = scorer._score_environment(
            {"score": 80}, {"score": 60}, None, {"green_score": 100}
        )
        expected = (80 + 60 + 100) / 3  # 80
        assert abs(result["score"] - expected) < 0.2
        assert result["air_score"] is None

    def test_all_missing(self, scorer):
        """No data → None."""
        result = scorer._score_environment(None, None, None, None)
        assert result is None

    def test_single_component(self, scorer):
        """Single component → full weight."""
        result = scorer._score_environment({"score": 75}, None, None, None)
        assert abs(result["score"] - 75) < 0.2

    def test_detail_uses_english_tokens(self, scorer):
        """detail is a language-neutral English breakdown (like transport's
        "Commute 64 + Transit 96"); pages localize it via translateLegacyDetail.
        It used to be hard-coded Chinese ("噪音80 + 洪水60 + ...")."""
        result = scorer._score_environment(
            {"score": 80.4}, {"score": 60}, None, {"green_score": 99.6})
        assert result["detail"] == "Noise 80 + Flood 60 + Parks 100"
        assert not any("\u4e00" <= ch <= "\u9fff" for ch in result["detail"])
