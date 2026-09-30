"""The Price dimension must arrive with its definition attached.

2026-09-03: chat was asked to compare Catford (SE6) with Dulwich (SE21/SE22).
Every price and score it quoted re-verified exactly — except it read the Price
dimension as "value for money", telling the reader that "although Dulwich is
expensive, relative to its transport/schools/environment the system judges its
pricing to be good value". That is the opposite of what the score measures:
simple_scorer._score_price weights price LEVEL at 35%, the more expensive the
better, so Dulwich outscores Catford on Price largely BECAUSE it is more
expensive.

The tool handed the model five bare dimension names and no definitions, so
the model supplied one. A number that is right and a meaning that is invented
still reaches the reader as a false claim. Attach the meanings to the output.

Run: pytest tests/test_postcode_scores_dimension_legend.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import get_postcode_scores as gps


AREA_SUMMARY = {
    "scope": "outcode",
    "area": "SE21",
    "scored_unit_postcodes": 130,
    "dimensions": {
        "Transport": {"median": 49.4, "min": 42.1, "max": 66.5, "n": 130},
        "Community": {"median": 71.6, "min": 54.5, "max": 81.4, "n": 130},
        "Environment": {"median": 67.4, "min": 25.5, "max": 93.7, "n": 130},
        "Price": {"median": 74.4, "min": 36.5, "max": 98.0, "n": 130},
        "Schools": {"median": 60.2, "min": 50.7, "max": 98.0, "n": 130},
    },
    "overall": {"median": 64.5, "min": 41.3, "max": 98.0, "n": 130},
    "median_price": {"median": 576050.0, "min": 145000.0, "max": 4000000.0, "n": 130},
}


class TestLegendContent:
    def test_price_says_dearer_scores_higher(self):
        legend = gps.DIMENSION_LEGEND
        assert "35%" in legend
        assert "higher" in legend.lower()

    def test_price_disowns_the_value_for_money_reading(self):
        legend = gps.DIMENSION_LEGEND.lower()
        assert "not" in legend
        assert "value for money" in legend
        assert "affordab" in legend

    def test_every_dimension_is_defined(self):
        for label in ("Transport", "Community", "Environment", "Price", "Schools"):
            assert f"{label}:" in gps.DIMENSION_LEGEND

    def test_weights_match_the_scorer(self):
        # simple_scorer: transport = commute 60% + transit 40%;
        # community = crime 40% + IMD 60%; price level 35% / long-run CAGR 30%
        # / 3Y momentum 20% / stability 10% / activity 5%.
        legend = gps.DIMENSION_LEGEND
        for token in ("60%", "40%", "35%", "30%", "20%", "10%", "5%"):
            assert token in legend


class TestLegendIsAttachedToOutput:
    def test_area_render_carries_the_legend(self):
        out = gps._render_area(AREA_SUMMARY)
        assert gps.DIMENSION_LEGEND in out

    def test_area_render_still_carries_the_scores(self):
        out = gps._render_area(AREA_SUMMARY)
        assert "Price: 74.4/100" in out
        assert "Median sold price: £576,050" in out

    def test_legend_precedes_the_structured_trailer(self):
        out = gps._render_area(AREA_SUMMARY)
        assert out.index(gps.DIMENSION_LEGEND) < out.index("--- structured ---")


class TestUnitScopeToo:
    """The unit-postcode path renders its own line list — same exposure."""

    def test_unit_render_carries_the_legend(self):
        out = gps._render_unit({
            "postcode": "SE21 7BQ",
            "dimensions": {},
            "calibrated": {
                "Transport": 49.4, "Community": 71.6, "Environment": 67.4,
                "Price": 74.4, "Schools": 60.2, "Overall": 64.5,
                "data_completeness": 1.0, "as_of": "2026-08-30",
            },
        })
        assert gps.DIMENSION_LEGEND in out
        assert "Price: 74.4/100" in out
