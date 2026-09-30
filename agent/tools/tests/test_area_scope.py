"""An outcode-level area question must resolve to the SAME answer every time.

Asked "is SW11 Battersea worth buying? look at it across the five dimensions"
(in Chinese) three times in four hours, chat
returned Schools 84.2, then 60.4, then 59.8 — because nothing resolves an
outcode, so the model improvised a different representative unit postcode each
run (SW11 3RA / SW11 4EE / SW11 8EQ). Every number was true of its own point
and none was an answer about SW11: across the 703 scored units in that outcode
the schools score spans 33.6 to 98.0.

The fix is to answer an outcode as an outcode — a deterministic aggregate over
every scored unit inside it, carrying the spread so "SW11 varies a lot
internally" can be said with numbers instead of being hidden behind one
arbitrary point.

Run: pytest tests/test_area_scope.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from area_scope import classify_postcode_scope, like_pattern, summarise_dimension


class TestClassifyPostcodeScope:
    def test_bare_outcode_is_an_outcode(self):
        assert classify_postcode_scope("SW11") == ("outcode", "SW11")

    def test_single_letter_outcode_is_an_outcode(self):
        assert classify_postcode_scope("E14") == ("outcode", "E14")

    def test_sector_keeps_its_single_trailing_digit(self):
        assert classify_postcode_scope("SW11 3") == ("sector", "SW11 3")

    def test_full_unit_postcode_is_a_unit(self):
        assert classify_postcode_scope("SW11 3RA") == ("unit", "SW11 3RA")

    def test_unspaced_unit_postcode_is_normalised(self):
        assert classify_postcode_scope("sw113ra") == ("unit", "SW11 3RA")

    def test_lowercase_and_padded_outcode_is_normalised(self):
        assert classify_postcode_scope("  sw11 ") == ("outcode", "SW11")

    def test_outcode_with_a_letter_district_is_an_outcode(self):
        # EC1V / W1A style districts end in a letter, not a digit.
        assert classify_postcode_scope("EC1V") == ("outcode", "EC1V")

    def test_full_ec_postcode_is_a_unit(self):
        assert classify_postcode_scope("EC1V 0HB") == ("unit", "EC1V 0HB")


class TestLikePattern:
    def test_outcode_pattern_matches_only_that_outcode(self):
        # "SW1 %" must not swallow SW11/SW19 — the space is what bounds it.
        assert like_pattern("outcode", "SW1") == "SW1 %"

    def test_sector_pattern_matches_units_in_that_sector(self):
        assert like_pattern("sector", "SW11 3") == "SW11 3%"

    def test_unit_pattern_is_exact(self):
        assert like_pattern("unit", "SW11 3RA") == "SW11 3RA"


class TestSummariseDimension:
    def test_median_of_an_odd_number_of_scores(self):
        assert summarise_dimension([10.0, 30.0, 20.0])["median"] == 20.0

    def test_median_of_an_even_number_of_scores_averages_the_middle_pair(self):
        assert summarise_dimension([10.0, 20.0, 30.0, 40.0])["median"] == 25.0

    def test_carries_the_spread_so_internal_variation_is_visible(self):
        s = summarise_dimension([33.6, 60.4, 98.0])
        assert (s["min"], s["max"], s["n"]) == (33.6, 98.0, 3)

    def test_ignores_missing_scores_rather_than_treating_them_as_zero(self):
        # A unit postcode with no schools score must not drag the median down.
        assert summarise_dimension([80.0, None, 60.0])["median"] == 70.0
        assert summarise_dimension([80.0, None, 60.0])["n"] == 2

    def test_returns_none_when_no_scores_are_available(self):
        assert summarise_dimension([None, None]) is None

    def test_is_order_independent_so_row_order_cannot_change_the_answer(self):
        a = summarise_dimension([98.0, 33.6, 60.4, 71.2])
        b = summarise_dimension([60.4, 71.2, 98.0, 33.6])
        assert a == b
