"""get_area_overview truncated an outcode input to a single letter.

Real case (2026-09-03, a real UI session comparing SW11 with SE18): the model called get_area_overview({"postcode":
"SW11"}) and got back

    # Area overview: SW11 (outcode S)
    _No cached evaluation for SW11 yet._
    "active_listings": {"count": 0, ...}

SW11 had 1,245 active listings at the time. Root cause identical to the
compare_postcodes one:
    outcode = pc.split(" ")[0] if " " in pc else pc[:-3]
"SW11" has no space, so pc[:-3] is "S". The active-listings SQL itself was
right (it derives the outcode from the stored postcode and compares), it was
just handed "S", hence zero matches.

The zero was then rendered as "count: 0" — **a false claim of absence**, the
same harm as compare_postcodes' "Not yet scored": the model could easily
repeat it as "SW11 has nothing on the market".

Contract (same source as area_scope.classify_postcode_scope — no second caliber):
  - outcode / sector / unit input all derive the correct outcode;
  - a sector's internal space is load-bearing; "SW11 1" must not become some
    other place;
  - unit_postcode_known only makes sense for a full unit postcode; area input
    must not trigger it (asked about the whole WD area it returns False, yet
    those postcodes are all live).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import get_area_overview as gao  # noqa: E402


class TestOutcodeDerivation:
    def test_outcode_input_keeps_all_four_characters(self):
        scope, pc, oc = gao._scope_and_outcode("SW11")
        assert oc == "SW11", f"got {oc!r} — the bug produced 'S'"
        assert scope == "outcode"

    def test_single_letter_outcode_survives(self):
        _scope, _pc, oc = gao._scope_and_outcode("N1")
        assert oc == "N1"

    def test_unit_postcode_still_derives_its_outcode(self):
        scope, pc, oc = gao._scope_and_outcode("SW11 3RA")
        assert (scope, pc, oc) == ("unit", "SW11 3RA", "SW11")

    def test_unspaced_unit_postcode(self):
        scope, pc, oc = gao._scope_and_outcode("sw113ra")
        assert (scope, pc, oc) == ("unit", "SW11 3RA", "SW11")

    def test_sector_is_not_mangled_into_another_place(self):
        """The internal space is load-bearing: "SW11 1" is a sector, not "SW1 11" and not "SW 111"."""
        scope, pc, oc = gao._scope_and_outcode("SW11 1")
        assert (scope, oc) == ("sector", "SW11")
        assert pc == "SW11 1"


class TestUnknownPostcodeAlertScope:
    def test_area_scopes_do_not_ask_the_unit_oracle(self):
        """unit_postcode_known returns False for the whole WD area, yet those
        postcodes are all live — asking it about area input only yields a false alarm."""
        assert gao._should_check_unit_known("outcode") is False
        assert gao._should_check_unit_known("sector") is False
        assert gao._should_check_unit_known("unit") is True
