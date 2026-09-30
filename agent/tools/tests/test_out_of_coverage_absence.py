"""A "not found" outside our coverage boundary must be reported as a coverage gap, never as a fact.

A user asked about a postcode in a postal area outside our coverage (ME). lookup_address answered

    Land Registry: no sales recorded in <postcode>.
    No active listing in <postcode> right now (checked live listing data).

Both lines assert "there's nothing there", while the truth is that **we hold
not a single LR row or listing for the whole ME area** (0 rows starting with
ME in lr_transactions). The model relayed it to a homeowner as "**No Land
Registry sale on record** for this address" — which, to a homeowner, means
"this home has never sold". get_comparables / get_price_trend had the same
disease:

    No Land Registry transactions found near <postcode> in the last 3 years.
    ME7 has no price analysis yet ... or suggest /evaluate?postcode=ME7

The last one even sent the user to a page that is bound to be empty for a
Kent postcode.

search_properties already separated the two cases (_area_coverage_note); this
adds the same guard to the three address/price tools.

Pinned:
  1. outside coverage → a NO COVERAGE block, and no bare absence sentence like
     "no sales recorded / no transactions found";
  2. genuinely empty INSIDE coverage (E/EC/N/NW/SE/SW/W/WC + BR/CR/DA/EN/HA/
     IG/KT/RM/SM/TW/UB/WD) → not a word added, original wording unchanged;
  3. county borders ≠ postal borders: WD/KT/BR/DA/EN are covered and must not
     be misjudged as blind spots;
  4. outside coverage, never push /evaluate (that page is just as empty for it).

(Extract note: the original file also covers lookup_address, and get_price_trend
against the production database; neither is part of this extract. The
out-of-coverage example postcode is a placeholder, not the user's.)
"""
import asyncio
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import coverage_note as cn  # noqa: E402


# --------------------------------------------------------------------------
# 1. pure-function layer
# --------------------------------------------------------------------------
class TestPostalArea:
    def test_strips_digits_and_space(self):
        assert cn.postal_area("ME7 0ZZ") == "ME"
        assert cn.postal_area("me70zz") == "ME"
        assert cn.postal_area("E14 9AA") == "E"
        assert cn.postal_area("EC1V 0HB") == "EC"
        assert cn.postal_area("WD17") == "WD"

    def test_unparseable_is_empty(self):
        assert cn.postal_area("") == ""
        assert cn.postal_area(None) == ""
        assert cn.postal_area("12345") == ""


class TestIsCovered:
    @pytest.mark.parametrize("pc", ["E14 9AA", "SW11 3RA", "EC1V 0HB", "N19",
                                    "HA1 1AA", "WD17 1AB", "KT6 4AA",
                                    "BR1 1AA", "DA1 1AA", "EN6 1AA"])
    def test_inside(self, pc):
        assert cn.is_covered(pc) is True

    @pytest.mark.parametrize("pc", ["ME7 0ZZ", "CT1 1AA", "TN1 1AA", "AL1 1AA",
                                    "HP1 1AA", "M1 1AA", "B1 1AA", "GU1 1AA"])
    def test_outside(self, pc):
        assert cn.is_covered(pc) is False

    def test_unparseable_is_treated_as_covered(self):
        """Don't brand an unrecognisable postal area a blind spot — better to stay silent."""
        assert cn.is_covered("") is True
        assert cn.is_covered("not a postcode") is True


class TestNote:
    def test_covered_postcode_gets_nothing(self):
        assert cn.out_of_coverage_note("HA1 1AA", missing="Land Registry sales") is None
        assert cn.out_of_coverage_note("WD17 1AB", missing="Land Registry sales") is None

    def test_uncovered_note_names_area_and_forbids_absence_claims(self):
        note = cn.out_of_coverage_note("ME7 0ZZ", missing="Land Registry sales")
        assert note is not None
        assert "NO COVERAGE" in note
        assert "ME" in note
        assert "Land Registry sales" in note
        # explicitly forbids stating the gap as fact
        assert "never sold" in note.lower()
        # the county-border ≠ postal-border reminder must be there, or the model treats KT/DA as blind spots too
        assert "KT" in note and "DA" in note
        # a way out for the user
        assert "landregistry.data.gov.uk" in note

    def test_still_valid_clause_is_appended_when_given(self):
        note = cn.out_of_coverage_note(
            "ME7 0ZZ", missing="Land Registry sales",
            still_valid="EPC and the VOA council-tax register are national.")
        assert "EPC and the VOA council-tax register are national." in note


# --------------------------------------------------------------------------
# 2. get_comparables
# --------------------------------------------------------------------------
import get_comparables as gc  # noqa: E402


def _comps_db(tmp_path):
    db = tmp_path / "evaluations.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE lr_transactions (postcode TEXT, paon TEXT, saon TEXT, "
        "street TEXT, price INT, date TEXT, property_type TEXT, tenure TEXT, "
        "new_build TEXT, category TEXT)"
    )
    conn.execute("CREATE TABLE rm_sales_overview (id TEXT, postcode TEXT,"
                 " bedrooms INT, bathrooms INT, asking_price INT,"
                 " property_type TEXT, address TEXT)")
    conn.commit()
    conn.close()
    return db


def _comps(tmp_path, monkeypatch, postcode):
    monkeypatch.setattr(gc, "DB_PATH", _comps_db(tmp_path))
    out = asyncio.run(gc.handle_call_tool("get_comparables", {"postcode": postcode}))
    return out[0].text


def test_comparables_out_of_coverage_says_coverage(tmp_path, monkeypatch):
    txt = _comps(tmp_path, monkeypatch, "ME7 0ZZ")
    assert "NO COVERAGE" in txt


def test_comparables_inside_coverage_empty_stays_plain(tmp_path, monkeypatch):
    txt = _comps(tmp_path, monkeypatch, "HA1 1AA")
    assert "NO COVERAGE" not in txt
    assert "No Land Registry transactions found" in txt
