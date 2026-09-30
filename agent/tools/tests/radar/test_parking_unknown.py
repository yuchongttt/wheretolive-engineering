"""A radar asking for parking must not read "the listing said nothing" as "no parking".

`o.parking` is the listing source's own structured field. On 2026-09-07 it was EMPTY on
32,033 of 71,875 active listings (45%) — and populated with an explicit
"No parking" on only 880. So a blank is UNKNOWN, not a denial, and the old
predicate (`parking IS NOT NULL AND parking != ''`) silently dropped nearly half
the market from any `parking:true` radar.

The user-visible harm: adding "with parking" to a live radar made the reconcile
pass evict listings whose `parking` field was merely blank, silently.

The fix keeps the listing's own field authoritative wherever it spoke and
consults the listing text ONLY where it stayed silent, so no regex can ever overturn an explicit
"No parking". That is zero-false-positive by construction, not by calibration.

Run: pytest tests/radar/test_parking_unknown.py
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import radar_vocab as rv


def _db():
    """Schema copied from the columns the predicate touches (internal note
    "fixture schema hides prod drift": a fixture that invents its own column
    names lets the code under test be wrong and the test still pass)."""
    c = sqlite3.connect(":memory:")
    c.execute("""CREATE TABLE rm_sales_overview (
        id TEXT PRIMARY KEY, parking TEXT, key_features TEXT)""")
    c.execute("""CREATE TABLE floorplan_vlm_results (
        rm_uuid TEXT, ok INT, has_garage INT)""")
    return c


def _matches(conn, value=True):
    where, params, joins = rv.build_where([{"type": "parking", "value": value}])
    return {r[0] for r in conn.execute(
        f"SELECT o.id FROM rm_sales_overview o {joins} WHERE {where}", params)}


def _add(conn, lid, parking=None, key_features=None, garage=None):
    conn.execute("INSERT INTO rm_sales_overview VALUES (?,?,?)", (lid, parking, key_features))
    if garage is not None:
        conn.execute("INSERT INTO floorplan_vlm_results VALUES (?,1,?)", (lid, garage))


class TestListingFieldIsAuthoritativeWhereItSpoke:
    def test_a_stated_parking_type_matches(self):
        c = _db(); _add(c, "yes", parking="Driveway")
        assert _matches(c) == {"yes"}

    def test_an_explicit_no_parking_never_matches(self):
        c = _db(); _add(c, "no", parking="No parking")
        assert _matches(c) == set()

    def test_listing_text_cannot_overturn_an_explicit_no_parking(self):
        # The whole point of consulting text only on silence: an agent blurb
        # mentioning the street's parking must not flip the listing's own "No".
        c = _db()
        _add(c, "no", parking="No parking",
             key_features='["Secure underground parking", "Garage"]', garage=1)
        assert _matches(c) == set()


class TestSilenceIsUnknownNotDenial:
    def test_a_blank_field_with_parking_in_the_key_features_now_matches(self):
        c = _db()
        _add(c, "blank", parking="", key_features='["Lift", "Secure Underground Parking"]')
        assert _matches(c) == {"blank"}

    def test_a_null_field_with_a_driveway_in_the_key_features_now_matches(self):
        c = _db()
        _add(c, "drive", parking=None, key_features='["Driveway", "Three bedrooms"]')
        assert _matches(c) == {"drive"}

    def test_a_blank_field_with_a_floorplan_garage_now_matches(self):
        # Same third source `_garage` has used all along — parking was the one
        # boolean still reading a single column.
        c = _db()
        _add(c, "fp", parking=None, key_features='["Three bedrooms"]', garage=1)
        assert _matches(c) == {"fp"}

    def test_a_blank_field_with_no_evidence_anywhere_still_does_not_match(self):
        # Unknown is not promoted to yes — a radar hit must still be backed by
        # evidence. What changed is only that the evidence may come from text.
        c = _db()
        _add(c, "silent", parking=None, key_features='["Two bedrooms", "Balcony"]')
        assert _matches(c) == set()

    def test_the_listing_this_bug_evicted(self):
        # The evicted listing as it stood in production: blank parking field,
        # and a key-features list that never mentions parking. It is genuinely
        # unknown — so it stays out, but it is now out for a stated reason
        # rather than because a blank was read as a denial. The regression this
        # test guards is the sibling case below.
        c = _db()
        _add(c, "evicted", parking=None,
             key_features='["End of terrace", "Three bedrooms", "Two bathrooms"]')
        _add(c, "sibling", parking=None,
             key_features='["End of terrace", "Off street parking"]')
        assert _matches(c) == {"sibling"}


class TestNegationsInsideTheText:
    def test_key_features_saying_no_parking_does_not_match(self):
        c = _db()
        _add(c, "kfno", parking=None, key_features='["Council Tax Band C", "No Parking"]')
        assert _matches(c) == set()

    def test_key_features_saying_no_allocated_parking_does_not_match(self):
        # Does NOT contain the substring "no parking" — needs its own guard.
        c = _db()
        _add(c, "kfna", parking=None,
             key_features='["Private balcony", "No allocated parking", "Great condition"]')
        assert _matches(c) == set()

    def test_key_features_saying_no_off_street_parking_does_not_match(self):
        c = _db()
        _add(c, "kfos", parking=None, key_features='["No off-street parking"]')
        assert _matches(c) == set()


class TestTheExcludeDirectionIsUnchanged:
    def test_value_false_still_excludes_everything_that_has_parking(self):
        c = _db()
        _add(c, "has", parking="Driveway")
        _add(c, "textual", parking=None, key_features='["Allocated parking space"]')
        _add(c, "none", parking="No parking")
        _add(c, "unknown", parking=None, key_features='["Two bedrooms"]')
        # Absence cannot be proven, so "no parking wanted" keeps the unknowns —
        # the loose direction, which is the safe one for the person asking.
        assert _matches(c, value=False) == {"none", "unknown"}
