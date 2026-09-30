"""A radar asking for a garden must not read "the listing said nothing" as "no garden".

`o.garden` is the listing source's structured outdoor-space field ("Yes" / "Private
garden" / "Rear garden" / "Communal garden" / "Terrace" / "Patio" / blank). On
2026-09-07 it was BLANK on 32,391 of 72,248 active listings — 44.8%, the same
shape as `parking` (44.5%) and the same bug: the old single-source predicate
dropped every one of them from a `garden:true` radar.

WHY THERE IS NO TEXT FALLBACK HERE, unlike parking
--------------------------------------------------
The obvious second source is key_features, and it looked promising (753 blank
listings mention "garden"). Measured against an independent judge — the
floorplan VLM's own has_garden read — it is mostly noise:

    floorplan says NO garden   28,560 rows   text fires on 4,009  (14.0%)
    floorplan says HAS garden   1,545 rows   text fires on   399  (25.8%)

4,009 false positives against 399 true ones. London is full of streets and parks
called "… Gardens", listings advertise "views over the communal gardens" and
"close to Kensington Gardens", and none of that is a garden the buyer gets. A
stricter phrase set (private/rear/own/landscaped garden, communal excluded) got
to 0.23% vs 7.96% — better, but still only ~65% precision and adding just 65
listings the floorplan did not already have. Not worth the false claims.

So the one added source is the floorplan read: independent, purpose-built,
already trusted by `_garage`/`_terrace`/`_fireplace`, and it recovers 1,545
listings on its own. If someone later wants the text source back, re-run the
measurement above first — the intuition that "garden" in the text means a garden
is exactly what the numbers refute.

Run: pytest tests/radar/test_garden_unknown.py
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import radar_vocab as rv


def _db():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE rm_sales_overview (id TEXT PRIMARY KEY, garden TEXT, key_features TEXT)")
    c.execute("CREATE TABLE floorplan_vlm_results (rm_uuid TEXT, ok INT, has_garden INT)")
    return c


def _matches(conn, value=True):
    where, params, joins = rv.build_where([{"type": "garden", "value": value}])
    return {r[0] for r in conn.execute(
        f"SELECT o.id FROM rm_sales_overview o {joins} WHERE {where}", params)}


def _add(conn, lid, garden=None, key_features=None, fp=None, fp_ok=1):
    conn.execute("INSERT INTO rm_sales_overview VALUES (?,?,?)", (lid, garden, key_features))
    if fp is not None:
        conn.execute("INSERT INTO floorplan_vlm_results VALUES (?,?,?)", (lid, fp_ok, fp))


class TestListingFieldIsAuthoritativeWhereItSpoke:
    def test_the_explicit_yes_flag_matches(self):
        c = _db(); _add(c, "y", garden="Yes")
        assert _matches(c) == {"y"}

    def test_any_stated_value_containing_garden_matches(self):
        # Unchanged semantics, communal included — this predicate has always
        # meant "some garden", and narrowing it to private is a separate call.
        c = _db()
        _add(c, "private", garden="Private garden")
        _add(c, "communal", garden="Communal garden")
        assert _matches(c) == {"private", "communal"}

    def test_a_stated_terrace_or_patio_alone_is_not_a_garden(self):
        c = _db()
        _add(c, "terrace", garden="Terrace")
        _add(c, "patio", garden="Patio")
        assert _matches(c) == set()

    def test_a_floorplan_cannot_overturn_a_stated_terrace_only(self):
        # The listing answered the question; the floorplan is only consulted on
        # silence, so a VLM read cannot promote a terrace-only listing.
        c = _db(); _add(c, "terrace", garden="Terrace", fp=1)
        assert _matches(c) == set()


class TestSilenceIsUnknownNotDenial:
    def test_a_blank_field_with_a_floorplan_garden_now_matches(self):
        c = _db(); _add(c, "blank", garden="", fp=1)
        assert _matches(c) == {"blank"}

    def test_a_null_field_with_a_floorplan_garden_now_matches(self):
        c = _db(); _add(c, "null", garden=None, fp=1)
        assert _matches(c) == {"null"}

    def test_a_blank_field_whose_floorplan_shows_no_garden_does_not_match(self):
        c = _db(); _add(c, "nofp", garden=None, fp=0)
        assert _matches(c) == set()

    def test_a_blank_field_with_no_floorplan_at_all_does_not_match(self):
        # Unknown is not promoted to yes; a radar hit still needs evidence.
        c = _db(); _add(c, "silent", garden=None)
        assert _matches(c) == set()

    def test_a_failed_floorplan_read_is_not_evidence(self):
        c = _db(); _add(c, "badfp", garden=None, fp=1, fp_ok=0)
        assert _matches(c) == set()


class TestTheTextSourceStaysOut:
    def test_key_features_mentioning_a_garden_is_not_enough_on_its_own(self):
        # 14% false-positive rate against the floorplan judge — see the module
        # docstring. These three are the actual shapes that made it noise.
        c = _db()
        _add(c, "park", garden=None,
             key_features='["Close to the high street and Kensington Gardens"]')
        _add(c, "views", garden=None, key_features='["Views over maintained gardens"]')
        _add(c, "street", garden=None, key_features='["Two bedrooms", "Elgin Gardens"]')
        assert _matches(c) == set()


class TestTheExcludeDirectionIsUnchanged:
    def test_value_false_keeps_the_unknowns(self):
        # Absence cannot be proven, so "no garden wanted" keeps what we don't
        # know — the loose direction, which is the safe one for that asker.
        c = _db()
        _add(c, "has", garden="Yes")
        _add(c, "fp", garden=None, fp=1)
        _add(c, "unknown", garden=None)
        _add(c, "terrace", garden="Terrace")
        assert _matches(c, value=False) == {"unknown", "terrace"}
