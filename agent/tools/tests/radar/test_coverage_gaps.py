"""Coverage blind spots must be said out loud, not disguised as "nothing on the market" (2026-08-25).

Two silent false zeros:
  1. a commute condition does `LEFT JOIN sector_hub_commute ... WHERE minutes
     <= ?`; a sector with no row → minutes is NULL → predicate false → the
     listing is dropped before commute_verify ever runs. Measured: 2-bed houses
     ≤350k without a commute filter gave DA=46 / KT=7 / WD=2; adding "King's
     Cross ≤90min" turned all three to 0 — while Dartford to London Bridge is
     really ~40 minutes. sector_hub_commute simply had no KT/DA/WD rows.
  2. _area doesn't coverage-check outcodes: "AL1"/"HP1"/"TN13" compile, the
     radar is created, and it matches 0 forever — the user can't tell "we have
     no data" from "nothing on the market".

Same principle as the search_properties multi-area fan-out rule "silent
truncation makes 'never searched' look like 'searched, found nothing'", this
time applied to commute and area coverage.
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import radar_vocab as rv  # noqa: E402


def _conn():
    """Four homes + a sector_hub_commute table covering only some sectors.

    SE1 2AB — has a King's Cross row, 20 min (within the limit)
    E14 5AB — has a King's Cross row, 70 min (over the limit: genuinely
              filtered, not a blind spot)
    DA1 1AA — no row at all in sector_hub_commute (blind spot, currently
              dropped silently)
    DA2 2BB — also a blind spot, but £900k is over budget, so it must not be
              counted in the blind-spot total
    """
    c = sqlite3.connect(":memory:")
    c.execute("""CREATE TABLE rm_sales_overview (
        id TEXT PRIMARY KEY, postcode TEXT, postcode_norm TEXT, asking_price INT,
        bedrooms INT, property_type TEXT, delisted_date TEXT, canonical_id TEXT)""")
    rows = [
        ("1", "SE1 2AB", 500000),
        ("2", "E14 5AB", 500000),
        ("3", "DA1 1AA", 500000),
        ("4", "DA2 2BB", 900000),
    ]
    for pid, pc, price in rows:
        c.execute("INSERT INTO rm_sales_overview VALUES (?,?,?,?,2,'house',NULL,NULL)",
                  (pid, pc, pc.replace(" ", ""), price))
    c.execute("CREATE TABLE sector_hub_commute (sector TEXT, hub TEXT, minutes INT)")
    c.execute("INSERT INTO sector_hub_commute VALUES ('SE1 2', \"King's Cross\", 20)")
    c.execute("INSERT INTO sector_hub_commute VALUES ('E14 5', \"King's Cross\", 70)")
    c.commit()
    return c


def _count(conn, conditions):
    where, params, join = rv.build_where(conditions, conn=conn)
    sql = f"SELECT COUNT(*) FROM rm_sales_overview o {join} WHERE {where}"
    return conn.execute(sql, params).fetchone()[0]


# --- 1. commute blind spot --------------------------------------------------

def test_commute_gap_counts_listings_with_no_row_for_that_hub():
    """The blind-spot count only counts listings with "no data for this hub", not ones genuinely over time."""
    conn = _conn()
    conds = [{"type": "price", "max": 600000},
             {"type": "commute", "hub": "King's Cross", "max_minutes": 60}]
    # status quo: DA1 is silently dropped, the query keeps only SE1
    assert _count(conn, conds) == 1

    gap = rv.commute_coverage_gap(conds, conn)
    assert gap is not None
    assert gap["hub"] == "King's Cross"
    # DA1 is the blind spot; E14 is genuinely over time (70>60), DA2 genuinely over budget (£900k>£600k)
    assert gap["n"] == 1
    assert gap["areas"] == [("DA", 1)]


def test_commute_gap_is_none_when_spec_has_no_commute_condition():
    conn = _conn()
    assert rv.commute_coverage_gap([{"type": "price", "max": 600000}], conn) is None


def test_commute_gap_zero_when_every_candidate_sector_is_covered():
    """With full coverage, never report a false blind spot — otherwise every radar carries a useless warning."""
    conn = _conn()
    conds = [{"type": "area", "value": ["SE1", "E14"]},
             {"type": "commute", "hub": "King's Cross", "max_minutes": 60}]
    gap = rv.commute_coverage_gap(conds, conn)
    assert gap is not None and gap["n"] == 0 and gap["areas"] == []


# --- 2. area coverage -------------------------------------------------------

def test_areas_outside_london_postal_districts_are_flagged():
    """The Herts/Bucks/Kent/Surrey areas the user named are outside our postal-area coverage."""
    assert rv.out_of_coverage_areas(["AL1", "HP1", "TN13", "GU2", "E14"]) == [
        "AL1", "HP1", "TN13", "GU2"]


def test_london_postal_districts_in_the_home_counties_are_covered():
    """County borders and postal borders don't coincide: EN6 is Hertfordshire,
    KT6 Surrey, DA1 Kent — but all three are in the London postal areas with
    full data, and must not be blocked as blind spots."""
    assert rv.out_of_coverage_areas(["EN6", "WD17", "KT6", "DA1", "BR6"]) == []


def test_bare_area_prefix_is_checked_too():
    assert rv.out_of_coverage_areas(["SW", "TN"]) == ["TN"]
