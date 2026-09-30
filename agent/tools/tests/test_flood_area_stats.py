"""Area-level EA flood aggregation (R2-4 Fix W).

Background: in R2-4 the reference agent warned that one side of a local park
is a flood storage area; our DB could prove it, and more precisely (every point
on one side is Zone 1, every FZ3 hit is in the neighbouring outcode) — but
get_area_profile / compare_postcodes had no area-aggregate output, so this
unique card could never be played. "Area-level aggregation" had been flagged as
a leftover since R2-1; this made it a real item.

Contract:
  1. outcode aggregate: FZ3/FZ2 counts and shares over active + canonical +
     stamped rows;
  2. coverage threshold: n_checked < 20 returns None (thin coverage → no number);
  3. unstamped rows are not in the denominator; delisted / duplicate (canonical)
     rows excluded;
  4. the sector argument narrows to that sector;
  5. rendering discipline (same source as fetch_listing's flood lines): rivers
     & sea only, defences ignored, surface water not assessed, and zero hits
     must still be stated (checked-and-clean is information).
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flood_area import flood_area_stats, flood_area_lines  # noqa: E402


_SEQ = [0]


def _db(tmp_path, rows):
    _SEQ[0] += 1
    path = tmp_path / f"evaluations{_SEQ[0]}.db"
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE rm_sales_overview (
        id TEXT PRIMARY KEY, postcode TEXT, flood_zone TEXT,
        flood_checked_at TEXT, delisted_date TEXT, canonical_id TEXT,
        postcode_norm TEXT GENERATED ALWAYS AS (upper(replace(postcode,' ',''))) VIRTUAL)""")
    for i, r in enumerate(rows):
        conn.execute("INSERT INTO rm_sales_overview (id,postcode,flood_zone,flood_checked_at,delisted_date,canonical_id) VALUES (?,?,?,?,?,?)",
                     (str(1000+i), r.get("pc", "TW9 1AA"), r.get("fz"),
                      r.get("checked", "2026-08-01"), r.get("delisted"), r.get("canon")))
    conn.commit(); conn.close()
    return path


def test_outcode_shares_and_counts(tmp_path):
    rows = ([dict(pc="TW10 7AA", fz="FZ3")] * 5 + [dict(pc="TW10 7AB", fz="FZ2")]
            + [dict(pc="TW10 6AA")] * 20)
    st = flood_area_stats(_db(tmp_path, rows), outcode="TW10")
    assert st["n_checked"] == 26 and st["fz3_n"] == 5 and st["fz2_n"] == 1
    assert abs(st["fz3_share"] - 5 / 26) < 1e-9


def test_below_threshold_returns_none(tmp_path):
    st = flood_area_stats(_db(tmp_path, [dict(pc="N6 5AA")] * 5), outcode="N6")
    assert st is None


def test_unchecked_delisted_dup_excluded(tmp_path):
    rows = ([dict(pc="TW9 2AA")] * 20
            + [dict(pc="TW9 2AA", fz="FZ3", checked=None)]      # not stamped
            + [dict(pc="TW9 2AA", fz="FZ3", delisted="2026-01-01")]  # delisted
            + [dict(pc="TW9 2AA", fz="FZ3", canon="999")])       # duplicate pointing elsewhere
    st = flood_area_stats(_db(tmp_path, rows), outcode="TW9")
    assert st["n_checked"] == 20 and st["fz3_n"] == 0


def test_sector_filter(tmp_path):
    rows = ([dict(pc="TW10 7AA", fz="FZ3")] * 21 + [dict(pc="TW10 6AA")] * 21)
    st = flood_area_stats(_db(tmp_path, rows), outcode="TW10", inward1="7")
    assert st["n_checked"] == 21 and st["fz3_n"] == 21
    assert st["scope"] == "TW10 7"


def test_outcode_never_bleeds_into_longer_outcode(tmp_path):
    """Review (both angles) #1, confirmed: in compact form E1's sector "E1 6"
    and outcode E16 are the same string — prefix matching pinned the Royal
    Docks' 91% FZ3 on Whitechapel. Fix: outcode equality (range + length) +
    a separate filter on the inward first digit; no more concatenated sector
    strings."""
    rows = ([dict(pc="E14 9AA", fz="FZ3")] * 25      # Isle of Dogs, all FZ3
            + [dict(pc="E1 6AN")] * 25)               # Whitechapel, clean
    db = _db(tmp_path, rows)
    st_e1 = flood_area_stats(db, outcode="E1")
    assert st_e1["n_checked"] == 25 and st_e1["fz3_n"] == 0      # doesn't absorb E14
    st_e1_6 = flood_area_stats(db, outcode="E1", inward1="6")
    assert st_e1_6["n_checked"] == 25 and st_e1_6["fz3_n"] == 0
    assert st_e1_6["scope"] == "E1 6"
    st_e14 = flood_area_stats(db, outcode="E14")
    assert st_e14["n_checked"] == 25 and st_e14["fz3_n"] == 25


def test_db_open_failure_degrades_to_none(tmp_path):
    """review (line-by-line) #5: connect was outside the try, so a missing DB
    file would blow up the whole get_area_profile call (its demographics/crime/
    rent come from another DB and were already fetched). Must degrade to None."""
    assert flood_area_stats(tmp_path / "nope" / "x.db", outcode="TW9") is None


def test_render_wording_discipline(tmp_path):
    hit = flood_area_stats(_db(tmp_path, [dict(pc="TW10 7AA", fz="FZ3")] * 21), outcode="TW10")
    clean = flood_area_stats(_db(tmp_path, [dict(pc="TW9 2AA")] * 30), outcode="TW9")
    txt = "\n".join(flood_area_lines([hit, clean]))
    assert "defences" in txt            # defences not counted
    assert "rivers & sea" in txt        # dataset scope stated
    assert "surface-water" in txt       # surface water not assessed
    assert "TW9" in txt and "TW10" in txt
    # the clean side needs an explicit statement too (checked, and not in FZ2/3)
    assert "outside" in txt or "0 of 30" in txt or "Zone 1" in txt


def test_footer_forbids_explaining_a_zone_by_flood_defences(tmp_path):
    """The model reaches for flood defences to "explain" zones, and often gets the direction backwards.

    2026-09-03 case (a real session comparing SW11 with SE18): after reading SW11 80% FZ3 / SE18 13%, the agent improvised
    "the Thames Barrier mainly protects eastwards; the Battersea stretch isn't
    protected by it". Wrong on both counts: the barrier sits at Charlton /
    Woolwich Reach and, when closed, stops the surge travelling **upstream**,
    protecting the upstream (west) side — Battersea included; Royal Arsenal is
    downstream (east) of the barrier and is exactly the part NOT protected.

    That claim is about something the dataset explicitly says it doesn't count
    (zones ignore defences), and the per-listing wording even says "maps FZ3
    behind the Thames Barrier" — the model turned its own tool's words
    backwards. The area-level footer only said "ignore defences"; it didn't
    forbid explaining by defences, so it couldn't stop this.
    """
    hit = flood_area_stats(_db(tmp_path, [dict(pc="TW10 7AA", fz="FZ3")] * 21), outcode="TW10")
    txt = "\n".join(flood_area_lines([hit])).lower()
    assert "do not explain" in txt or "never explain" in txt, txt
    # the correct direction must be written in too, or the model merely "doesn't explain" and gets it backwards next time in other words
    assert "upstream" in txt, txt
