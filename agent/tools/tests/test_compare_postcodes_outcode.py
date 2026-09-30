"""compare_postcodes failed silently in three ways on outcode input.

Background (2026-09-03, a real session comparing SW11 with SE18): the model made the most natural call, compare_postcodes(["SW11",
"SE18"]), and got back a table of all "—" plus "⚠ Not yet scored: SW11, SE18".
Three things were broken:

  1. _load matched the outcode exactly against unit postcodes in
     postcode_scores, which can never hit → the whole score table blank;
  2. the blank was rendered as "not yet scored" — a **false claim of
     absence**. SW11 has 747 scored unit postcodes, SE18 has 477. The model
     could easily have told the user "we have no data for these two areas",
     which is wrong;
  3. deriving _ocs required len>=5 (a full unit postcode), so a 4-5 character
     outcode was filtered out entirely → the flood and council-tax sections
     **silently vanished**. Yet the tool's own flood comment says plainly
     "cross-area moves are exactly where 'one side of the park floods and the
     other doesn't' changes the decision" — a cross-area comparison is
     precisely when it should appear, and that is exactly when it went quiet.

Contract (same source as get_postcode_scores.area_summary — no second caliber):
  - outcode/sector input is aggregated as area medians and explicitly labelled
    as an area median, not an address;
  - if scores exist, never say "Not yet scored";
  - outcode/sector input still derives an outcode; flood/council-tax sections
    must be present;
  - unit-postcode behaviour unchanged (regression).
"""
import asyncio
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import compare_postcodes  # noqa: E402
import flood_area  # noqa: E402


def _db(tmp_path, scores=(), listings=()):
    path = tmp_path / "evaluations.db"
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE postcode_scores (
        postcode TEXT, data_version TEXT, total_score REAL, transport REAL,
        community REAL, environment REAL, price REAL, schools REAL,
        median_price REAL, data_completeness TEXT, scored_at TEXT)""")
    for pc, t, c, e, p, s, tot, med in scores:
        conn.execute(
            "INSERT INTO postcode_scores (postcode,transport,community,environment,"
            "price,schools,total_score,median_price,data_completeness) "
            "VALUES (?,?,?,?,?,?,?,?,'5/5')", (pc, t, c, e, p, s, tot, med))
    conn.execute("""CREATE TABLE rm_sales_overview (
        id TEXT PRIMARY KEY, postcode TEXT, asking_price REAL, flood_zone TEXT,
        flood_checked_at TEXT, delisted_date TEXT, canonical_id TEXT,
        postcode_norm TEXT GENERATED ALWAYS AS (upper(replace(postcode,' ',''))) VIRTUAL)""")
    for i, (pc, price, fz) in enumerate(listings):
        conn.execute("INSERT INTO rm_sales_overview (id,postcode,asking_price,"
                     "flood_zone,flood_checked_at) VALUES (?,?,?,?,'2026-08-01')",
                     (str(1000 + i), pc, price, fz))
    conn.commit()
    conn.close()
    return path


def _run(tmp_path, monkeypatch, postcodes, scores=(), listings=()):
    path = _db(tmp_path, scores, listings)
    monkeypatch.setattr(compare_postcodes, "DB_PATH", path)
    monkeypatch.setattr(flood_area, "DB_PATH", path)
    res = asyncio.run(compare_postcodes.handle_call_tool(
        "compare_postcodes", {"postcodes": postcodes}))
    return res[0].text


# three scored unit postcodes on the SW11 side, Transport median should be 60.0; SE18 side 70.0.
_SCORES = [
    ("SW11 1AA", 50.0, 60.0, 55.0, 70.0, 65.0, 62.0, 700000.0),
    ("SW11 1AB", 60.0, 65.0, 60.0, 72.0, 66.0, 68.0, 726000.0),
    ("SW11 1AC", 70.0, 70.0, 65.0, 74.0, 67.0, 74.0, 750000.0),
    ("SE18 6AA", 65.0, 55.0, 70.0, 55.0, 52.0, 58.0, 350000.0),
    ("SE18 6AB", 70.0, 58.0, 75.0, 57.0, 54.0, 61.0, 367000.0),
    ("SE18 6AC", 75.0, 61.0, 80.0, 59.0, 56.0, 64.0, 380000.0),
]


def test_outcode_scores_are_area_medians_not_blank(tmp_path, monkeypatch):
    """SW11 Transport median 60.0 and SE18 70.0 must appear in the table."""
    out = _run(tmp_path, monkeypatch, ["SW11", "SE18"], scores=_SCORES)
    transport = [ln for ln in out.splitlines() if ln.startswith("Transport")]
    assert transport, f"no Transport row:\n{out}"
    assert "60" in transport[0] and "70" in transport[0], transport[0]


def test_outcode_with_scores_is_never_called_unscored(tmp_path, monkeypatch):
    """False absence claim: if scored postcodes exist underneath, never say Not yet scored."""
    out = _run(tmp_path, monkeypatch, ["SW11", "SE18"], scores=_SCORES)
    assert "Not yet scored" not in out, out


def test_outcode_rows_are_labelled_as_area_medians(tmp_path, monkeypatch):
    """Must say this is an area median, not an address — otherwise the model quotes it as an address-level fact."""
    out = _run(tmp_path, monkeypatch, ["SW11", "SE18"], scores=_SCORES)
    assert "median across" in out.lower() or "area-wide median" in out.lower(), out


def test_outcode_input_still_emits_flood_section(tmp_path, monkeypatch):
    """The len>=5 guard on _ocs filtered outcodes out and the flood section silently vanished."""
    listings = ([("SW11 1AA", 600000.0, "FZ3")] * 21
                + [("SE18 6AA", 400000.0, None)] * 21)
    out = _run(tmp_path, monkeypatch, ["SW11", "SE18"],
               scores=_SCORES, listings=listings)
    assert "Flood zones" in out, out


def test_unit_postcode_behaviour_unchanged(tmp_path, monkeypatch):
    """Regression: full unit postcodes still use the exact row and still get the flood section."""
    listings = [("SW11 1AA", 600000.0, "FZ3")] * 21
    out = _run(tmp_path, monkeypatch, ["SW11 1AA", "SW11 1AB"],
               scores=_SCORES, listings=listings)
    assert "Not yet scored" not in out, out
    transport = [ln for ln in out.splitlines() if ln.startswith("Transport")][0]
    assert "50" in transport and "60" in transport, transport
    assert "Flood zones" in out, out


def test_sector_input_is_not_mangled_into_a_different_place(tmp_path, monkeypatch):
    """The local normalise_postcode strips spaces then re-inserts one: "SW11 1" → "SW 111".

    area_scope.classify_postcode_scope documents this: the internal space is
    load-bearing, and unspaced "SW11" and spaced "SW1 1" are different places.
    What's wanted here is sector SW11 1 — it must not turn into anything else.
    """
    out = _run(tmp_path, monkeypatch, ["SW11 1", "SE18 6"], scores=_SCORES)
    assert "SW11 1" in out, out
    assert "SW 111" not in out, out
    assert "Not yet scored" not in out, out
