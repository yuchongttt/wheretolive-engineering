"""Unknown-unit-postcode alert (R3-2 T18b fix): the zero-match signal must come from the tool.

T18b as measured: asked "what is E14 5AB worth?", the area tools had
sector-level data and silently fell back to it; the model never received a
"this unit postcode doesn't exist" signal, so the precondition for the AT rule
(zero matches anywhere → search for what it really is) was never established,
and the answer used sector medians to give a business postcode a house price.
Fix: tools that accept a unit postcode (get_comparables / get_area_overview /
get_price_trend) put UNKNOWN_POSTCODE_ALERT at the top of their output when
both coords storage formats and LR have zero matches, saying explicitly "what
follows is an outcode aggregate and says nothing about this address; identify
it first".
Reverse discipline: an existing postcode never triggers the alert; outcode/
sector-shaped input is not applicable (None).
"""
import asyncio
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import get_comparables as gc  # noqa: E402
from geo_radius import unit_postcode_known  # noqa: E402


def _mk_db(tmp_path):
    db = tmp_path / "evaluations.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE lr_transactions (postcode TEXT, paon TEXT, saon TEXT, "
        "street TEXT, price INT, date TEXT, property_type TEXT, tenure TEXT, "
        "new_build TEXT, category TEXT)")
    conn.executemany(
        "INSERT INTO lr_transactions VALUES (?,?,?,?,?,?,?,?,?,?)",
        [("N1 9DT", str(i), "", "TEST ST", 500000 + i, "2026-01-0%d" % (i + 1),
          "F", "L", "N", "A") for i in range(3)])
    conn.execute("CREATE TABLE postcode_coords (postcode TEXT PRIMARY KEY, "
                 "latitude REAL, longitude REAL)")
    conn.execute("INSERT INTO postcode_coords VALUES ('N1 9DT', 51.53, -0.11)")
    # the unspaced storage form counts as existing too (the coords dual-format trap)
    conn.execute("INSERT INTO postcode_coords VALUES ('N19EE', 51.53, -0.11)")
    conn.execute("CREATE TABLE rm_sales_overview (property_id TEXT, "
                 "postcode TEXT, bedrooms INT, property_type TEXT, "
                 "asking_price INT)")
    conn.commit()
    conn.close()
    return db


def test_predicate_three_states(tmp_path):
    conn = sqlite3.connect(_mk_db(tmp_path))
    assert unit_postcode_known(conn, "N1 9DT") is True
    assert unit_postcode_known(conn, "N19EE") is True      # unspaced storage also hits
    assert unit_postcode_known(conn, "N1 9XX") is False    # zero matches anywhere
    assert unit_postcode_known(conn, "N1") is None         # outcode: not applicable
    assert unit_postcode_known(conn, "N1 9") is None       # sector: not applicable


def _call(db, monkeypatch, pc):
    monkeypatch.setattr(gc, "DB_PATH", db)
    out = asyncio.run(gc.handle_call_tool("get_comparables", {"postcode": pc}))
    return out[0].text


def test_comparables_alert_on_unknown_unit_pc(tmp_path, monkeypatch):
    text = _call(_mk_db(tmp_path), monkeypatch, "N1 9XX")
    assert "POSTCODE NOT FOUND" in text
    assert "web search" in text
    assert text.index("POSTCODE NOT FOUND") < text.index("Comparables for")


def test_comparables_no_alert_on_known_pc(tmp_path, monkeypatch):
    """Reverse: an existing postcode never triggers the alert."""
    text = _call(_mk_db(tmp_path), monkeypatch, "N1 9DT")
    assert "POSTCODE NOT FOUND" not in text
