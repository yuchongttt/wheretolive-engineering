"""Borough council-tax rate table (R2-4 Fix V): lookup / rendering.

Third root cause of the R2-4 loss: every listing had a council_tax_band but
there was no borough rate table, so we couldn't compute real cross-borough
cost differences like "the same Band D costs £1,550/yr more in Croydon than in
Westminster" (the reference agent scored this point with public rates).
Source: gov.uk Council Tax levels Table_9 ("area" basis incl. the GLA precept,
what residents actually pay), updated yearly; the ingest fails loudly unless
all 33 boroughs are present.

(Extract note: the original file also tested the ingest script's ONS-code
matcher, lib/council_tax_ods.py, which is not part of this extract.)
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import council_tax as ct  # noqa: E402


def _dbs(tmp_path):
    area = tmp_path / "area_intel.db"
    conn = sqlite3.connect(area)
    conn.execute("""CREATE TABLE council_tax_rates (
        year TEXT, borough TEXT, ons_code TEXT,
        band_a REAL, band_b REAL, band_c REAL, band_d REAL,
        band_e REAL, band_f REAL, band_g REAL, band_h REAL, fetched_at TEXT,
        PRIMARY KEY (year, ons_code))""")
    conn.executemany("INSERT INTO council_tax_rates VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
        ("2026-27", "Croydon", "E09000008", 1733, 2022, 2311, 2600, 3178, 3755, 4333, 5200, "t"),
        ("2026-27", "Westminster", "E09000033", 700, 816, 933, 1050, 1283, 1516, 1750, 2100, "t"),
    ])
    conn.commit(); conn.close()
    ev = tmp_path / "evaluations.db"
    conn = sqlite3.connect(ev)
    # the real production shape (review #3): PRIMARY KEY(outcode), one row per outcode, share = main borough's share.
    conn.execute("CREATE TABLE outcode_borough (outcode TEXT PRIMARY KEY, borough TEXT, share REAL, source TEXT)")
    conn.executemany("INSERT INTO outcode_borough VALUES (?,?,?,NULL)", [
        ("CR0", "Croydon", 0.9), ("SW1A", "Westminster", 1.0),
        ("WC2A", "Westminster", 0.5)])
    conn.commit(); conn.close()
    return area, ev


def test_rates_for_outcode_majority_borough(tmp_path, monkeypatch):
    area, ev = _dbs(tmp_path)
    monkeypatch.setattr(ct, "AREA_DB", area)
    monkeypatch.setattr(ct, "EVAL_DB", ev)
    r = ct.rates_for_outcode("CR0")
    assert r["borough"] == "Croydon" and r["bands"]["D"] == 2600
    assert ct.rates_for_outcode("ZZ99") is None


def test_lines_render_diff(tmp_path, monkeypatch):
    area, ev = _dbs(tmp_path)
    monkeypatch.setattr(ct, "AREA_DB", area)
    monkeypatch.setattr(ct, "EVAL_DB", ev)
    lines = ct.council_tax_lines(
        [ct.rates_for_outcode("CR0"), ct.rates_for_outcode("SW1A")], band="D")
    txt = "\n".join(lines)
    assert "£2,600" in txt and "£1,050" in txt
    assert "£1,550" in txt          # the difference line
    assert "GLA precept" in txt   # the basis is stated


def test_straddling_outcode_gets_hedge_not_directive(tmp_path, monkeypatch):
    """review #3 (confirmed: 63 outcodes with share < 0.85): WC2A's main
    borough has only 0.50 (Westminster £1,050 / the Camden side £2,208, a
    £1,158 difference). With share < 0.85 the output must carry a straddling
    hedge and must NOT emit a "cite it"-style assertive instruction."""
    area, ev = _dbs(tmp_path)
    monkeypatch.setattr(ct, "AREA_DB", area)
    monkeypatch.setattr(ct, "EVAL_DB", ev)
    r = ct.rates_for_outcode("WC2A")
    assert r["borough_share"] == 0.5
    lines = ct.council_tax_lines([r, ct.rates_for_outcode("SW1A")], band="D")
    txt = "\n".join(lines)
    assert "straddles" in txt or "跨" in txt        # the hedge is present
    assert "cite it" not in txt                     # the assertive instruction is suppressed
    # with both sides ≥ 0.85 the instruction appears as normal
    lines2 = ct.council_tax_lines(
        [ct.rates_for_outcode("CR0"), ct.rates_for_outcode("SW1A")], band="D")
    assert "cite it" in "\n".join(lines2)
