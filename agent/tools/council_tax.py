"""Borough council-tax rates lookup for chat tools (R2-4 Fix V).

area_intel.db council_tax_rates (year x 33 boroughs x bands A-H; "area"
basis = what residents actually pay, incl. the GLA precept) + evaluations.db
outcode_borough (outcode -> borough). Third root cause of the R2-4 loss:
every listing had a band but there was no rate table, so we could not compute
real cross-borough cost differences like "the same Band D costs over £1,500/yr
more in Croydon than in Westminster".

An outcode spanning boroughs (largest share wins) is costed at its main
borough — the output names the borough so the model can hedge. The year is the
table's latest.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

# Data directory holding area_intel.db + evaluations.db:
# $WTL_DATA_DIR, else ./data next to this file.
DATA_DIR = Path(os.environ.get("WTL_DATA_DIR") or Path(__file__).resolve().parent / "data")
AREA_DB = DATA_DIR / "area_intel.db"
EVAL_DB = DATA_DIR / "evaluations.db"

_BAND_ORDER = "ABCDEFGH"


# In production outcode_borough is keyed PRIMARY KEY(outcode), one row per
# outcode, share = the main borough's share. Review 2026-08-15 #3 confirmed 63
# outcodes with share < 0.85 (WC2A/SW1X as low as 0.50; Westminster £1,050 vs
# Camden £2,208 — picking the wrong half is £1,158/yr), so share must travel
# with the borough and consumers hedge on it.
_STRADDLE_SHARE = 0.85


def borough_for_outcode(outcode: str) -> tuple[str, float] | None:
    try:
        conn = sqlite3.connect(f"file:{EVAL_DB}?mode=ro", uri=True)
        row = conn.execute(
            "SELECT borough, share FROM outcode_borough WHERE outcode = ?",
            (outcode.strip().upper(),)).fetchone()
        conn.close()
        return (row[0], row[1] if row[1] is not None else 1.0) if row else None
    except sqlite3.Error:
        return None


def rates_for_outcode(outcode: str) -> dict | None:
    """-> {borough, borough_share, year, bands: {'A':…,'D':…}} or None."""
    got = borough_for_outcode(outcode)
    if not got:
        return None
    borough, share = got
    try:
        conn = sqlite3.connect(f"file:{AREA_DB}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM council_tax_rates WHERE borough = ? "
            "ORDER BY year DESC LIMIT 1", (borough,)).fetchone()
        conn.close()
    except sqlite3.Error:
        return None
    if not row:
        return None
    bands = {b: row[f"band_{b.lower()}"] for b in _BAND_ORDER}
    return {"borough": borough, "borough_share": share,
            "year": row["year"], "bands": bands,
            "basis": ("area Band figures incl. GLA precept, full-occupancy "
                      "(before the 25% single-person discount)")}


def council_tax_lines(rates_list, band: str = "D") -> list[str]:
    """Render comparison lines: one per borough + a difference line (>= 2 boroughs)."""
    band = (band or "D").upper()
    got = [r for r in rates_list if r]
    out = []
    any_straddle = False
    for r in got:
        amt = r["bands"].get(band)
        if not amt:
            continue
        line = (f"{r['borough']}: Band {band} £{amt:,.0f}/yr ({r['year']}, "
                "incl. GLA precept)")
        if (r.get("borough_share") or 1.0) < _STRADDLE_SHARE:
            any_straddle = True
            line += (f" ⚠ this outcode straddles boroughs ({r['borough']} covers "
                     f"only ~{(r['borough_share'] or 0) * 100:.0f}% of it) — "
                     "confirm the exact address's borough before relying on this")
        out.append(line)
    if len(got) >= 2 and not any_straddle:
        amts = [(r["borough"], r["bands"].get(band)) for r in got if r["bands"].get(band)]
        if len(amts) >= 2:
            hi = max(amts, key=lambda x: x[1])
            lo = min(amts, key=lambda x: x[1])
            if hi[1] - lo[1] >= 1 and hi[0] != lo[0]:
                out.append(f"same band, {hi[0]} costs £{hi[1] - lo[1]:,.0f}/yr more "
                           f"than {lo[0]} — real moving-cost difference, cite it.")
    return out
