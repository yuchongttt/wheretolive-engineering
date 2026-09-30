"""Area-level EA flood-zone aggregation (R2-4 Fix W, 2026-08-14).

R2-4 (agent-benchmark): the reference answer warned that one side of a local
park is a flood storage area while the other side sits on higher ground. Our own
per-listing FZ stamps proved it, sharper than the warning — every point in one
outcode was Zone 1 and every FZ3 hit sat in the neighbouring one — but no tool
exposed an AREA aggregate, so the card never got played.
The R2-1 flood probe had already named "area-level aggregation" as the remaining gap.

Shared by get_area_profile (outcode/sector of the asked postcode) and
compare_postcodes (one line per compared side).

Wording discipline (same as fetch_listing's flood lines): the dataset is
rivers & sea ONLY, zones deliberately ignore flood defences, surface-water is
not assessed, and a clean aggregate is INFORMATION (say it) — while thin
coverage (<20 checked points) yields None rather than a shaky share.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

# Data directory holding evaluations.db: $WTL_DATA_DIR, else ./data next to this file.
DATA_DIR = Path(os.environ.get("WTL_DATA_DIR") or Path(__file__).resolve().parent / "data")
DB_PATH = DATA_DIR / "evaluations.db"

MIN_CHECKED = 20


def flood_area_stats(db_path=None, outcode: str | None = None,
                     inward1: str | None = None) -> dict | None:
    """FZ2/FZ3 share among ACTIVE canonical listings' checked points in an
    outcode (optionally narrowed to one inward first digit = a sector).
    None when the area has <MIN_CHECKED checked points, or on any DB error.

    Design note (review 2026-08-15, both angles' top finding): a compact
    "sector string" is AMBIGUOUS — E1's sector "E1 6" compacts to "E16",
    which IS the Royal Docks outcode (91% FZ3), so prefix-matching fabricated
    flood risk for Whitechapel. The sector is therefore expressed as
    outcode + separate inward first digit, matched as: index-friendly prefix
    RANGE + exact unit length (= exact outcode) + substr on the inward digit.
    """
    oc = (outcode or "").strip().upper().replace(" ", "")
    if not oc or not (2 <= len(oc) <= 4):
        return None
    d1 = (inward1 or "").strip().upper() or None
    try:
        conn = sqlite3.connect(f"file:{db_path or DB_PATH}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        preds = ["postcode_norm >= ?", "postcode_norm < ?",
                 "length(postcode_norm) = ?"]
        params: list = [oc, oc[:-1] + chr(ord(oc[-1]) + 1), len(oc) + 3]
        if d1:
            preds.append("substr(postcode_norm, -3, 1) = ?")
            params.append(d1)
        row = conn.execute(
            f"""SELECT COUNT(*),
                       SUM(flood_zone = 'FZ3'),
                       SUM(flood_zone = 'FZ2')
                FROM rm_sales_overview
                WHERE delisted_date IS NULL
                  AND (canonical_id IS NULL OR canonical_id = id)
                  AND flood_checked_at IS NOT NULL
                  AND {' AND '.join(preds)}""",
            params).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    n, fz3, fz2 = row[0] or 0, row[1] or 0, row[2] or 0
    if n < MIN_CHECKED:
        return None
    return {"scope": oc + (f" {d1}" if d1 else ""), "n_checked": n,
            "fz3_n": fz3, "fz3_share": fz3 / n,
            "fz2_n": fz2, "fz2_share": fz2 / n}


def flood_area_lines(stats_list) -> list[str]:
    """Render one line per area + one shared discipline footer."""
    out: list[str] = []
    any_stat = False
    for st in stats_list:
        if not st:
            continue
        any_stat = True
        if st["fz3_n"] or st["fz2_n"]:
            bits = []
            if st["fz3_n"]:
                bits.append(f"{st['fz3_n']} in FZ3 ({st['fz3_share']:.0%})")
            if st["fz2_n"]:
                bits.append(f"{st['fz2_n']} in FZ2 ({st['fz2_share']:.0%})")
            out.append(f"{st['scope']}: {' + '.join(bits)} of "
                       f"{st['n_checked']} checked active-listing points — "
                       "concentrated pockets are typical; check the specific "
                       "address before offering.")
        else:
            out.append(f"{st['scope']}: all {st['n_checked']} checked "
                       "active-listing locations sit outside EA Flood Zones "
                       "2/3 (≈ Zone 1) — a statement about current stock's "
                       "points, not every parcel of land in the area.")
    if any_stat:
        out.append("(EA Flood Map for Planning, rivers & sea only — zones "
                   "deliberately ignore flood defences, and surface-water "
                   "flash-flood risk is NOT assessed by this dataset.)")
        # The model reaches for flood defences to "explain" zone differences, and
        # often gets the direction backwards (2026-09-03 case: "the barrier
        # protects eastwards, Battersea isn't protected by it" — wrong on both
        # counts). Since the zones ignore defences anyway, explaining zones by
        # defences is methodologically unsound: forbid it and state the correct
        # direction.
        out.append("Do NOT explain these shares by flood defences — the zones "
                   "already ignore them, so a defence cannot account for a "
                   "difference between two areas. (For reference, since this "
                   "is routinely stated backwards: the Thames Barrier sits at "
                   "Woolwich Reach and protects UPSTREAM of itself — central "
                   "London and the reaches west of it, Battersea included; "
                   "riverside downstream of the barrier is not behind it.)")
    return out
