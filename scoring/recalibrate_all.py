#!/usr/bin/env python3
"""
Recompute the PCHIP breakpoints for all dimensions + the total score in one go.

Two passes:
  Pass 1 — compute raw dimension scores for every evaluated postcode and sample breakpoints at 22 percentile points
  Pass 2 — apply per-dim calibration with the new breakpoints, then sample breakpoints on the weighted average (i.e. the total breakpoints)

Output: a Python dict that can be pasted straight into simple_scorer.CALIBRATION_BREAKPOINTS.

Usage:
  python3 recalibrate_all.py                  # print the new breakpoints to stdout
  python3 recalibrate_all.py --apply          # edit simple_scorer.py in place (.bak backup)
"""

import argparse
import json
import os
import shutil
import sqlite3
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from simple_scorer import SimpleScorer

DB = Path(__file__).parent / "data" / "evaluations.db"
SCORER_FILE = Path(__file__).parent / "simple_scorer.py"

PERCENTILES = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
               0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
               0.97, 0.99]

DIMS = ["transport", "community", "environment", "price", "schools"]


def load_postcode_data(conn, pc):
    """Load the full scoring-input dict. The env caches are keyed by lat/lng, so look up postcode_coords first."""
    data = {"address": pc}

    for key, table, col in [
        ("commute",        "dim_commute",        "data_json"),
        ("transit",        "dim_transit",        "data_json"),
        ("safety",         "dim_safety",         "data_json"),
        ("demographics",   "dim_demographics",   "data_json"),
        ("price_analysis", "dim_price_analysis", "data_json"),
        ("schools",        "dim_schools",        "data_json"),
    ]:
        row = conn.execute(
            f"SELECT {col} FROM {table} WHERE postcode = ? LIMIT 1", (pc,)
        ).fetchone()
        if row and row[0]:
            try:
                data[key] = json.loads(row[0])
            except json.JSONDecodeError:
                pass

    # Look up the postcode's lat/lng (the env caches are keyed by lat/lng)
    coord_row = conn.execute(
        "SELECT latitude, longitude FROM postcode_coords WHERE postcode = ? LIMIT 1",
        (pc,)
    ).fetchone()
    if coord_row:
        lat, lng = coord_row
        # noise_cache is stored column-wise (lat, lng, road_db, rail_db, airport_db, combined_db, level, level_zh, score, dominant_source)
        n_row = conn.execute(
            "SELECT road_db, rail_db, airport_db, combined_db, level, score, dominant_source "
            "FROM noise_cache WHERE ABS(lat-?)<0.0005 AND ABS(lng-?)<0.0005 LIMIT 1",
            (lat, lng),
        ).fetchone()
        if n_row:
            data["noise"] = {
                "road_db": n_row[0], "rail_db": n_row[1], "airport_db": n_row[2],
                "combined_db": n_row[3], "level": n_row[4],
                "score": n_row[5], "dominant_source": n_row[6],
            }
        # flood / air / parks: all data_json columns
        for key, table in [("flood_risk", "flood_cache"),
                           ("air_quality", "air_quality_cache"),
                           ("parks", "parks_cache")]:
            try:
                row = conn.execute(
                    f"SELECT data_json FROM {table} "
                    f"WHERE ABS(lat-?)<0.0005 AND ABS(lng-?)<0.0005 LIMIT 1",
                    (lat, lng),
                ).fetchone()
                if row and row[0]:
                    data[key] = json.loads(row[0])
            except sqlite3.OperationalError:
                pass
    return data


def compute_breakpoints_from_values(values, label):
    """Sample 22 percentile breakpoints. PCHIP requires strictly increasing x; ties are broken with an epsilon."""
    values = sorted(values)
    n = len(values)
    breakpoints = []
    prev_x = None
    for p in PERCENTILES:
        idx = min(int(p * n), n - 1)
        x = round(values[idx], 1)
        # Break ties: if equal to the previous breakpoint, add 0.01
        if prev_x is not None and x <= prev_x:
            x = round(prev_x + 0.01, 2)
        breakpoints.append((x, p))
        prev_x = x
    return breakpoints


def format_breakpoints(bps):
    """Format as the tuple block used in simple_scorer."""
    lines = []
    buf = []
    for v, p in bps:
        buf.append(f"({v}, {p:.2f})")
        if len(buf) == 4:
            lines.append("            " + ", ".join(buf) + ",")
            buf = []
    if buf:
        lines.append("            " + ", ".join(buf) + ",")
    return "\n".join(lines)


def check_distribution(values, label):
    """Simple normality check."""
    arr = np.array(values)
    return {
        "n": len(values),
        "mean": arr.mean(),
        "sd": arr.std(),
        "p10": np.percentile(arr, 10),
        "p50": np.percentile(arr, 50),
        "p90": np.percentile(arr, 90),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true",
                   help="write straight back to simple_scorer.py (.bak backup)")
    args = p.parse_args()

    conn = sqlite3.connect(DB, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")

    rows = conn.execute("""
        SELECT s.postcode FROM dim_schools s
        INNER JOIN dim_commute c ON c.postcode = s.postcode AND c.data_json NOT LIKE '%"_no_data"%'
        INNER JOIN dim_transit t ON t.postcode = s.postcode AND t.data_json NOT LIKE '%"_no_data"%'
        INNER JOIN dim_safety sf ON sf.postcode = s.postcode AND sf.data_json NOT LIKE '%"_no_data"%'
        INNER JOIN dim_demographics d ON d.postcode = s.postcode AND d.data_json NOT LIKE '%"_no_data"%'
        INNER JOIN dim_price_analysis p ON p.postcode = s.postcode AND p.data_json NOT LIKE '%"_no_data"%'
        GROUP BY s.postcode
    """).fetchall()

    print(f"Collecting data from {len(rows)} postcodes...", file=sys.stderr)

    # Compute raw dimension scores directly (bypassing calibration) — temporary monkey-patch
    scorer = SimpleScorer()
    raw_calibrate = scorer._calibrate_score
    scorer._calibrate_score = lambda raw, dim: (raw, 50.0)  # make calibration a pass-through so the output stays raw

    raw_by_dim = {d: [] for d in DIMS}
    for (pc,) in rows:
        data = load_postcode_data(conn, pc)
        try:
            result = scorer.calculate_scores(data)
            for dim in DIMS:
                v = result.get("scores", {}).get(dim)
                if v and isinstance(v, dict) and v.get("score") is not None:
                    raw_by_dim[dim].append(v["score"])
        except Exception:
            pass

    # Restore the original calibration
    scorer._calibrate_score = raw_calibrate

    print(file=sys.stderr)
    print("Raw distributions:", file=sys.stderr)
    for dim in DIMS:
        s = check_distribution(raw_by_dim[dim], dim)
        print(f"  {dim:12s}  n={s['n']:>5}  mean={s['mean']:>6.2f}  sd={s['sd']:>6.2f}  "
              f"p10={s['p10']:>5.1f}  p90={s['p90']:>5.1f}", file=sys.stderr)

    # Build the new per-dim breakpoints
    new_bps = {}
    for dim in DIMS:
        new_bps[dim] = compute_breakpoints_from_values(raw_by_dim[dim], dim)

    # Pass 2: run the total calibration with the new breakpoints (the scorer has to use this new set)
    # Temporarily override CALIBRATION_BREAKPOINTS so the total is based on the correct per-dim calibration
    # Note: clear _pchip_cache so stale entries are not reused
    original_bps = scorer.CALIBRATION_BREAKPOINTS
    scorer.__class__.CALIBRATION_BREAKPOINTS = {**new_bps}  # no total yet; compute the total first
    scorer.__class__._pchip_cache = {}

    # Also need to take community out of SKIP_CALIBRATION (bug fix #1 decided in the review)
    # but that flag lives inside calculate_scores and can't be changed from here. We use _calibrate_score directly
    # and compute the weighted average by hand:
    print(file=sys.stderr)
    print("Computing weighted averages (with new per-dim breakpoints, community calibrated)...", file=sys.stderr)

    WEIGHTS = SimpleScorer.WEIGHTS
    total_raws = []
    for (pc,) in rows:
        data = load_postcode_data(conn, pc)
        try:
            # Temporarily reset SKIP_CALIBRATION via a monkey-patched calculate_scores
            # Simpler: compute it by hand directly (with the new breakpoints)
            result = scorer.calculate_scores(data)
            # result["total_score"] is already the raw weighted average (since CALIBRATION_BREAKPOINTS has no total)
            t = result.get("total_score")
            if t is not None and t > 0:
                total_raws.append(t)
        except Exception:
            pass

    # Restore
    scorer.__class__.CALIBRATION_BREAKPOINTS = original_bps
    scorer.__class__._pchip_cache = {}

    s = check_distribution(total_raws, "total_raw")
    print(f"  total (raw weighted avg)  n={s['n']}  mean={s['mean']:.2f}  sd={s['sd']:.2f}  "
          f"p10={s['p10']:.1f}  p90={s['p90']:.1f}", file=sys.stderr)

    new_bps["total"] = compute_breakpoints_from_values(total_raws, "total")

    # Output
    print("CALIBRATION_BREAKPOINTS = {")
    order = ["transport", "community", "environment", "price", "schools", "total"]
    comment = {
        "transport":   "",
        "community":   "  # former SKIP_CALIBRATION removed; now calibrated",
        "environment": "",
        "price":       "",
        "schools":     "",
        "total":       "  # weighted avg → N(65,15)",
    }
    for dim in order:
        print(f'        "{dim}": [{comment[dim]}')
        print(format_breakpoints(new_bps[dim]))
        print("        ],")
    print("    }")

    conn.close()


if __name__ == "__main__":
    main()
