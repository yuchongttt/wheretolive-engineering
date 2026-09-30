#!/usr/bin/env python3
"""v8 deeper analysis:
  Q1. Bias correction sanity: re-eval with pred × 1.031 calibration
  Q2. Per-feature-group drop-one importance (incl. baseline groups)
  Q3. Minimal-info mode — only address + bedrooms, with LR/EPC lookup
       allowed at predict time; everything else (council, SigLIP,
       bathrooms, listing-sqm) treated as unavailable.
"""
from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).parent))
from valuation_v7_explore import (
    V6_BEST, load_pool, fetch_luxury_from_linux_v6,
    V5_BASE_COLS as _V5_BASE_COLS, V5_LUX_COLS,
)
from valuation_v8_temporal import gbr_default, GROUPS as V8_GROUPS

DB = Path(__file__).resolve().parents[1] / "data" / "evaluations.db"
SOLD_DB = Path(__file__).resolve().parents[1] / "data" / "sold.db"


# v8 baseline column structure broken into droppable units.
# Each entry is a feature-builder that returns a DataFrame fragment to
# include. Removing one = passing `drop_baseline={"name"}`.
def build_baseline(df: pd.DataFrame, drop: set[str]) -> pd.DataFrame:
    parts = []
    if "geo" not in drop:
        parts.append(df[["latitude", "longitude"]].copy())
    if "bedrooms" not in drop:
        parts.append(df[["bedrooms"]].copy())
    if "floor_area" not in drop:
        parts.append(df[["floor_area_sqm"]].copy())
    if "time" not in drop:
        parts.append(df[["year_sold", "months_since_sale"]].copy())
    if "council" not in drop:
        parts.append(df[["council_ord_v5"]].copy())
    if "tenure" not in drop:
        s = (df["tenure"].fillna("") == "Leasehold").astype(int).to_frame("is_leasehold")
        parts.append(s)
    if "type" not in drop:
        parts.append(pd.get_dummies(df["type_bucket"], prefix="type"))
    if "siglip" not in drop:
        lux = df[V5_LUX_COLS].copy()
        lux["n_photos"] = lux["n_photos"].fillna(0)
        for c in ("interior_pct", "mean_lux", "max_lux", "interior_mean_lux"):
            lux[c] = lux[c].fillna(lux[c].median())
        parts.append(lux)
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)


def build_v8_best(df: pd.DataFrame,
                  drop_baseline: set[str] = None,
                  drop_groups:   set[str] = None) -> pd.DataFrame:
    """v8_best = baseline + v6_best + area_per_bed.
    drop_baseline can include: geo, bedrooms, floor_area, time, council,
                                tenure, type, siglip
    drop_groups can include any v8 GROUPS key (e.g., repeat_sales, epc,
                                area_fusion, new_build, dist_zone1, area_per_bed)
    """
    drop_baseline = drop_baseline or set()
    drop_groups = drop_groups or set()

    base = build_baseline(df, drop_baseline)
    parts = [base]
    v8_best_groups = V6_BEST | {"area_per_bed"}
    for g in v8_best_groups:
        if g in drop_groups:
            continue
        cols = V8_GROUPS[g]
        if not cols:
            continue
        sub = df[cols].copy()
        for c in cols:
            if sub[c].isna().any():
                med = sub[c].median()
                if pd.isna(med):
                    med = 0
                sub[c] = sub[c].fillna(med)
        parts.append(sub)
    return pd.concat(parts, axis=1)


def evaluate_temporal(X, y_log, df, label, return_preds=False):
    """Temporal walk-forward by year. Returns aggregate dict; per-year
    list available via return_preds for bias-correction analysis."""
    periods = sorted(set(df.loc[df["year_sold"] >= 2020, "year_sold"]))
    apes_all = []; signed_all = []; preds_all = []; actuals_all = []
    for y in periods:
        cutoff = pd.Timestamp(f"{int(y)}-01-01")
        train_mask = (df["sold_date"] < cutoff).values
        test_mask  = (df["year_sold"] == y).values
        if test_mask.sum() < 20 or train_mask.sum() < 500:
            continue
        gbr = gbr_default()
        gbr.fit(X[train_mask], y_log[train_mask])
        pred = np.exp(gbr.predict(X[test_mask]))
        actual = df.loc[test_mask, "sold_price"].values.astype(float)
        apes_all.append(np.abs(pred - actual) / actual)
        signed_all.append((pred - actual) / actual)
        if return_preds:
            preds_all.append(pred)
            actuals_all.append(actual)
    apes = np.concatenate(apes_all)
    signed = np.concatenate(signed_all)
    res = {
        "label": label,
        "n_features": X.shape[1],
        "n_test": len(apes),
        "MAPE_pct": float(np.mean(apes) * 100),
        "median_APE_pct": float(np.median(apes) * 100),
        "within_5pct":  float((apes < 0.05).mean() * 100),
        "within_10pct": float((apes < 0.10).mean() * 100),
        "within_20pct": float((apes < 0.20).mean() * 100),
        "bias_pct": float(np.median(signed) * 100),
    }
    if return_preds:
        res["preds"] = np.concatenate(preds_all)
        res["actuals"] = np.concatenate(actuals_all)
    return res


def fmt_row(r, baseline_mape=None):
    delta = "" if baseline_mape is None else f"{r['MAPE_pct']-baseline_mape:>+5.2f}pp"
    return (f"{r['label']:<42s}  {r['n_features']:>3d}f  "
            f"{r['MAPE_pct']:>6.2f}% {delta:>8s}  "
            f"{r['within_5pct']:>4.1f}%  {r['within_10pct']:>4.1f}%  "
            f"{r['within_20pct']:>4.1f}%  {r['bias_pct']:>+5.1f}%")


def main():
    conn = sqlite3.connect(DB, timeout=30)
    conn.execute(f"ATTACH DATABASE '{SOLD_DB}' AS sold")
    conn.execute("PRAGMA query_only=1")
    uuids = pd.read_sql_query(
        "SELECT DISTINCT rm_uuid FROM sold.rm_sold_properties WHERE outcode IN ('E15','W7')",
        conn,
    )["rm_uuid"].tolist()
    lux_df = fetch_luxury_from_linux_v6(uuids)
    t0 = time.time()
    df = load_pool(conn, ["E15", "W7"], lux_df)
    print(f"[pool {len(df):,} rows built in {time.time()-t0:.1f}s]\n", flush=True)
    y_log = np.log(df["sold_price"].astype(float).values)

    HEADER = (f"{'Variant':<42s}  {'feat':>3s}  {'MAPE':>7s}  {'Δ':>8s}  "
              f"{'<5%':>5s}  {'<10%':>5s}  {'<20%':>5s}  {'bias':>6s}")

    # ─── v8_best reference run with predictions for Q1 ───
    print("=" * 100, flush=True)
    print("Q1. v8_best reference + bias-correction effect on MAPE", flush=True)
    print("=" * 100, flush=True)
    X_best = build_v8_best(df)
    r_best = evaluate_temporal(X_best, y_log, df, "v8_best (temporal)", return_preds=True)
    print(HEADER, flush=True)
    print(fmt_row(r_best), flush=True)
    # Apply naive bias correction: multiply all preds by (1 - bias/100)
    correction = 1.0 / (1.0 + r_best["bias_pct"] / 100.0)
    corrected = r_best["preds"] * correction
    apes_corr = np.abs(corrected - r_best["actuals"]) / r_best["actuals"]
    signed_corr = (corrected - r_best["actuals"]) / r_best["actuals"]
    print(f"{'v8_best × bias correction (× ' + f'{correction:.4f}'+')':<42s}  "
          f"{r_best['n_features']:>3d}f  "
          f"{apes_corr.mean()*100:>6.2f}% "
          f"{(apes_corr.mean()*100 - r_best['MAPE_pct']):>+7.2f}pp  "
          f"{(apes_corr < 0.05).mean()*100:>4.1f}%  "
          f"{(apes_corr < 0.10).mean()*100:>4.1f}%  "
          f"{(apes_corr < 0.20).mean()*100:>4.1f}%  "
          f"{np.median(signed_corr)*100:>+5.1f}%", flush=True)

    # ─── Q2. Drop-one importance ───
    print()
    print("=" * 100, flush=True)
    print("Q2. Drop-one importance (temporal split, v8_best as reference)", flush=True)
    print("=" * 100, flush=True)
    print(HEADER, flush=True)
    print(fmt_row(r_best, r_best["MAPE_pct"]), flush=True)

    # Drop each baseline group
    baseline_groups = ["geo", "bedrooms", "floor_area", "time", "council",
                       "tenure", "type", "siglip"]
    print("\n--- drop baseline group ---", flush=True)
    for g in baseline_groups:
        X = build_v8_best(df, drop_baseline={g})
        r = evaluate_temporal(X, y_log, df, f"v8_best - {g}")
        print(fmt_row(r, r_best["MAPE_pct"]), flush=True)

    # Drop each v8_best enrichment group
    print("\n--- drop enrichment group ---", flush=True)
    for g in ["epc", "area_fusion", "new_build", "dist_zone1", "repeat_sales",
              "area_per_bed"]:
        X = build_v8_best(df, drop_groups={g})
        r = evaluate_temporal(X, y_log, df, f"v8_best - {g}")
        print(fmt_row(r, r_best["MAPE_pct"]), flush=True)

    # ─── Q3. Minimal-info mode ───
    print()
    print("=" * 100, flush=True)
    print("Q3. Minimal-info eval: only address + bedrooms; LR + EPC lookup allowed", flush=True)
    print("=" * 100, flush=True)
    print("Available at predict time:", flush=True)
    print("  - address → geocode (lat, lng, dist_zone1)", flush=True)
    print("  - bedrooms (user input)", flush=True)
    print("  - LR lookup: prev sale info, tenure, new_build, property_type", flush=True)
    print("  - EPC lookup: floor_area_sqm, energy_rating, energy_efficiency", flush=True)
    print("Excluded (would require listing data or VOA lookup):", flush=True)
    print("  - council_tax_band  → drop council_ord_v5", flush=True)
    print("  - bathrooms         → already excluded since v5", flush=True)
    print("  - listing_floor_area_sqm → area_fusion collapses to EPC value, redundant", flush=True)
    print("  - SigLIP photo features → no photos at predict time, drop all 5", flush=True)
    print("  - area_per_bed is computable from bedrooms + floor_area → KEEP", flush=True)
    print(HEADER, flush=True)
    print(fmt_row(r_best, r_best["MAPE_pct"]), flush=True)
    X_min = build_v8_best(df,
        drop_baseline={"council", "siglip"},
        drop_groups={"area_fusion"})
    r_min = evaluate_temporal(X_min, y_log, df, "v8_minimal (address+beds+LR+EPC)")
    print(fmt_row(r_min, r_best["MAPE_pct"]), flush=True)

    # Even more minimal: NO LR lookup (no repeat-sales), only address+beds+EPC
    X_nolr = build_v8_best(df,
        drop_baseline={"council", "siglip", "tenure"},
        drop_groups={"area_fusion", "repeat_sales", "new_build"})
    r_nolr = evaluate_temporal(X_nolr, y_log, df, "v8_no_LR (address+beds+EPC only)")
    print(fmt_row(r_nolr, r_best["MAPE_pct"]), flush=True)

    # Pure v5_core: just geo + beds + floor_area + type + time (no enrichments at all)
    X_v5core = build_v8_best(df,
        drop_baseline={"council", "siglip", "tenure"},
        drop_groups={"area_fusion", "repeat_sales", "new_build", "epc",
                     "dist_zone1", "area_per_bed"})
    r_v5core = evaluate_temporal(X_v5core, y_log, df, "v5_core (geo+beds+sqft+type+time)")
    print(fmt_row(r_v5core, r_best["MAPE_pct"]), flush=True)

    return 0


if __name__ == "__main__":
    main()
