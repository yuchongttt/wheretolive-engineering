#!/usr/bin/env python3
"""Valuation v8 — fix v7 target leakage + add temporal-split evaluation.

Two corrections to v7:

(1) DROP `prev_annualised_growth`. Bug found 2026-05-21: the formula
    growth = sold_price / prev_sold_price uses the row's own target
    in the feature, so the model could recover sold_price ≈
    prev_sold_price × (1+cagr)^yrs almost directly. v7_best 7.89% MAPE
    was inflated by this leakage. The legitimate "repeat-sales anchor"
    info (prev_sold_price, years_since_prev, prev_log_ppsf) stays — it
    only uses *strictly prior* sale info, no target.

(2) Add temporal cross-validation. Current GroupKFold-by-uuid evaluation
    matches production scenario A ("predict today, may use all historical
    public data") which is correct for deployment but lets the model see
    other properties' future-period sales during training, masking
    market-regime memorisation. Temporal split is scenario B ("predict
    period Y using only data from before Y") — more pessimistic but
    diagnostic of generalisation across market regimes.

Run yields a 2-column table per variant: GroupKFold MAPE vs temporal
walk-forward MAPE. Gap reveals how much the model relies on knowing the
current market level vs property fundamentals.

Usage:
    venv/bin/python3 scripts/valuation_v8_temporal.py --outcodes E15,W7
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# Reuse v7's pool builder + feature registry — same code path, just
# without the leaky `prev_annualised_growth` column.
import sys
sys.path.insert(0, str(Path(__file__).parent))
from valuation_v7_explore import (
    GROUPS as V7_GROUPS,
    V5_BASE_COLS,
    V5_LUX_COLS,
    V6_BEST,
    load_pool,
    fetch_luxury_from_linux_v6,
)

DB = Path(__file__).resolve().parents[1] / "data" / "evaluations.db"
SOLD_DB = Path(__file__).resolve().parents[1] / "data" / "sold.db"

# v8 groups = v7 groups minus the leakage feature. Keep `prev_cagr` slot
# in the dict but with an empty column list so the toggle still works
# for back-compat scripts; effectively disabled.
GROUPS = dict(V7_GROUPS)
GROUPS["prev_cagr"] = []   # disabled — target leakage


def make_features(df: pd.DataFrame, enabled: set[str]) -> pd.DataFrame:
    """Identical to v7's make_features except GROUPS is the v8 version
    (prev_cagr's column list is empty so the group adds nothing)."""
    parts = [df[V5_BASE_COLS].copy()]
    parts[0]["is_leasehold"] = (df["tenure"].fillna("") == "Leasehold").astype(int)

    if "council_fix" in enabled:
        parts[0] = parts[0].drop(columns=["council_ord_v5"])
        parts.append(df[GROUPS["council_fix"]].copy())

    parts.append(pd.get_dummies(df["type_bucket"], prefix="type"))

    lux = df[V5_LUX_COLS].copy()
    lux["n_photos"] = lux["n_photos"].fillna(0)
    for c in ("interior_pct", "mean_lux", "max_lux", "interior_mean_lux"):
        lux[c] = lux[c].fillna(lux[c].median())
    parts.append(lux)

    for g, cols in GROUPS.items():
        if g == "council_fix" or g not in enabled or not cols:
            continue
        if g == "bed_x_type":
            sub = df[[c for c in df.columns if c.startswith("bedXtype_")]].copy()
        elif g == "epc_x_type":
            sub = df[[c for c in df.columns if c.startswith("epcXtype_")]].copy()
        else:
            sub = df[cols].copy()
            for c in cols:
                if sub[c].isna().any():
                    med = sub[c].median()
                    if pd.isna(med):
                        med = 0
                    sub[c] = sub[c].fillna(med)
        parts.append(sub)

    return pd.concat(parts, axis=1)


def gbr_default():
    return GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        subsample=0.8, random_state=42,
    )


# ─────────── Eval A: GroupKFold by uuid (production scenario) ───────────

def evaluate_groupkfold(X, y_log, df, label):
    """5-fold GroupKFold by rm_uuid. Test set = 2020+ sales of test
    properties; training set = all sales of train properties (including
    2020+). Matches v5/v6/v7 evaluation for backward comparability."""
    eval_mask = df["sold_date"] >= "2020-01-01"
    eval_props = sorted(df.loc[eval_mask, "rm_uuid"].unique().tolist())
    rng = np.random.RandomState(7)
    rng.shuffle(eval_props)
    fold_props = np.array_split(eval_props, 5)
    groups = df["rm_uuid"].values

    apes = []
    for fold_i, test_props in enumerate(fold_props):
        test_prop_set = set(test_props)
        test_rows = [i for i in range(len(df))
                     if groups[i] in test_prop_set and eval_mask.iloc[i]]
        train_rows = [i for i in range(len(df))
                      if groups[i] not in test_prop_set]
        if not test_rows or not train_rows:
            continue
        gbr = gbr_default()
        gbr.fit(X.iloc[train_rows], y_log[train_rows])
        pred = np.exp(gbr.predict(X.iloc[test_rows]))
        actual = df.iloc[test_rows]["sold_price"].values.astype(float)
        apes.append(np.abs(pred - actual) / actual)
    apes = np.concatenate(apes)
    return {
        "label": label,
        "split": "GroupKFold",
        "n_features": X.shape[1],
        "n_test": len(apes),
        "MAPE_pct":      float(np.mean(apes) * 100),
        "median_APE_pct":float(np.median(apes) * 100),
        "within_5pct":   float(np.mean(apes < 0.05) * 100),
        "within_10pct":  float(np.mean(apes < 0.10) * 100),
        "within_20pct":  float(np.mean(apes < 0.20) * 100),
    }


# ─────────── Eval B: temporal walk-forward by year ───────────

def evaluate_temporal(X, y_log, df, label, periods=None):
    """For each test year Y, train on sales strictly before Y-01-01 and
    test on sales in Y. Aggregates MAPE across years (equal-weighted by
    sample count). Also returns per-year breakdown.

    `periods` defaults to {2020, 2021, 2022, 2023, 2024, 2025}.

    This is the honest "what would the model have done if deployed at
    start of year Y" evaluation. Strictly tighter than GroupKFold: no
    other property's sale from year Y can leak the year's market level
    into training. Same uuid's pre-Y sale IS kept in training (it's
    observable at the deployment cutoff).
    """
    if periods is None:
        periods = sorted(set(df.loc[df["year_sold"] >= 2020, "year_sold"]))
    per_period = []
    apes_all = []
    for y in periods:
        cutoff = pd.Timestamp(f"{int(y)}-01-01")
        train_mask = (df["sold_date"] < cutoff).values
        test_mask  = (df["year_sold"] == y).values
        # Minimum sample guards
        if test_mask.sum() < 20 or train_mask.sum() < 500:
            continue
        gbr = gbr_default()
        gbr.fit(X[train_mask], y_log[train_mask])
        pred = np.exp(gbr.predict(X[test_mask]))
        actual = df.loc[test_mask, "sold_price"].values.astype(float)
        ape = np.abs(pred - actual) / actual
        apes_all.append(ape)
        per_period.append({
            "year": int(y),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "MAPE_pct": float(np.mean(ape) * 100),
            "within_10pct": float(np.mean(ape < 0.10) * 100),
        })
    apes = np.concatenate(apes_all)
    return {
        "label": label,
        "split": "temporal",
        "n_features": X.shape[1],
        "n_test": len(apes),
        "MAPE_pct":      float(np.mean(apes) * 100),
        "median_APE_pct":float(np.median(apes) * 100),
        "within_5pct":   float(np.mean(apes < 0.05) * 100),
        "within_10pct":  float(np.mean(apes < 0.10) * 100),
        "within_20pct":  float(np.mean(apes < 0.20) * 100),
        "per_period":    per_period,
    }


# ─────────── Battery ───────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outcodes", default="E15,W7")
    args = p.parse_args()
    outcodes = [o.strip().upper() for o in args.outcodes.split(",") if o.strip()]

    conn = sqlite3.connect(DB, timeout=30)

    conn.execute(f"ATTACH DATABASE '{SOLD_DB}' AS sold")
    conn.execute("PRAGMA query_only=1")
    uuids = pd.read_sql_query(
        f"SELECT DISTINCT rm_uuid FROM sold.rm_sold_properties "
        f"WHERE outcode IN ({','.join('?' * len(outcodes))})",
        conn, params=outcodes,
    )["rm_uuid"].tolist()
    lux_df = fetch_luxury_from_linux_v6(uuids)
    t0 = time.time()
    df = load_pool(conn, outcodes, lux_df)
    print(f"[pool built in {time.time()-t0:.1f}s]\n", flush=True)
    y_log = np.log(df["sold_price"].astype(float).values)

    # The combos we'll evaluate under BOTH splits.
    # Excludes prev_cagr everywhere (the leakage feature). Keeps v7
    # candidates that showed legitimate small gains in v7 ablation:
    # area_per_bed, street_n, street_ppsf, log_year, knn_price.
    combos = [
        ("v5 baseline (no groups)", set()),
        ("v6 best (5 groups)", V6_BEST),
        ("v6_best + area_per_bed", V6_BEST | {"area_per_bed"}),
        ("v6_best + street_n", V6_BEST | {"street_n"}),
        ("v6_best + street_ppsf", V6_BEST | {"street_ppsf"}),
        ("v6_best + log_year", V6_BEST | {"log_year"}),
        ("v6_best + knn_price", V6_BEST | {"knn_price"}),
        ("v6_best + top-3 v7-clean (area_per_bed,street_n,log_year)",
         V6_BEST | {"area_per_bed", "street_n", "log_year"}),
        ("v6_best + top-5 v7-clean (+street_ppsf,knn_price)",
         V6_BEST | {"area_per_bed", "street_n", "log_year",
                    "street_ppsf", "knn_price"}),
        ("v6_best - repeat_sales (control)", V6_BEST - {"repeat_sales"}),
    ]

    rows = []
    print(f"{'Variant':<58s}  {'GroupKFold':>11s}  {'Temporal':>11s}  {'Δ':>6s}", flush=True)
    print("-" * 95, flush=True)
    per_period_dump = {}
    for label, enabled in combos:
        X = make_features(df, set(enabled))
        rg = evaluate_groupkfold(X, y_log, df, label)
        rt = evaluate_temporal (X, y_log, df, label)
        gap = rt["MAPE_pct"] - rg["MAPE_pct"]
        print(f"{label:<58s}  {rg['MAPE_pct']:>9.2f}%  {rt['MAPE_pct']:>9.2f}%  "
              f"{gap:>+5.1f}pp", flush=True)
        rows.append({**rg, "temporal_MAPE": rt["MAPE_pct"]})
        per_period_dump[label] = rt["per_period"]

    # Detailed per-year for the headline combo
    print(f"\n=== per-year temporal breakdown ===", flush=True)
    print(f"{'Variant':<35s} {'year':>5s} {'n_train':>8s} {'n_test':>7s} {'MAPE':>7s} {'<10%':>6s}")
    print("-" * 80)
    for label in ["v5 baseline (no groups)", "v6 best (5 groups)",
                  "v6_best + top-3 v7-clean (area_per_bed,street_n,log_year)"]:
        for pp in per_period_dump.get(label, []):
            print(f"{label[:34]:<35s} {pp['year']:>5d} {pp['n_train']:>8,} {pp['n_test']:>7,} "
                  f"{pp['MAPE_pct']:>6.2f}% {pp['within_10pct']:>5.1f}%")

    return 0


if __name__ == "__main__":
    main()
