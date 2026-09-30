#!/usr/bin/env python3
"""Detailed per-year breakdown of v8_best (v6_best + area_per_bed)
under temporal walk-forward eval. Reports 5%/10%/20% bracket hit rates
and bias per year — cleanest evaluation available (post-leakage-fix).
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
)
from valuation_v8_temporal import make_features, gbr_default

DB = Path(__file__).resolve().parents[1] / "data" / "evaluations.db"
SOLD_DB = Path(__file__).resolve().parents[1] / "data" / "sold.db"

V8_BEST = V6_BEST | {"area_per_bed"}


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

    X = make_features(df, set(V8_BEST))
    print(f"v8_best = v6_best + area_per_bed · {X.shape[1]} features\n", flush=True)

    # Temporal walk-forward per year, with full bracket stats
    periods = sorted(set(df.loc[df["year_sold"] >= 2020, "year_sold"]))
    rows = []
    print(f"{'Year':>5s}  {'n_train':>8s}  {'n_test':>6s}  "
          f"{'MAPE':>7s}  {'medAPE':>7s}  {'<5%':>6s}  {'<10%':>6s}  "
          f"{'<20%':>6s}  {'bias':>7s}", flush=True)
    print("-" * 80, flush=True)
    all_apes = []; all_signed = []
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
        ape = np.abs(pred - actual) / actual
        signed = (pred - actual) / actual
        all_apes.append(ape)
        all_signed.append(signed)
        print(f"{int(y):>5d}  {train_mask.sum():>8,}  {test_mask.sum():>6,}  "
              f"{ape.mean()*100:>6.2f}%  {np.median(ape)*100:>6.2f}%  "
              f"{(ape < 0.05).mean()*100:>5.1f}%  "
              f"{(ape < 0.10).mean()*100:>5.1f}%  "
              f"{(ape < 0.20).mean()*100:>5.1f}%  "
              f"{np.median(signed)*100:>+6.1f}%", flush=True)
        rows.append({
            "year": int(y),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "MAPE_pct": float(ape.mean() * 100),
            "median_APE_pct": float(np.median(ape) * 100),
            "within_5pct":  float((ape < 0.05).mean() * 100),
            "within_10pct": float((ape < 0.10).mean() * 100),
            "within_20pct": float((ape < 0.20).mean() * 100),
            "median_signed_pct": float(np.median(signed) * 100),
        })

    apes = np.concatenate(all_apes)
    signed = np.concatenate(all_signed)
    print("-" * 80, flush=True)
    print(f"{'ALL':>5s}  {'·':>8s}  {len(apes):>6,}  "
          f"{apes.mean()*100:>6.2f}%  {np.median(apes)*100:>6.2f}%  "
          f"{(apes < 0.05).mean()*100:>5.1f}%  "
          f"{(apes < 0.10).mean()*100:>5.1f}%  "
          f"{(apes < 0.20).mean()*100:>5.1f}%  "
          f"{np.median(signed)*100:>+6.1f}%", flush=True)

    return 0


if __name__ == "__main__":
    main()
