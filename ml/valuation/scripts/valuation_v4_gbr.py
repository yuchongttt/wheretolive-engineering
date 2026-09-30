#!/usr/bin/env python3
"""E15 valuation model v4 — v3 + property-grouped CV + outlier hard caps.

v3 had two correctness issues:
  (1) Train/eval split via row-level KFold could land two sales of the
      same property in different folds — model memorises that property
      and the eval MAPE under-reports real error.
  (2) No sqft/price hard cap — a single data-entry slip (5 m² or
      £10M) would distort GBR's tree splits.

v4 fixes both:
  (1) GroupKFold by rm_uuid so all sales of a property go to the same
      fold; the model can never see a property's other sales during
      training.
  (2) sqft cap [15, 1000] m²; price cap [£20k, £10M]. Drop rows
      outside. These bounds are wide enough that nothing legitimate
      crosses them.

Other v3 features retained: SO/RtB ppsf-floor filter, GBR on
log(price), SigLIP luxury features, council band, bathrooms, etc.


Baseline (v1.2 kNN) gave MAPE 14.5% on E15. v2 adds:
  - bathrooms (from listing detail data — 65% coverage)
  - council_tax_band (ordinal A-H)
  - SigLIP-derived image features per property:
      * mean luxury score across all photos
      * mean luxury score restricted to interior photos
      * interior_pct (share of photos that are indoor)
      * photo_count (more photos = better listing maybe)
  - explicit year_sold + months_since_sale as time features

Model: sklearn GradientBoostingRegressor on log(price). One-hot
property_type, ordinal council band. Leave-one-out CV against the
same 2020+ test set as v1.2 so the numbers compare 1:1.

SigLIP features are read from the GPU box's dataset.db over ssh — one
roundtrip per script run pulls all rows then caches locally.

Usage:
    venv/bin/python3 scripts/valuation_v2_gbr.py [--outcode E15]
"""
from __future__ import annotations

import argparse
import json
import os
import math
import sqlite3
import subprocess
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

DB = Path(__file__).resolve().parents[1] / "data" / "evaluations.db"
SOLD_DB = Path(__file__).resolve().parents[1] / "data" / "sold.db"
# GPU box that holds the SigLIP image-score dataset (ssh target, e.g. "user@gpu-box").
LINUX = os.environ.get("GPU_HOST", "gpu-box")
LINUX_DB = "/data/ml/dataset/dataset.db"
LUX_CACHE = Path(__file__).resolve().parents[1] / "data" / "_luxury_cache.parquet"

COUNCIL_BAND_ORDINAL = {b: i for i, b in enumerate("ABCDEFGH")}

TYPE_BUCKETS = {
    'flat': ['flat', 'apartment', 'maisonette', 'penthouse', 'studio'],
    'terraced': ['terraced'],
    'semi':     ['semi'],
    'detached': ['detached'],
    'other':    ['bungalow', 'cottage', 'mews'],
}


def type_bucket(t):
    if not t: return None
    t = t.lower()
    for bucket, kws in TYPE_BUCKETS.items():
        if any(k in t for k in kws):
            return bucket
    return None


def fetch_luxury_from_linux(rm_uuids: list[str]) -> pd.DataFrame:
    """Pull image_luxury_score_v2 rows for the given property_ids.
    Returns DataFrame with columns: property_id, n_photos, n_interior,
    mean_luxury, interior_mean_luxury, max_luxury, interior_pct.
    """
    print(f"[lux] fetching SigLIP scores for {len(rm_uuids):,} properties from Linux", flush=True)
    tf = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
    tf.write("\n".join(rm_uuids))
    tf.close()
    subprocess.run(["scp", "-q", tf.name, f"{LINUX}:/tmp/_v2_pids.txt"], check=True)
    py = (
        'import sqlite3, json\n'
        f'c = sqlite3.connect("file:{LINUX_DB}?mode=ro", uri=True, timeout=20)\n'
        'c.execute("ATTACH DATABASE \\":memory:\\" AS mem")\n'
        'c.execute("CREATE TABLE mem.pids (pid TEXT PRIMARY KEY)")\n'
        'with open("/tmp/_v2_pids.txt") as f:\n'
        '    rows = [(l.strip(),) for l in f if l.strip()]\n'
        'c.executemany("INSERT OR IGNORE INTO mem.pids VALUES (?)", rows)\n'
        'q = """\n'
        '  SELECT property_id,\n'
        '         COUNT(*) AS n_photos,\n'
        '         SUM(is_interior) AS n_interior,\n'
        '         AVG(score) AS mean_lux,\n'
        '         MAX(score) AS max_lux,\n'
        '         AVG(CASE WHEN is_interior=1 THEN score END) AS interior_mean_lux\n'
        '  FROM image_luxury_score_v2\n'
        '  WHERE property_id IN (SELECT pid FROM mem.pids)\n'
        '  GROUP BY property_id\n'
        '"""\n'
        'out = []\n'
        'for r in c.execute(q):\n'
        '    out.append({\n'
        '        "property_id": r[0], "n_photos": r[1], "n_interior": r[2],\n'
        '        "mean_lux": r[3], "max_lux": r[4],\n'
        '        "interior_mean_lux": r[5],\n'
        '    })\n'
        'print(json.dumps(out))\n'
    )
    r = subprocess.run(["ssh", LINUX, "python3 -"], input=py,
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"linux pull failed: {r.stderr[:200]}")
    data = json.loads(r.stdout.strip().split("\n")[-1])
    df = pd.DataFrame(data)
    df["interior_pct"] = df["n_interior"] / df["n_photos"].clip(lower=1)
    print(f"[lux] {len(df):,} properties with SigLIP scores", flush=True)
    return df


def load_pool(conn, outcodes: list[str], lux_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (rm_uuid, sold_date, sold_price). Joins:
      - rm_sold_properties (lat/lng, beds, baths, sqm, type, council_tax_band)
      - rm_sold_transactions (sold_date, sold_price)
      - lux_df (SigLIP aggregates per property)
    Filters to rows with the core numeric features non-null.
    """
    placeholders = ",".join("?" * len(outcodes))
    sql = f"""
        SELECT
          p.rm_uuid, p.full_address, p.postcode, p.outcode,
          p.latitude, p.longitude,
          p.bedrooms, p.bathrooms, p.floor_area_sqm,
          p.property_type, p.tenure, p.council_tax_band,
          t.sold_date, t.sold_price, t.lr_category
        FROM sold.rm_sold_properties p
        JOIN sold.rm_sold_transactions t ON t.rm_uuid = p.rm_uuid
        WHERE p.outcode IN ({placeholders})
          AND p.bedrooms IS NOT NULL
          AND p.bedrooms > 0  -- beds=0/NULL are mislabel-prone; excluded from training. NOTE: current production model still needs a retrain to benefit.
          AND p.floor_area_sqm IS NOT NULL
          AND p.latitude IS NOT NULL
          AND p.property_type IS NOT NULL
          AND t.sold_date >= '2006-01-01'
          AND (t.lr_category IS NULL OR t.lr_category != 'B')
    """
    df = pd.read_sql_query(sql, conn, params=[o.upper() for o in outcodes])
    df["type_bucket"] = df["property_type"].apply(type_bucket)
    df = df[df["type_bucket"].notna()].copy()
    df["sold_date"] = pd.to_datetime(df["sold_date"])
    df["year_sold"] = df["sold_date"].dt.year
    today = pd.Timestamp.today()
    df["months_since_sale"] = ((today - df["sold_date"]).dt.days / 30.44).clip(lower=0)
    df["council_ord"] = df["council_tax_band"].map(COUNCIL_BAND_ORDINAL).fillna(-1)
    # Merge luxury features (left join — properties without photos get NaN → 0 later)
    df = df.merge(lux_df, left_on="rm_uuid", right_on="property_id", how="left")
    print(f"[pool] {len(df):,} rows after joins, "
          f"{df['mean_lux'].notna().sum():,} with SigLIP features", flush=True)
    # ----- v4: hard caps on sqft and sold_price to filter data errors -----
    # 15 m² is below any legal habitable studio (UK minimum ~37 m², we
    # leave a generous floor for studios/HMO rooms). 1000 m² is bigger
    # than any non-commercial flat-or-house in our outcode set. £20k
    # and £10M bracket valid market prices.
    SQFT_MIN, SQFT_MAX = 15, 1000
    PRICE_MIN, PRICE_MAX = 20_000, 10_000_000
    before = len(df)
    df = df[(df["floor_area_sqm"] >= SQFT_MIN) & (df["floor_area_sqm"] <= SQFT_MAX)
            & (df["sold_price"] >= PRICE_MIN) & (df["sold_price"] <= PRICE_MAX)].copy()
    print(f"[hard-caps] drop {before - len(df)} rows outside sqft [{SQFT_MIN}, {SQFT_MAX}] "
          f"or price [£{PRICE_MIN:,}, £{PRICE_MAX:,}]", flush=True)
    # ----- SO/RtB / family-transfer outlier filter -----
    # LR `category` only flags 6% of London sales as 'B', and per the
    # E15 sample our worst-predicted cases all
    # recorded as 'A' despite obvious sub-market prices. Use a ppsf
    # floor instead: drop sales whose £/sqm is below FLOOR_RATIO of the
    # property-type-stratified median over a recent window. This is
    # the same heuristic the kNN v1.2 used at predict time, applied
    # earlier (at training pool build) so the model never sees the
    # outliers.
    df["ppsf"] = df["sold_price"] / df["floor_area_sqm"]
    # Per (type_bucket, year-decade) median to absorb time inflation
    df["decade"] = (df["year_sold"] // 5 * 5).astype(int)
    group_med = df.groupby(["type_bucket", "decade"])["ppsf"].transform("median")
    floor_ratio = 0.5  # drop sales below 50% of the bucket median
    abnormal_mask = df["ppsf"] < (group_med * floor_ratio)
    n_dropped = int(abnormal_mask.sum())
    print(f"[ppsf-filter] drop {n_dropped} sales with ppsf < {floor_ratio*100:.0f}% of "
          f"(type, decade) median (SO/RtB/family-transfer proxy)", flush=True)
    df = df[~abnormal_mask].copy()
    df = df.reset_index(drop=True)
    return df


def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Returns (X, feature_names). One-hot type_bucket; ordinal council
    band; numeric lat/lng/beds/sqm/etc. NaN-fill luxury fields with the
    pool median so the model sees a sensible value instead of NaN."""
    parts = [df[["latitude", "longitude", "bedrooms", "floor_area_sqm",
                 "year_sold", "months_since_sale", "council_ord"]].copy()]
    parts[0]["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median())
    # Type bucket one-hot
    type_dum = pd.get_dummies(df["type_bucket"], prefix="type")
    parts.append(type_dum)
    # Luxury features — NaN means no photos; fill with median (0.5 is the
    # population center because score is sigmoid'd, but median is safer
    # under skew). photo_count fills with 0.
    lux_cols = ["n_photos", "interior_pct", "mean_lux", "max_lux", "interior_mean_lux"]
    lux_filled = df[lux_cols].copy()
    lux_filled["n_photos"] = lux_filled["n_photos"].fillna(0)
    for c in ("interior_pct", "mean_lux", "max_lux", "interior_mean_lux"):
        lux_filled[c] = lux_filled[c].fillna(lux_filled[c].median())
    parts.append(lux_filled)

    X = pd.concat(parts, axis=1)
    return X, list(X.columns)


def evaluate(X: pd.DataFrame, y_log: np.ndarray, df: pd.DataFrame,
             use_lux: bool, model_label: str) -> dict:
    """5-fold GroupKFold by rm_uuid.

    v3 used row-level KFold which leaked information: a property that
    sold in 2018 and 2023 could land 2018 in train and 2023 in eval,
    letting GBR memorise that property's level. v4 groups by rm_uuid
    so every sale of a property goes to the same fold — no leakage.

    Eval is still restricted to 2020+ sales (matches v1.2/v2/v3
    baselines). Pre-2020 sales of EVAL-fold properties are also
    excluded from training to keep the property fully unseen.
    """
    if not use_lux:
        X = X.drop(columns=[c for c in ("n_photos","interior_pct","mean_lux",
                                         "max_lux","interior_mean_lux") if c in X.columns])

    eval_mask = df["sold_date"] >= "2020-01-01"
    n_props_total = df["rm_uuid"].nunique()
    print(f"[eval/{model_label}] pool={len(X):,} · "
          f"properties={n_props_total:,} · "
          f"eval mask 2020+ = {eval_mask.sum():,}", flush=True)

    # Property-grouped 5-fold: shuffle eval-set rm_uuids, split into 5
    # roughly-equal property groups. For each fold:
    #   - test_rows = 2020+ sales of fold properties
    #   - train_rows = everything NOT a fold property (including
    #     non-eval sales of train properties)
    # This guarantees a property's 2018 sale doesn't leak into training
    # while its 2023 sale is in eval.
    eval_props = sorted(df.loc[eval_mask, "rm_uuid"].unique().tolist())
    np.random.seed(7)
    np.random.shuffle(eval_props)
    fold_props = np.array_split(eval_props, 5)
    print(f"[eval/{model_label}] eval properties split into 5 folds of "
          f"{[len(f) for f in fold_props]}", flush=True)

    groups = df["rm_uuid"].values
    apes = []
    signed_apes = []
    preds_full = np.full(len(df), np.nan)

    for fold_i, test_props in enumerate(fold_props):
        test_prop_set = set(test_props)
        test_rows = [i for i in range(len(df))
                     if groups[i] in test_prop_set and eval_mask.iloc[i]]
        train_rows = [i for i in range(len(df))
                      if groups[i] not in test_prop_set]
        if not test_rows or not train_rows:
            continue
        gbr = GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=42,
        )
        gbr.fit(X.iloc[train_rows], y_log[train_rows])
        pred_log = gbr.predict(X.iloc[test_rows])
        pred_price = np.exp(pred_log)
        actual = df.iloc[test_rows]["sold_price"].values.astype(float)
        ape_fold = np.abs(pred_price - actual) / actual
        apes.append(ape_fold)
        signed_apes.append((pred_price - actual) / actual)
        for k, i in enumerate(test_rows):
            preds_full[i] = pred_price[k]

    apes = np.concatenate(apes)
    signed = np.concatenate(signed_apes)
    metrics = {
        "model": model_label,
        "n_train_pool": len(X),
        "n_eval": len(apes),
        "MAPE_pct": float(np.mean(apes) * 100),
        "median_APE_pct": float(np.median(apes) * 100),
        "within_5pct":  float(np.mean(apes < 0.05) * 100),
        "within_10pct": float(np.mean(apes < 0.10) * 100),
        "within_20pct": float(np.mean(apes < 0.20) * 100),
        "median_signed_pct": float(np.median(signed) * 100),
    }
    df_eval = df[eval_mask].copy()
    df_eval["pred"] = preds_full[eval_mask.values]
    df_eval["ape"] = np.abs(df_eval["pred"] - df_eval["sold_price"]) / df_eval["sold_price"]
    df_eval = df_eval.sort_values("ape")
    print(f"  best 3:")
    for _, r in df_eval.head(3).iterrows():
        print(f"    ape={r['ape']*100:5.1f}% pred=£{int(r['pred'])//1000}k "
              f"actual=£{int(r['sold_price'])//1000}k  {r['full_address'][:60]}")
    print(f"  worst 3:")
    for _, r in df_eval.tail(3).iterrows():
        print(f"    ape={r['ape']*100:5.1f}% pred=£{int(r['pred'])//1000}k "
              f"actual=£{int(r['sold_price'])//1000}k  {r['full_address'][:60]}")
    return metrics


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--outcodes", default="E15",
                   help="comma-separated outcodes, e.g. E15,W7")
    p.add_argument("--no-lux", action="store_true",
                   help="ablate luxury features for comparison")
    args = p.parse_args()
    outcodes = [o.strip().upper() for o in args.outcodes.split(",") if o.strip()]

    conn = sqlite3.connect(DB)

    conn.execute(f"ATTACH DATABASE '{SOLD_DB}' AS sold")
    placeholders = ",".join("?" * len(outcodes))
    rm_uuids = [r[0] for r in conn.execute(
        f"SELECT rm_uuid FROM sold.rm_sold_properties WHERE outcode IN ({placeholders})",
        outcodes,
    )]
    print(f"[init] {len(rm_uuids):,} properties for outcodes={outcodes}", flush=True)

    lux_df = fetch_luxury_from_linux(rm_uuids)
    df = load_pool(conn, outcodes, lux_df)
    X, feat_names = make_features(df)
    y_log = np.log(df["sold_price"].astype(float).values)
    print(f"[features] {len(feat_names)}: {feat_names}", flush=True)

    # Run two side-by-side: ablation without luxury, full with luxury
    print("\n" + "="*60)
    print("ABLATION 1: no luxury features (tabular only)")
    print("="*60)
    m_tab = evaluate(X, y_log, df, use_lux=False, model_label="tabular_only")

    print("\n" + "="*60)
    print("FULL: tabular + SigLIP luxury features")
    print("="*60)
    m_full = evaluate(X, y_log, df, use_lux=True, model_label="tabular_plus_luxury")

    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"{'metric':<25} {'tabular':>10} {'+ luxury':>12} {'delta':>8}")
    print("-" * 60)
    for k in ("MAPE_pct","median_APE_pct","within_5pct","within_10pct","within_20pct","median_signed_pct"):
        t = m_tab[k]; f = m_full[k]
        delta_str = f"{f-t:+.2f}"
        print(f"{k:<25} {t:>10.2f} {f:>12.2f} {delta_str:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
