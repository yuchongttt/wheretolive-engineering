#!/usr/bin/env python3
"""Valuation model v6 — v5 + 7 data-side enrichments + ablation harness.

Each feature group can be toggled on/off via flags so we measure
contribution in isolation. Default run = all on. `--ablate` mode runs
v5_baseline + each group alone + v6_full → 9 evaluations + pretty table.

The 7 enrichments:
  (1) Repeat-sales anchor: prev_sold_price + years_since_prev +
      prev_log_ppsf + has_prev_sale. 52% of E15+W7 UUIDs have ≥2 sales.
      Strictly-earlier-only lag → no future leakage. The cleanest
      single feature for a property's "level".
  (2) EPC: energy_rating (A-G → ordinal 0-6) + energy_efficiency
      (0-100 score). 75% coverage; missing → median impute + missing flag.
  (3) postcode location: distance_to_zone1_km (great-circle from
      property lat/lng to King's Cross 51.5308°N, -0.1238°W — the
      conventional "centre" for London property valuation). Raw
      postcode one-hot rejected (1099 unique × 10K rows = sparse).
  (4) LR new_build (Y/N): joined via (postcode, sold_date, sold_price)
      tuple match. ~85% coverage.
  (5) SigLIP extra aggregates: std_lux, p25_lux, p75_lux, lux_spread
      (max-mean), mean_outdoor_cos. Re-aggregated from image_luxury_score_v2.
  (6) Area fusion: when listing_floor_area_sqm (listing, 9.9% coverage) AND
      floor_area_sqm (EPC) both present, use their average. Add
      sqm_source_disagree flag when they differ >15%.
  (7) Council tax NA fix: replace -1 sentinel with median imputation +
      `council_band_missing` 1/0 flag. Old -1 collided with band A=0 in
      tree splits.

Usage:
    venv/bin/python3 scripts/valuation_v6_gbr.py --outcodes E15,W7
    venv/bin/python3 scripts/valuation_v6_gbr.py --outcodes E15,W7 --ablate
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
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

DB = Path(__file__).resolve().parents[1] / "data" / "evaluations.db"
SOLD_DB = Path(__file__).resolve().parents[1] / "data" / "sold.db"
# GPU box that holds the SigLIP image-score dataset (ssh target, e.g. "user@gpu-box").
LINUX = os.environ.get("GPU_HOST", "gpu-box")
LINUX_DB = "/data/ml/dataset/dataset.db"
LUX_CACHE_V6 = Path(__file__).resolve().parents[1] / "data" / "_luxury_cache_v6.pkl"

COUNCIL_BAND_ORDINAL = {b: i for i, b in enumerate("ABCDEFGH")}
EPC_RATING_ORDINAL = {b: i for i, b in enumerate("ABCDEFG")}

# King's Cross — the canonical London "centre" for property research
# (LR uses it for distance bands; TfL fare zone 1 north-edge).
ZONE1_LAT, ZONE1_LNG = 51.5308, -0.1238

TYPE_BUCKETS = {
    'flat':     ['flat', 'apartment', 'maisonette', 'penthouse', 'studio'],
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


def haversine_km(lat1, lng1, lat2, lng2):
    """Vectorised great-circle distance (km). lat/lng in degrees."""
    R = 6371.0088
    lat1 = np.radians(lat1); lng1 = np.radians(lng1)
    lat2 = np.radians(lat2); lng2 = np.radians(lng2)
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


# ─────────── SigLIP aggregates (v6: extended) ───────────

def fetch_luxury_from_linux_v6(rm_uuids: list[str]) -> pd.DataFrame:
    """Pull image_luxury_score_v2 rows aggregated per property.
    v6 adds: std_lux, p25_lux, p75_lux, lux_spread, mean_outdoor_cos.
    """
    if LUX_CACHE_V6.exists():
        print(f"[lux/v6] cache hit: {LUX_CACHE_V6}", flush=True)
        return pd.read_pickle(LUX_CACHE_V6)

    print(f"[lux/v6] fetching extended SigLIP aggregates for {len(rm_uuids):,} properties", flush=True)
    tf = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
    tf.write("\n".join(rm_uuids))
    tf.close()
    subprocess.run(["scp", "-q", tf.name, f"{LINUX}:/tmp/_v6_pids.txt"], check=True)
    py = '''
import sqlite3, json, statistics
c = sqlite3.connect("file:''' + LINUX_DB + '''?mode=ro", uri=True, timeout=30)
c.execute("ATTACH DATABASE ':memory:' AS mem")
c.execute("CREATE TABLE mem.pids (pid TEXT PRIMARY KEY)")
with open("/tmp/_v6_pids.txt") as f:
    rows = [(l.strip(),) for l in f if l.strip()]
c.executemany("INSERT OR IGNORE INTO mem.pids VALUES (?)", rows)

# Pull raw per-image rows then aggregate in Python (SQLite has no native
# percentile fn). 33K rows × ~5 fields = trivial.
q = """
  SELECT property_id, score, indoor_cos, outdoor_cos, is_interior
  FROM image_luxury_score_v2
  WHERE property_id IN (SELECT pid FROM mem.pids)
"""
per_pid = {}
for pid, score, ind, out, is_int in c.execute(q):
    per_pid.setdefault(pid, []).append((score, ind, out, is_int))

out_rows = []
for pid, recs in per_pid.items():
    scores = sorted([r[0] for r in recs])
    indoors = [r[1] for r in recs]
    outdoors = [r[2] for r in recs]
    interiors = [r[0] for r in recs if r[3] == 1]
    n = len(scores)
    if n == 0: continue
    mean_lux = sum(scores) / n
    std_lux = (sum((s - mean_lux)**2 for s in scores) / n) ** 0.5
    p25 = scores[int(n * 0.25)]
    p75 = scores[int(min(n - 1, n * 0.75))]
    out_rows.append({
        "property_id": pid,
        "n_photos": n,
        "n_interior": sum(1 for r in recs if r[3] == 1),
        "mean_lux": mean_lux,
        "max_lux": max(scores),
        "interior_mean_lux": (sum(interiors) / len(interiors)) if interiors else None,
        # v6 additions
        "std_lux": std_lux,
        "p25_lux": p25,
        "p75_lux": p75,
        "lux_spread": max(scores) - mean_lux,
        "mean_outdoor_cos": sum(outdoors) / len(outdoors),
    })
print(json.dumps(out_rows))
'''
    r = subprocess.run(["ssh", LINUX, "python3 -"], input=py,
                       capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f"linux pull failed: {r.stderr[:300]}")
    data = json.loads(r.stdout.strip().split("\n")[-1])
    df = pd.DataFrame(data)
    df["interior_pct"] = df["n_interior"] / df["n_photos"].clip(lower=1)
    df.to_pickle(LUX_CACHE_V6)
    print(f"[lux/v6] {len(df):,} properties cached → {LUX_CACHE_V6}", flush=True)
    return df


# ─────────── Pool loading + LR new_build join + derived features ───────────

def load_pool(conn, outcodes: list[str], lux_df: pd.DataFrame) -> pd.DataFrame:
    """v6 adds energy_rating, energy_efficiency, listing_floor_area_sqm,
    postcode (for distance), and LEFT JOIN lr_transactions for new_build.
    """
    placeholders = ",".join("?" * len(outcodes))
    params = [o.upper() for o in outcodes]
    sql = f"""
        SELECT
          p.rm_uuid, p.full_address, p.postcode, p.outcode,
          p.latitude, p.longitude,
          p.bedrooms, p.bathrooms,
          p.floor_area_sqm, p.listing_floor_area_sqm,
          p.energy_rating, p.energy_efficiency,
          p.property_type, p.tenure, p.council_tax_band,
          t.sold_date, t.sold_price, t.lr_category,
          lr.new_build AS lr_new_build
        FROM sold.rm_sold_properties p
        JOIN sold.rm_sold_transactions t ON t.rm_uuid = p.rm_uuid
        LEFT JOIN lr_transactions lr
          ON lr.postcode = p.postcode
         AND lr.date = t.sold_date
         AND lr.price = t.sold_price
        WHERE p.outcode IN ({placeholders})
          AND p.bedrooms IS NOT NULL
          AND p.bedrooms > 0  -- beds=0/NULL are mislabel-prone; excluded from training. NOTE: current production model still needs a retrain to benefit.
          AND p.floor_area_sqm IS NOT NULL
          AND p.latitude IS NOT NULL
          AND p.property_type IS NOT NULL
          AND t.sold_date >= '2006-01-01'
          AND (t.lr_category IS NULL OR t.lr_category != 'B')
    """
    df = pd.read_sql_query(sql, conn, params=params)
    df["type_bucket"] = df["property_type"].apply(type_bucket)
    df = df[df["type_bucket"].notna()].copy()
    df["sold_date"] = pd.to_datetime(df["sold_date"])
    df["year_sold"] = df["sold_date"].dt.year
    today = pd.Timestamp.today()
    df["months_since_sale"] = ((today - df["sold_date"]).dt.days / 30.44).clip(lower=0)

    # ─── (7) council tax: ordinal + missing flag (no more -1 sentinel) ───
    df["council_ord_raw"] = df["council_tax_band"].map(COUNCIL_BAND_ORDINAL)
    council_median = df["council_ord_raw"].median()
    df["council_band_missing"] = df["council_ord_raw"].isna().astype(int)
    df["council_ord"] = df["council_ord_raw"].fillna(council_median)

    # ─── (2) EPC: ordinal rating + numeric efficiency + missing flags ───
    df["epc_ord_raw"] = df["energy_rating"].map(EPC_RATING_ORDINAL)
    epc_ord_med = df["epc_ord_raw"].median()
    epc_eff_med = df["energy_efficiency"].median()
    df["epc_rating_missing"] = df["epc_ord_raw"].isna().astype(int)
    df["epc_rating_ord"] = df["epc_ord_raw"].fillna(epc_ord_med)
    df["epc_efficiency"] = df["energy_efficiency"].fillna(epc_eff_med)

    # ─── (6) area fusion: average EPC + listing sqm when both exist ───
    df["sqm_disagree_15pct"] = 0
    both_mask = df["listing_floor_area_sqm"].notna() & df["floor_area_sqm"].notna()
    if both_mask.any():
        rel_diff = (df.loc[both_mask, "listing_floor_area_sqm"]
                    - df.loc[both_mask, "floor_area_sqm"]).abs() \
                   / df.loc[both_mask, "floor_area_sqm"]
        df.loc[both_mask, "sqm_disagree_15pct"] = (rel_diff > 0.15).astype(int)
    # Use the average when both, fall back to whichever exists, default to EPC.
    df["floor_area_fused"] = df["floor_area_sqm"]
    df.loc[both_mask, "floor_area_fused"] = (
        df.loc[both_mask, "floor_area_sqm"] + df.loc[both_mask, "listing_floor_area_sqm"]
    ) / 2

    # ─── (4) new_build: 'Y' → 1, 'N' → 0, NULL → 0 + missing flag ───
    df["new_build"] = (df["lr_new_build"] == "Y").astype(int)
    df["new_build_missing"] = df["lr_new_build"].isna().astype(int)

    # ─── (3) postcode location: distance to King's Cross (zone 1) ───
    df["dist_zone1_km"] = haversine_km(
        df["latitude"].values, df["longitude"].values, ZONE1_LAT, ZONE1_LNG,
    )

    # ─── merge SigLIP features ───
    df = df.merge(lux_df, left_on="rm_uuid", right_on="property_id", how="left")
    print(f"[pool] {len(df):,} rows after joins, "
          f"{df['mean_lux'].notna().sum():,} with SigLIP, "
          f"{df['lr_new_build'].notna().sum():,} matched to LR new_build", flush=True)

    # ─── v4 hard caps ───
    SQFT_MIN, SQFT_MAX = 15, 1000
    PRICE_MIN, PRICE_MAX = 20_000, 10_000_000
    before = len(df)
    df = df[(df["floor_area_sqm"] >= SQFT_MIN) & (df["floor_area_sqm"] <= SQFT_MAX)
            & (df["sold_price"] >= PRICE_MIN) & (df["sold_price"] <= PRICE_MAX)].copy()
    print(f"[hard-caps] drop {before - len(df)} rows outside bounds", flush=True)

    # ─── SO/RtB ppsf filter (v3+) ───
    df["ppsf"] = df["sold_price"] / df["floor_area_sqm"]
    df["decade"] = (df["year_sold"] // 5 * 5).astype(int)
    group_med = df.groupby(["type_bucket", "decade"])["ppsf"].transform("median")
    abnormal_mask = df["ppsf"] < (group_med * 0.5)
    print(f"[ppsf-filter] drop {int(abnormal_mask.sum())} rows below 50% (type,decade) median", flush=True)
    df = df[~abnormal_mask].copy().reset_index(drop=True)

    # ─── (1) repeat-sales anchor: lag previous sale within same rm_uuid ───
    # MUST sort by (uuid, date) so shift(1) gives strictly-earlier sale.
    # No future-leakage: each row's anchor is a sale that completed BEFORE
    # its own sold_date. GroupKFold handles cross-uuid leakage; this
    # in-uuid history is legitimate context for prediction.
    df = df.sort_values(["rm_uuid", "sold_date"]).reset_index(drop=True)
    df["prev_sold_price"] = df.groupby("rm_uuid")["sold_price"].shift(1)
    df["prev_sold_date"] = df.groupby("rm_uuid")["sold_date"].shift(1)
    df["has_prev_sale"] = df["prev_sold_price"].notna().astype(int)
    df["years_since_prev"] = ((df["sold_date"] - df["prev_sold_date"]).dt.days / 365.25)
    df["years_since_prev"] = df["years_since_prev"].fillna(-1)
    # prev_log_ppsf — log £/sqm of the previous sale (using same property's
    # floor area; if floor changed we don't know that, accept the noise).
    df["prev_ppsf"] = df["prev_sold_price"] / df["floor_area_sqm"]
    df["prev_log_ppsf"] = np.log(df["prev_ppsf"].clip(lower=1)).fillna(0)
    df["prev_sold_price_log"] = np.log(df["prev_sold_price"].fillna(0).clip(lower=1))
    # Restore original ordering side-effect-free (eval splits use df[mask])
    df = df.reset_index(drop=True)
    print(f"[repeat-sales] {df['has_prev_sale'].sum():,} / {len(df):,} rows "
          f"({df['has_prev_sale'].mean()*100:.0f}%) have a prior sale anchor", flush=True)
    return df


# ─────────── Feature assembly with per-group toggles ───────────

V5_BASE_COLS = ["latitude", "longitude", "bedrooms", "floor_area_sqm",
                "year_sold", "months_since_sale"]
V5_LUX_COLS = ["n_photos", "interior_pct", "mean_lux", "max_lux", "interior_mean_lux"]

# Each enrichment group as a dict of {column: imputation_strategy}
ENRICHMENTS = {
    "council_fix":  ["council_ord", "council_band_missing"],         # (7)
    "epc":          ["epc_rating_ord", "epc_efficiency",
                     "epc_rating_missing"],                          # (2)
    "area_fusion":  ["floor_area_fused", "sqm_disagree_15pct"],      # (6)
    "new_build":    ["new_build", "new_build_missing"],              # (4)
    "lux_extra":    ["std_lux", "p25_lux", "p75_lux",
                     "lux_spread", "mean_outdoor_cos"],              # (5)
    "dist_zone1":   ["dist_zone1_km"],                               # (3)
    "repeat_sales": ["has_prev_sale", "prev_sold_price_log",
                     "years_since_prev", "prev_log_ppsf"],           # (1)
}


def make_features(df: pd.DataFrame, enabled: set[str]) -> tuple[pd.DataFrame, list[str]]:
    """Build the feature matrix. `enabled` is a set of enrichment group
    names; everything outside the set is omitted. v5 baseline always
    includes V5_BASE_COLS + V5_LUX_COLS + is_leasehold + type one-hot +
    council_ord (raw, with -1 sentinel — replaced by 'council_fix' group).
    """
    parts = []
    # Baseline (v5) numeric features. Note we exclude council_ord here when
    # council_fix is enabled — to avoid duplicating the column.
    base = df[V5_BASE_COLS].copy()
    base["is_leasehold"] = (df["tenure"].fillna("") == "Leasehold").astype(int)
    parts.append(base)

    # Council: v5 used -1 sentinel; council_fix replaces with median impute
    # plus missing flag.
    if "council_fix" in enabled:
        parts.append(df[ENRICHMENTS["council_fix"]].copy())
    else:
        cc = df[["council_ord_raw"]].copy()
        cc["council_ord"] = cc["council_ord_raw"].fillna(-1)
        parts.append(cc[["council_ord"]])

    # Type one-hot
    parts.append(pd.get_dummies(df["type_bucket"], prefix="type"))

    # SigLIP baseline (v5)
    lux = df[V5_LUX_COLS].copy()
    lux["n_photos"] = lux["n_photos"].fillna(0)
    for c in ("interior_pct", "mean_lux", "max_lux", "interior_mean_lux"):
        lux[c] = lux[c].fillna(lux[c].median())
    parts.append(lux)

    # Enrichment groups
    for group in ("epc", "area_fusion", "new_build", "lux_extra",
                  "dist_zone1", "repeat_sales"):
        if group not in enabled:
            continue
        cols = ENRICHMENTS[group]
        sub = df[cols].copy()
        # Generic median fill for any straggler NaN in the group's columns
        for c in cols:
            if sub[c].isna().any():
                med = sub[c].median()
                if pd.isna(med):
                    med = 0
                sub[c] = sub[c].fillna(med)
        parts.append(sub)

    X = pd.concat(parts, axis=1)
    # Drop council_ord_raw if it slipped through
    if "council_ord_raw" in X.columns:
        X = X.drop(columns=["council_ord_raw"])
    return X, list(X.columns)


# ─────────── Eval (unchanged from v5: GroupKFold by rm_uuid, 2020+ test) ───────────

def evaluate(X: pd.DataFrame, y_log: np.ndarray, df: pd.DataFrame,
             label: str) -> dict:
    eval_mask = df["sold_date"] >= "2020-01-01"
    eval_props = sorted(df.loc[eval_mask, "rm_uuid"].unique().tolist())
    rng = np.random.RandomState(7)
    rng.shuffle(eval_props)
    fold_props = np.array_split(eval_props, 5)
    groups = df["rm_uuid"].values

    apes = []; signed = []
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
        signed.append((pred_price - actual) / actual)
    apes = np.concatenate(apes)
    signed = np.concatenate(signed)
    return {
        "label": label,
        "n_features": X.shape[1],
        "n_eval": len(apes),
        "MAPE_pct": float(np.mean(apes) * 100),
        "median_APE_pct": float(np.median(apes) * 100),
        "within_5pct":  float(np.mean(apes < 0.05) * 100),
        "within_10pct": float(np.mean(apes < 0.10) * 100),
        "within_20pct": float(np.mean(apes < 0.20) * 100),
        "median_signed_pct": float(np.median(signed) * 100),
    }


# ─────────── Main / CLI / ablation ───────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--outcodes", default="E15,W7",
                   help="comma-separated outcodes (default E15,W7)")
    p.add_argument("--ablate", action="store_true",
                   help="run baseline + each enrichment alone + all-together "
                        "for contribution measurement")
    p.add_argument("--enable", default="",
                   help="when not --ablate, run with only these groups enabled "
                        "(comma-separated). Empty = all groups.")
    args = p.parse_args()
    outcodes = [o.strip().upper() for o in args.outcodes.split(",") if o.strip()]

    conn = sqlite3.connect(DB, timeout=30)

    conn.execute(f"ATTACH DATABASE '{SOLD_DB}' AS sold")
    conn.execute("PRAGMA query_only=1")

    # First-pass uuid list for SigLIP cache fetch
    uuids = pd.read_sql_query(
        f"SELECT DISTINCT rm_uuid FROM sold.rm_sold_properties "
        f"WHERE outcode IN ({','.join('?' * len(outcodes))})",
        conn, params=outcodes,
    )["rm_uuid"].tolist()
    lux_df = fetch_luxury_from_linux_v6(uuids)

    df = load_pool(conn, outcodes, lux_df)
    y_log = np.log(df["sold_price"].astype(float).values)

    all_groups = list(ENRICHMENTS.keys())

    if args.ablate:
        results = []
        # v5 baseline reproduction (no enrichments)
        X, cols = make_features(df, set())
        results.append(evaluate(X, y_log, df, "v5 baseline"))

        # Each enrichment alone
        for grp in all_groups:
            X, _ = make_features(df, {grp})
            results.append(evaluate(X, y_log, df, f"+ {grp} only"))

        # All together
        X, _ = make_features(df, set(all_groups))
        results.append(evaluate(X, y_log, df, "v6 all-on"))

        # Print pretty table
        base_mape = results[0]["MAPE_pct"]
        print()
        print(f"{'Variant':<28s}  {'feat':>4s}  {'MAPE':>7s}  {'Δvs base':>9s}  "
              f"{'med APE':>8s}  {'<5%':>5s}  {'<10%':>5s}  {'<20%':>5s}  {'bias':>6s}")
        print("-" * 98)
        for r in results:
            delta = r["MAPE_pct"] - base_mape
            print(f"{r['label']:<28s}  {r['n_features']:>4d}  "
                  f"{r['MAPE_pct']:>6.2f}%  {delta:>+8.2f}pp  "
                  f"{r['median_APE_pct']:>7.2f}%  "
                  f"{r['within_5pct']:>4.1f}%  {r['within_10pct']:>4.1f}%  "
                  f"{r['within_20pct']:>4.1f}%  "
                  f"{r['median_signed_pct']:>+5.1f}%")
        print()
    else:
        enabled = set(g.strip() for g in args.enable.split(",") if g.strip()) \
                  if args.enable else set(all_groups)
        invalid = enabled - set(all_groups)
        if invalid:
            print(f"unknown groups: {invalid}", file=sys.stderr); return 2
        label = f"v6 (groups={','.join(sorted(enabled)) or 'baseline'})"
        X, cols = make_features(df, enabled)
        r = evaluate(X, y_log, df, label)
        print(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
