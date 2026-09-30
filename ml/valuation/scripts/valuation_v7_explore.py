#!/usr/bin/env python3
"""Valuation v7 — large feature-engineering exploration on E15+W7.

Goal: try ≥40 feature combinations + ≥10 new candidate features, pick
the best mix. Builds on v6 (which itself adds 7 groups over v5).

New v7 candidate groups (each toggleable):
  log_area     – log(floor_area_sqm)
  area_per_bed – sqft_per_bedroom (room density)
  bath_smart   – bathrooms imputed by (bed, type) conditional median
                 + bath_missing flag (rescue of the v5-dropped feature)
  month        – month_of_year (1-12, seasonality)
  prev_hpi     – prev sale price inflated to current date using
                 pool-median ppsf as a poor-man's HPI
  prev_cagr    – per-property annualised growth between first/last sale
  sale_rank    – which sale # this is for the rm_uuid (1, 2, 3…)
  hold_cap     – holding_months_capped at 60 (vs years_since_prev which
                 can grow huge for old anchors)
  has_floorplan – listing had a floorplan (1/0) — listing quality
  bed_x_type   – bedrooms × type one-hot interaction (4 cols)
  epc_x_type   – epc_rating_ord × type one-hot interaction
  street_ppsf  – street-level median £/sqm last 2y (joined via LR street)
  street_n     – street-level sales count last 2y
  knn_price    – per-row mean £/sqm of 5 nearest geographic comps
                 (same type, ±1 bed, weighted Haversine k=5)
  lux_quartile – binary: rm_uuid's mean_lux is in top quartile of outcode
  log_year     – log(year_sold - 2005)  (compress time non-linearly)

Plus all v6 groups: epc, area_fusion, new_build, dist_zone1,
                    repeat_sales, lux_extra, council_fix

Battery: tests baseline + each candidate alone (singles_add) + each
v6_best column dropped (singles_drop) + top pairs + full + best-mix
search. ≥50 combinations end-to-end. Sub-15-minute total runtime on
the existing 7k-row eval pool.

Usage:
    venv/bin/python3 scripts/valuation_v7_explore.py --outcodes E15,W7 --battery
    venv/bin/python3 scripts/valuation_v7_explore.py --enable repeat_sales,prev_cagr
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from itertools import combinations
from pathlib import Path

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
    R = 6371.0088
    lat1 = np.radians(lat1); lng1 = np.radians(lng1)
    lat2 = np.radians(lat2); lng2 = np.radians(lng2)
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def fetch_luxury_from_linux_v6(rm_uuids: list[str]) -> pd.DataFrame:
    if LUX_CACHE_V6.exists():
        return pd.read_pickle(LUX_CACHE_V6)
    # (cache should exist from v6 run; if not, error so user knows)
    raise RuntimeError("Run scripts/valuation_v6_gbr.py once first to "
                       "warm the luxury cache.")


def load_pool(conn, outcodes: list[str], lux_df: pd.DataFrame) -> pd.DataFrame:
    placeholders = ",".join("?" * len(outcodes))
    params = [o.upper() for o in outcodes]
    # Additional LR fields: street (for street-level aggregates), paon,
    # new_build. lr_transactions is large but we only need rows matching
    # our outcodes' postcodes — the LEFT JOIN's price+date+postcode key
    # is selective enough.
    sql = f"""
        SELECT
          p.rm_uuid, p.full_address, p.postcode, p.outcode,
          p.latitude, p.longitude,
          p.bedrooms, p.bathrooms,
          p.floor_area_sqm, p.listing_floor_area_sqm,
          p.energy_rating, p.energy_efficiency,
          p.property_type, p.tenure, p.council_tax_band,
          p.has_floorplan_scraped,
          t.sold_date, t.sold_price, t.lr_category,
          lr.new_build AS lr_new_build,
          lr.street    AS lr_street,
          lr.paon      AS lr_paon
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
    df["month_of_year"] = df["sold_date"].dt.month
    today = pd.Timestamp.today()
    df["months_since_sale"] = ((today - df["sold_date"]).dt.days / 30.44).clip(lower=0)

    # ── v6 reused: council fix ──
    df["council_ord_raw"] = df["council_tax_band"].map(COUNCIL_BAND_ORDINAL)
    council_median = df["council_ord_raw"].median()
    df["council_band_missing"] = df["council_ord_raw"].isna().astype(int)
    df["council_ord"] = df["council_ord_raw"].fillna(council_median)
    df["council_ord_v5"] = df["council_ord_raw"].fillna(-1)   # v5 baseline form

    # ── v6 reused: EPC ──
    df["epc_ord_raw"] = df["energy_rating"].map(EPC_RATING_ORDINAL)
    epc_ord_med = df["epc_ord_raw"].median()
    epc_eff_med = df["energy_efficiency"].median()
    df["epc_rating_missing"] = df["epc_ord_raw"].isna().astype(int)
    df["epc_rating_ord"] = df["epc_ord_raw"].fillna(epc_ord_med)
    df["epc_efficiency"] = df["energy_efficiency"].fillna(epc_eff_med)

    # ── v6 reused: area fusion ──
    df["sqm_disagree_15pct"] = 0
    both_mask = df["listing_floor_area_sqm"].notna() & df["floor_area_sqm"].notna()
    if both_mask.any():
        rel_diff = (df.loc[both_mask, "listing_floor_area_sqm"]
                    - df.loc[both_mask, "floor_area_sqm"]).abs() \
                   / df.loc[both_mask, "floor_area_sqm"]
        df.loc[both_mask, "sqm_disagree_15pct"] = (rel_diff > 0.15).astype(int)
    df["floor_area_fused"] = df["floor_area_sqm"]
    df.loc[both_mask, "floor_area_fused"] = (
        df.loc[both_mask, "floor_area_sqm"] + df.loc[both_mask, "listing_floor_area_sqm"]
    ) / 2

    # ── v6 reused: new_build ──
    df["new_build"] = (df["lr_new_build"] == "Y").astype(int)
    df["new_build_missing"] = df["lr_new_build"].isna().astype(int)

    # ── v6 reused: distance to zone 1 ──
    df["dist_zone1_km"] = haversine_km(
        df["latitude"].values, df["longitude"].values, ZONE1_LAT, ZONE1_LNG,
    )

    # ── merge SigLIP ──
    df = df.merge(lux_df, left_on="rm_uuid", right_on="property_id", how="left")

    # ── hard caps + SO filter ──
    df = df[(df["floor_area_sqm"] >= 15) & (df["floor_area_sqm"] <= 1000)
            & (df["sold_price"] >= 20_000) & (df["sold_price"] <= 10_000_000)].copy()
    df["ppsf"] = df["sold_price"] / df["floor_area_sqm"]
    df["decade"] = (df["year_sold"] // 5 * 5).astype(int)
    group_med = df.groupby(["type_bucket", "decade"])["ppsf"].transform("median")
    df = df[df["ppsf"] >= group_med * 0.5].copy().reset_index(drop=True)

    # ── v6 reused: repeat sales ──
    df = df.sort_values(["rm_uuid", "sold_date"]).reset_index(drop=True)
    df["prev_sold_price"] = df.groupby("rm_uuid")["sold_price"].shift(1)
    df["prev_sold_date"] = df.groupby("rm_uuid")["sold_date"].shift(1)
    df["has_prev_sale"] = df["prev_sold_price"].notna().astype(int)
    df["years_since_prev"] = ((df["sold_date"] - df["prev_sold_date"]).dt.days / 365.25).fillna(-1)
    df["prev_ppsf"] = df["prev_sold_price"] / df["floor_area_sqm"]
    df["prev_log_ppsf"] = np.log(df["prev_ppsf"].clip(lower=1)).fillna(0)
    df["prev_sold_price_log"] = np.log(df["prev_sold_price"].fillna(0).clip(lower=1))

    # ═══════════ v7 new candidate features ═══════════

    # log_area
    df["log_floor_area"] = np.log(df["floor_area_sqm"])

    # area_per_bed
    df["sqft_per_bedroom"] = df["floor_area_sqm"] / df["bedrooms"].clip(lower=1)

    # bath_smart — conditional median by (bedrooms, type)
    df["bath_missing"] = df["bathrooms"].isna().astype(int)
    df["bathrooms_smart"] = df.groupby(["type_bucket", "bedrooms"])["bathrooms"] \
                              .transform(lambda s: s.fillna(s.median()))
    # If group had no observation (all NaN), fall back to global median
    global_bath_med = df["bathrooms"].median()
    df["bathrooms_smart"] = df["bathrooms_smart"].fillna(global_bath_med)

    # month already added above

    # prev_hpi — inflate prev sale by ratio of pool-median ppsf at
    # (year_sold) vs (prev_year_sold). Crude HPI proxy; uses train+test
    # together for inflation series (acceptable as it's just a price
    # normalisation, not the target).
    year_med_ppsf = df.groupby("year_sold")["ppsf"].median()
    df["_curr_year_med_ppsf"] = df["year_sold"].map(year_med_ppsf)
    prev_year = df["prev_sold_date"].dt.year
    df["_prev_year_med_ppsf"] = prev_year.map(year_med_ppsf)
    inflation = df["_curr_year_med_ppsf"] / df["_prev_year_med_ppsf"]
    df["prev_sale_hpi_adjusted"] = (df["prev_sold_price"] * inflation).fillna(0)
    df["prev_sale_hpi_adjusted_log"] = np.log(df["prev_sale_hpi_adjusted"].clip(lower=1))

    # prev_cagr — annualised growth between this property's first & last
    # known sale dates (rolling, observable strictly before each row)
    df["_first_price"] = df.groupby("rm_uuid")["sold_price"].cummin().shift(1)
    df["_first_price_alt"] = df.groupby("rm_uuid")["sold_price"].cumsum().shift(1)  # unused, placeholder
    # Simpler: use just (prev_price → curr_price) annualised
    growth = (df["sold_price"] / df["prev_sold_price"])
    yrs = df["years_since_prev"].replace(0, np.nan)
    df["prev_annualised_growth"] = np.where(
        df["has_prev_sale"] == 1,
        (growth ** (1.0 / yrs.clip(lower=0.25))) - 1,
        0.0
    )
    df["prev_annualised_growth"] = df["prev_annualised_growth"].replace([np.inf, -np.inf], 0).fillna(0)

    # sale_rank — 1, 2, 3… within rm_uuid sorted by date
    df["sale_rank"] = df.groupby("rm_uuid").cumcount() + 1

    # hold_cap — holding period capped at 60 months
    months_since_prev = df["years_since_prev"] * 12
    df["holding_months_capped"] = months_since_prev.clip(lower=-1, upper=60).fillna(-1)

    # has_floorplan
    df["has_floorplan"] = df["has_floorplan_scraped"].fillna(0).astype(int)

    # bed_x_type — bedrooms × type_one-hot (4 interaction columns)
    type_dum = pd.get_dummies(df["type_bucket"], prefix="bxt_type")
    for c in type_dum.columns:
        df[f"bedXtype_{c.split('_')[-1]}"] = type_dum[c] * df["bedrooms"]
    bedXtype_cols = [c for c in df.columns if c.startswith("bedXtype_")]

    # epc_x_type — epc rating × type
    for c in type_dum.columns:
        df[f"epcXtype_{c.split('_')[-1]}"] = type_dum[c] * df["epc_rating_ord"]
    epcXtype_cols = [c for c in df.columns if c.startswith("epcXtype_")]

    # street_ppsf / street_n — street-level aggregates over the prior 2y
    # using LR street. For each row, compute median £/sqm of OTHER rows
    # on the same street whose sold_date < current row's sold_date AND
    # >= sold_date - 2 years. Done in Python per-row (slow on huge data
    # but our pool is ~7K rows; acceptable).
    df["lr_street_norm"] = df["lr_street"].fillna("").str.upper().str.strip()
    df["street_ppsf_2y_median"] = np.nan
    df["street_n_sales_2y"] = 0
    # Build a per-street index
    by_street = {}
    for i, r in df[["lr_street_norm", "sold_date", "ppsf"]].iterrows():
        if not r["lr_street_norm"]:
            continue
        by_street.setdefault(r["lr_street_norm"], []).append((r["sold_date"], r["ppsf"]))
    # Sort each street's list by date once
    for k in by_street:
        by_street[k].sort()
    for i, r in df.iterrows():
        s = r["lr_street_norm"]
        if not s:
            continue
        lst = by_street.get(s, [])
        cutoff_lo = r["sold_date"] - pd.Timedelta(days=730)
        recent = [p for (d, p) in lst if cutoff_lo <= d < r["sold_date"]]
        if recent:
            df.at[i, "street_ppsf_2y_median"] = np.median(recent)
            df.at[i, "street_n_sales_2y"] = len(recent)
    # Impute global median for streets with no recent comp
    st_med = df["street_ppsf_2y_median"].median()
    df["street_ppsf_2y_median"] = df["street_ppsf_2y_median"].fillna(st_med)

    # knn_price — for each row, find 5 nearest comparables by Haversine
    # (same type_bucket, ±1 bed, sold within 4y BEFORE this row), take
    # median £/sqm.
    # Brute O(N²) on 7K rows = ~50M ops — runs in a few seconds with numpy.
    print("[v7] computing knn_price (5-NN comps)...", flush=True)
    df["knn_price_ppsf"] = compute_knn_price(df)

    # lux_quartile — top 25% mean_lux in current outcode
    df["lux_outcode_q4"] = 0
    for oc in df["outcode"].unique():
        m = (df["outcode"] == oc) & df["mean_lux"].notna()
        if m.sum() > 4:
            q3 = df.loc[m, "mean_lux"].quantile(0.75)
            df.loc[m & (df["mean_lux"] >= q3), "lux_outcode_q4"] = 1

    # log_year — compress year linearly; small effect, mostly redundant
    df["log_year_sold"] = np.log(df["year_sold"] - 2005 + 1)

    print(f"[pool] {len(df):,} rows after joins+filters, "
          f"{df['has_prev_sale'].sum():,} with prev sale anchor, "
          f"{df['lr_street_norm'].astype(bool).sum():,} matched to LR street", flush=True)
    return df


def compute_knn_price(df: pd.DataFrame, k: int = 5,
                      look_back_years: int = 4) -> np.ndarray:
    """For each row, return median £/sqm of k nearest comparables that
    sold strictly BEFORE this row, within look_back_years, same type
    bucket, beds ±1. Brute O(N²). Returns array of length len(df).
    Fills global median when fewer than k comps."""
    n = len(df)
    out = np.full(n, np.nan)
    lats = df["latitude"].values
    lngs = df["longitude"].values
    beds = df["bedrooms"].values
    types = df["type_bucket"].values
    dates = df["sold_date"].values
    ppsf = df["ppsf"].values
    look_back_ns = np.timedelta64(look_back_years * 365, 'D')
    for i in range(n):
        mask = (
            (types == types[i])
            & (np.abs(beds - beds[i]) <= 1)
            & (dates < dates[i])
            & (dates >= dates[i] - look_back_ns)
        )
        if mask.sum() == 0:
            continue
        idx = np.flatnonzero(mask)
        d = haversine_km(lats[i], lngs[i], lats[idx], lngs[idx])
        if len(d) <= k:
            picks = idx
        else:
            top = np.argpartition(d, k)[:k]
            picks = idx[top]
        out[i] = np.median(ppsf[picks])
    # Fill global median for rows with no comp at all
    global_med = np.nanmedian(out)
    out[np.isnan(out)] = global_med
    return out


# ─────────── Feature group registry ───────────

# v5 baseline columns (always on)
V5_BASE_COLS = ["latitude", "longitude", "bedrooms", "floor_area_sqm",
                "year_sold", "months_since_sale", "council_ord_v5",
                # is_leasehold added in features fn
                ]
V5_LUX_COLS = ["n_photos", "interior_pct", "mean_lux", "max_lux", "interior_mean_lux"]

# Toggleable groups (v6 + v7)
GROUPS = {
    # v6 carry-over
    "council_fix":   ["council_ord", "council_band_missing"],
    "epc":           ["epc_rating_ord", "epc_efficiency", "epc_rating_missing"],
    "area_fusion":   ["floor_area_fused", "sqm_disagree_15pct"],
    "new_build":     ["new_build", "new_build_missing"],
    "lux_extra":     ["std_lux", "p25_lux", "p75_lux", "lux_spread", "mean_outdoor_cos"],
    "dist_zone1":    ["dist_zone1_km"],
    "repeat_sales":  ["has_prev_sale", "prev_sold_price_log",
                      "years_since_prev", "prev_log_ppsf"],
    # v7 new
    "log_area":      ["log_floor_area"],
    "area_per_bed":  ["sqft_per_bedroom"],
    "bath_smart":    ["bathrooms_smart", "bath_missing"],
    "month":         ["month_of_year"],
    "prev_hpi":      ["prev_sale_hpi_adjusted_log"],
    "prev_cagr":     ["prev_annualised_growth"],
    "sale_rank":     ["sale_rank"],
    "hold_cap":      ["holding_months_capped"],
    "has_floorplan": ["has_floorplan"],
    "bed_x_type":    [],  # filled at feature-make time (bedXtype_*)
    "epc_x_type":    [],  # same (epcXtype_*)
    "street_ppsf":   ["street_ppsf_2y_median"],
    "street_n":      ["street_n_sales_2y"],
    "knn_price":     ["knn_price_ppsf"],
    "lux_quartile":  ["lux_outcode_q4"],
    "log_year":      ["log_year_sold"],
}

# v6_best mix (from prior ablation)
V6_BEST = {"epc", "area_fusion", "new_build", "dist_zone1", "repeat_sales"}


def make_features(df: pd.DataFrame, enabled: set[str]) -> pd.DataFrame:
    parts = [df[V5_BASE_COLS].copy()]
    parts[0]["is_leasehold"] = (df["tenure"].fillna("") == "Leasehold").astype(int)

    # If council_fix enabled, prefer it over the v5 -1-sentinel version
    if "council_fix" in enabled:
        parts[0] = parts[0].drop(columns=["council_ord_v5"])
        parts.append(df[GROUPS["council_fix"]].copy())

    # Type one-hot (always on)
    parts.append(pd.get_dummies(df["type_bucket"], prefix="type"))

    # SigLIP baseline (always on)
    lux = df[V5_LUX_COLS].copy()
    lux["n_photos"] = lux["n_photos"].fillna(0)
    for c in ("interior_pct", "mean_lux", "max_lux", "interior_mean_lux"):
        lux[c] = lux[c].fillna(lux[c].median())
    parts.append(lux)

    # Toggleable groups
    for g, cols in GROUPS.items():
        if g == "council_fix":
            continue  # already handled
        if g not in enabled:
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

    X = pd.concat(parts, axis=1)
    return X


def evaluate(X: pd.DataFrame, y_log: np.ndarray, df: pd.DataFrame, label: str) -> dict:
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


# ─────────── Battery generator ───────────

V7_NEW = ["log_area", "area_per_bed", "bath_smart", "month", "prev_hpi",
          "prev_cagr", "sale_rank", "hold_cap", "has_floorplan",
          "bed_x_type", "epc_x_type", "street_ppsf", "street_n",
          "knn_price", "lux_quartile", "log_year"]


def run_battery(df: pd.DataFrame, y_log: np.ndarray) -> list[dict]:
    results = []

    def run(label, enabled):
        t0 = time.time()
        X = make_features(df, set(enabled))
        r = evaluate(X, y_log, df, label)
        r["secs"] = round(time.time() - t0, 1)
        print(f"  [{r['MAPE_pct']:5.2f}%  {r['n_features']:>3d}f  "
              f"{r['secs']:>4.1f}s] {label}", flush=True)
        results.append(r)

    print("\n=== Phase A: baselines ===", flush=True)
    run("v5 baseline (no groups)",           set())
    run("v6 best (5 groups)",                V6_BEST)
    run("v6 all (7 groups)", V6_BEST | {"council_fix", "lux_extra"})

    print("\n=== Phase B: v7 singles (each new feature added to v6_best) ===", flush=True)
    for g in V7_NEW:
        run(f"v6_best + {g}", V6_BEST | {g})

    print("\n=== Phase C: v6_best drop-one (find weakest current feature) ===", flush=True)
    for g in V6_BEST:
        run(f"v6_best - {g}", V6_BEST - {g})

    print("\n=== Phase D: pairs of top v7 singles ===", flush=True)
    # Use Phase B results to rank v7 singles
    phase_b = [r for r in results if r["label"].startswith("v6_best + ")
               and r["label"] != "v6_best + repeat_sales"]
    phase_b_sorted = sorted(phase_b, key=lambda r: r["MAPE_pct"])
    top_5_singles = [r["label"].split(" + ")[1] for r in phase_b_sorted[:5]]
    print(f"  (top 5 v7 singles by MAPE: {top_5_singles})", flush=True)
    for a, b in combinations(top_5_singles, 2):
        run(f"v6_best + {a} + {b}", V6_BEST | {a, b})

    print("\n=== Phase E: full kitchen sink + targeted exclusions ===", flush=True)
    all_groups = set(GROUPS.keys())
    run("ALL groups (every v6 + v7)", all_groups)
    run("ALL - lux_extra - council_fix", all_groups - {"lux_extra", "council_fix"})
    run("v6_best + ALL v7", V6_BEST | set(V7_NEW))
    # v6_best + top-3 v7
    top3 = set(top_5_singles[:3])
    run(f"v6_best + top-3 v7 ({','.join(sorted(top3))})", V6_BEST | top3)
    # v6_best + top-5 v7
    top5 = set(top_5_singles[:5])
    run(f"v6_best + top-5 v7", V6_BEST | top5)
    # v6_best + only v7 winners (those with -0.05pp or better single delta)
    v6_best_mape = next(r["MAPE_pct"] for r in results if r["label"] == "v6 best (5 groups)")
    winners = {g for g, r in [(l.split(" + ")[1], r) for l, r in
               [(r["label"], r) for r in results
                if r["label"].startswith("v6_best + ") and " + " not in r["label"][10:]]]
               if r["MAPE_pct"] < v6_best_mape - 0.02}
    if winners:
        run(f"v6_best + v7 winners ({','.join(sorted(winners))})", V6_BEST | winners)

    return results


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--outcodes", default="E15,W7")
    p.add_argument("--battery", action="store_true")
    p.add_argument("--enable", default="",
                   help="comma-sep group names (when not --battery)")
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

    t_pool = time.time()
    df = load_pool(conn, outcodes, lux_df)
    print(f"[pool built in {time.time()-t_pool:.1f}s]", flush=True)
    y_log = np.log(df["sold_price"].astype(float).values)

    if args.battery:
        results = run_battery(df, y_log)
        # Final ranked table
        print(f"\n{'='*100}", flush=True)
        print(f"FINAL RANKING ({len(results)} combinations on E15+W7)", flush=True)
        print(f"{'='*100}", flush=True)
        print(f"{'Rank':>4s}  {'MAPE':>7s}  {'feat':>4s}  {'<5%':>5s}  "
              f"{'<10%':>5s}  {'<20%':>5s}  {'bias':>6s}  Variant")
        print("-" * 100)
        v5_mape = next(r["MAPE_pct"] for r in results if r["label"].startswith("v5 baseline"))
        for rank, r in enumerate(sorted(results, key=lambda x: x["MAPE_pct"]), 1):
            delta = r["MAPE_pct"] - v5_mape
            marker = " " if delta >= 0 else "↓"
            print(f"{rank:>4d}{marker} {r['MAPE_pct']:>6.2f}%  {r['n_features']:>3d}f  "
                  f"{r['within_5pct']:>4.1f}%  {r['within_10pct']:>4.1f}%  "
                  f"{r['within_20pct']:>4.1f}%  "
                  f"{r['median_signed_pct']:>+5.1f}%  {r['label']}")
        print("="*100, flush=True)
    else:
        enabled = set(g.strip() for g in args.enable.split(",") if g.strip())
        X = make_features(df, enabled)
        r = evaluate(X, y_log, df, f"manual: {','.join(sorted(enabled)) or 'baseline'}")
        print(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
