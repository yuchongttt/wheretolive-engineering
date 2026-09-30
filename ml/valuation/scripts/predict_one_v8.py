#!/usr/bin/env python3
"""One-off v8 inference for a single property (research, not deployed).

Trains v8_best (= v6_best + area_per_bed) GBR on all 20 trained outcodes
using everything sold up to today, then predicts a single hand-built row.
Caches the fitted model + training schema so subsequent predictions are
fast (skip the ~5-min pool build + GBR fit).

(Extract note: the site's valuation API does call this script with --json,
as one "sanity bound" input to an LLM-written valuation.)

Caveats:
- Luxury (VLM) features are median-imputed when the target isn't in the
  listing dataset — model treats visual quality as "average".
- In-sample MAPE ~14% so point estimate has ±10-15% confidence band.

Usage (example values):
    .venv/bin/python3 scripts/predict_one_v8.py \\
        --postcode "EC4M 8AD" --lat 51.514 --lng -0.098 \\
        --sqm 60 --bedrooms 2 --epc C --epc-eff 72 \\
        [--prev-price 400000 --prev-date 2019-06-01]
    .venv/bin/python3 scripts/predict_one_v8.py --rebuild-cache
"""
from __future__ import annotations

import argparse
import math
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from valuation_v7_explore import (  # noqa: E402
    V6_BEST, load_pool, fetch_luxury_from_linux_v6, make_features,
    EPC_RATING_ORDINAL, ZONE1_LAT, ZONE1_LNG, haversine_km,
)
from valuation_v8_temporal import gbr_default  # noqa: E402

DB = ROOT / "data" / "evaluations.db"
SOLD_DB = ROOT / "data" / "sold.db"
CACHE = ROOT / "data" / "_v8_model_cache.pkl"

TRAIN_OUTCODES = [
    "E14", "SW18", "SW11", "N1", "SE1", "DA1", "E3", "EN1", "HA8", "W2",
    "SE10", "NW1", "E15", "CR5", "N17", "RM10", "N7", "EN4", "W7", "TW10",
]
V8_BEST = V6_BEST | {"area_per_bed"}


def train_and_cache():
    print("[1/4] querying training uuids…", flush=True)
    conn = sqlite3.connect(DB, timeout=30)
    conn.execute(f"ATTACH DATABASE '{SOLD_DB}' AS sold")
    conn.execute("PRAGMA query_only=1")
    uuids = pd.read_sql_query(
        f"SELECT DISTINCT rm_uuid FROM sold.rm_sold_properties "
        f"WHERE outcode IN ({','.join('?'*len(TRAIN_OUTCODES))})",
        conn, params=TRAIN_OUTCODES,
    )["rm_uuid"].tolist()
    print(f"      {len(uuids):,} uuids", flush=True)

    print("[2/4] fetching luxury VLM features from the GPU box…", flush=True)
    lux_df = fetch_luxury_from_linux_v6(uuids)
    print(f"      {len(lux_df):,} rows with VLM data", flush=True)

    print("[3/4] loading + feature-engineering pool…", flush=True)
    df = load_pool(conn, TRAIN_OUTCODES, lux_df)
    print(f"      {len(df):,} training rows", flush=True)

    y_log = np.log(df["sold_price"].astype(float).values)
    X = make_features(df, set(V8_BEST))
    print(f"      {X.shape[1]} features", flush=True)

    print("[4/4] fitting GBR…", flush=True)
    gbr = gbr_default()
    gbr.fit(X, y_log)
    in_sample = np.exp(gbr.predict(X))
    ape = np.abs(in_sample - df["sold_price"]) / df["sold_price"]
    print(f"      in-sample MAPE {ape.mean()*100:.2f}%  "
          f"medAPE {np.median(ape)*100:.2f}%", flush=True)

    medians = {
        "n_photos":          float(df["n_photos"].median()),
        "interior_pct":      float(df["interior_pct"].median()),
        "mean_lux":          float(df["mean_lux"].median()),
        "max_lux":           float(df["max_lux"].median()),
        "interior_mean_lux": float(df["interior_mean_lux"].median()),
    }
    cache = {
        "gbr": gbr,
        "feature_cols": list(X.columns),
        "medians": medians,
        "in_sample_mape": float(ape.mean() * 100),
    }
    with open(CACHE, "wb") as f:
        pickle.dump(cache, f)
    print(f"      cached → {CACHE.name}", flush=True)
    return cache


def predict_one(args, cache):
    today = pd.Timestamp.today()
    medians = cache["medians"]
    feature_cols = cache["feature_cols"]
    gbr = cache["gbr"]

    has_prev = args.prev_price is not None and args.prev_date is not None
    prev_date = pd.Timestamp(args.prev_date) if has_prev else None
    prev_price = float(args.prev_price) if has_prev else 0.0

    row = {
        "latitude": args.lat,
        "longitude": args.lng,
        "bedrooms": args.bedrooms,
        "floor_area_sqm": args.sqm,
        "year_sold": today.year,
        "months_since_sale": 0.0,
        "council_ord_v5": -1,
        "tenure": "Leasehold",
        "property_type": "Flat",
        "type_bucket": "F",
        "n_photos":          medians["n_photos"],
        "interior_pct":      medians["interior_pct"],
        "mean_lux":          medians["mean_lux"],
        "max_lux":           medians["max_lux"],
        "interior_mean_lux": medians["interior_mean_lux"],
        "energy_rating": args.epc,
        "energy_efficiency": args.epc_eff,
        "epc_rating_ord": EPC_RATING_ORDINAL.get(args.epc, 2),
        "epc_efficiency": args.epc_eff,
        "epc_rating_missing": 0,
        "listing_floor_area_sqm": np.nan,
        "floor_area_fused": args.sqm,
        "sqm_disagree_15pct": 0,
        "lr_new_build": "N",
        "new_build": 0,
        "new_build_missing": 0,
        "dist_zone1_km": float(haversine_km(
            np.array([args.lat]), np.array([args.lng]), ZONE1_LAT, ZONE1_LNG)[0]),
        "prev_sold_price": prev_price if has_prev else 0.0,
        "prev_sold_date": prev_date if has_prev else pd.NaT,
        "has_prev_sale": 1 if has_prev else 0,
        "years_since_prev": (today - prev_date).days / 365.25 if has_prev else -1,
        "prev_log_ppsf": math.log(prev_price / args.sqm) if has_prev else 0.0,
        "prev_sold_price_log": math.log(prev_price) if has_prev else 0.0,
        "sqft_per_bedroom": args.sqm / max(1, args.bedrooms),
        "bathrooms": args.bathrooms or 1,
        "bath_missing": 0 if args.bathrooms else 1,
        "bathrooms_smart": args.bathrooms or 1,
    }
    target_df = pd.DataFrame([row])
    X_target = make_features(target_df, set(V8_BEST))
    for c in feature_cols:
        if c not in X_target.columns:
            X_target[c] = 0
    X_target = X_target[feature_cols]
    pred = float(np.exp(gbr.predict(X_target)[0]))
    return pred


def build_target_row(args, cache):
    """Build the feature dict for the target property (mirror of predict_one)."""
    today = pd.Timestamp.today()
    medians = cache["medians"]
    has_prev = args.prev_price is not None and args.prev_date is not None
    prev_date = pd.Timestamp(args.prev_date) if has_prev else None
    prev_price = float(args.prev_price) if has_prev else 0.0
    return {
        "latitude": args.lat, "longitude": args.lng,
        "bedrooms": args.bedrooms, "floor_area_sqm": args.sqm,
        "year_sold": today.year, "months_since_sale": 0.0,
        "council_ord_v5": -1, "tenure": "Leasehold",
        "property_type": "Flat", "type_bucket": "F",
        "n_photos": medians["n_photos"], "interior_pct": medians["interior_pct"],
        "mean_lux": medians["mean_lux"], "max_lux": medians["max_lux"],
        "interior_mean_lux": medians["interior_mean_lux"],
        "energy_rating": args.epc, "energy_efficiency": args.epc_eff,
        "epc_rating_ord": EPC_RATING_ORDINAL.get(args.epc, 2),
        "epc_efficiency": args.epc_eff, "epc_rating_missing": 0,
        "listing_floor_area_sqm": np.nan, "floor_area_fused": args.sqm,
        "sqm_disagree_15pct": 0, "lr_new_build": "N", "new_build": 0,
        "new_build_missing": 0,
        "dist_zone1_km": float(haversine_km(
            np.array([args.lat]), np.array([args.lng]), ZONE1_LAT, ZONE1_LNG)[0]),
        "prev_sold_price": prev_price if has_prev else 0.0,
        "prev_sold_date": prev_date if has_prev else pd.NaT,
        "has_prev_sale": 1 if has_prev else 0,
        "years_since_prev": (today - prev_date).days / 365.25 if has_prev else -1,
        "prev_log_ppsf": math.log(prev_price / args.sqm) if has_prev else 0.0,
        "prev_sold_price_log": math.log(prev_price) if has_prev else 0.0,
        "sqft_per_bedroom": args.sqm / max(1, args.bedrooms),
        "bathrooms": args.bathrooms or 1,
        "bath_missing": 0 if args.bathrooms else 1,
        "bathrooms_smart": args.bathrooms or 1,
    }


def predict_from_row(row, cache):
    """Run prediction given a fully-built target row dict. Returns £ price."""
    gbr = cache["gbr"]
    feature_cols = cache["feature_cols"]
    X = make_features(pd.DataFrame([row]), set(V8_BEST))
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_cols]
    return float(np.exp(gbr.predict(X)[0]))


# Baseline values for counterfactual analysis — represents an "average London property"
# Each feature group, when swapped to its baseline, removes that feature's contribution
# to the prediction. Delta (actual - cf) = what that group adds/subtracts from price.
def _baseline_overrides(row, group, cache):
    """Return a copy of row with one feature group reset to neutral baseline."""
    medians = cache["medians"]
    cf = dict(row)
    today = pd.Timestamp.today()

    if group == "location":
        # Neutral location = at zone 1 boundary (dist=8km, a typical outer-zone-2 spot)
        # Setting both lat/lng AND dist_zone1_km to break geographic signal
        cf["latitude"] = 51.5074
        cf["longitude"] = -0.1278
        cf["dist_zone1_km"] = 0.0
    elif group == "size":
        # Neutral = 65 sqm (median London flat size)
        cf["floor_area_sqm"] = 65.0
        cf["floor_area_fused"] = 65.0
        cf["sqft_per_bedroom"] = 65.0 / max(1, cf.get("bedrooms", 1))
    elif group == "beds":
        # Neutral = 2 bed (most common)
        cf["bedrooms"] = 2
        cf["sqft_per_bedroom"] = cf.get("floor_area_sqm", 65) / 2
    elif group == "epc":
        # Neutral = EPC D (median London rating)
        cf["energy_rating"] = "D"
        cf["epc_rating_ord"] = EPC_RATING_ORDINAL.get("D", 3)
        cf["epc_efficiency"] = 60
        cf["energy_efficiency"] = 60
    elif group == "prev_sale":
        # Neutral = no prior sale recorded
        cf["prev_sold_price"] = 0.0
        cf["prev_sold_date"] = pd.NaT
        cf["has_prev_sale"] = 0
        cf["years_since_prev"] = -1
        cf["prev_log_ppsf"] = 0.0
        cf["prev_sold_price_log"] = 0.0
    elif group == "type":
        # Neutral = Flat (most common in London)
        cf["property_type"] = "Flat"
        cf["type_bucket"] = "F"
    elif group == "photos":
        # Neutral = median VLM signals (no quality signal — model already does this)
        for k in ("n_photos", "interior_pct", "mean_lux", "max_lux", "interior_mean_lux"):
            cf[k] = medians[k]
    return cf


def compute_contributions(row, cache):
    """Counterfactual feature contributions: predict with each group reset to
    baseline, delta from actual prediction = that group's £ contribution.

    Returns: dict with actual_pred, baseline_pred (all groups neutral), and a
    list of {group, label, contribution_pounds, contribution_pct} entries."""
    actual_pred = predict_from_row(row, cache)

    # Full baseline: ALL groups reset (median London property)
    cf_full = row
    for g in ("location", "size", "beds", "epc", "prev_sale", "type", "photos"):
        cf_full = _baseline_overrides(cf_full, g, cache)
    baseline_pred = predict_from_row(cf_full, cache)

    # Per-group contributions
    groups = [
        ("location",  "Location"),
        ("size",      "Size"),
        ("beds",      "Bedrooms"),
        ("epc",       "Energy rating"),
        ("prev_sale", "Recent sale history"),
        ("type",      "Property type"),
        ("photos",    "Visual quality (photos)"),
    ]
    out = []
    for g, label in groups:
        cf = _baseline_overrides(row, g, cache)
        cf_pred = predict_from_row(cf, cache)
        delta = actual_pred - cf_pred
        out.append({
            "group": g,
            "label": label,
            "contribution_pounds": round(delta, 2),
            "contribution_pct": round(delta / actual_pred * 100, 2),
        })
    # Sort by absolute contribution descending
    out.sort(key=lambda x: abs(x["contribution_pounds"]), reverse=True)

    return {
        "actual_pred": round(actual_pred, 2),
        "baseline_pred": round(baseline_pred, 2),
        "contributions": out,
    }


def main():
    import json as _json
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--postcode", required=False, default="(unspecified)")
    p.add_argument("--lat", type=float)
    p.add_argument("--lng", type=float)
    p.add_argument("--sqm", type=float)
    p.add_argument("--bedrooms", type=int)
    p.add_argument("--bathrooms", type=int, default=0)
    p.add_argument("--epc", default="C")
    p.add_argument("--epc-eff", type=int, default=70)
    p.add_argument("--prev-price", type=float, default=None)
    p.add_argument("--prev-date", default=None, help="YYYY-MM-DD")
    p.add_argument("--json", action="store_true",
                   help="Emit a single JSON line with prediction + per-feature contributions")
    args = p.parse_args()

    if args.rebuild_cache or not CACHE.exists():
        cache = train_and_cache()
    else:
        with open(CACHE, "rb") as f:
            cache = pickle.load(f)
        if not args.json:
            print(f"loaded cached model (in-sample MAPE {cache['in_sample_mape']:.2f}%)",
                  flush=True)

    if args.lat is None:
        if args.json:
            print(_json.dumps({"error": "no prediction requested"}))
        else:
            print("(no prediction requested — pass --lat/--lng/--sqm/--bedrooms etc.)")
        return

    if args.json:
        # JSON mode: emit prediction + counterfactual feature contributions
        row = build_target_row(args, cache)
        breakdown = compute_contributions(row, cache)
        sqft = args.sqm * 10.7639
        payload = {
            "predicted_price": breakdown["actual_pred"],
            "baseline_price": breakdown["baseline_pred"],
            "ppsf": round(breakdown["actual_pred"] / sqft, 2),
            "in_sample_mape": cache["in_sample_mape"],
            "feature_contributions": breakdown["contributions"],
            "inputs": {
                "postcode": args.postcode, "sqm": args.sqm, "sqft": round(sqft, 1),
                "bedrooms": args.bedrooms, "epc": args.epc, "epc_efficiency": args.epc_eff,
            },
        }
        print(_json.dumps(payload))
        return

    pred = predict_one(args, cache)
    print()
    print(f"=== {args.postcode} · {args.sqm:.0f} sqm · {args.bedrooms}-bed · EPC {args.epc} ===")
    print(f"  predicted price : £{pred:,.0f}")
    print(f"  £/sqft          : £{pred / (args.sqm * 10.7639):,.0f}")
    if args.prev_price:
        diff_pct = (pred - args.prev_price) / args.prev_price * 100
        yrs = (pd.Timestamp.today() - pd.Timestamp(args.prev_date)).days / 365.25
        cagr = ((pred / args.prev_price) ** (1/yrs) - 1) * 100 if yrs > 0 else 0
        print(f"  vs prev (£{args.prev_price:,.0f} on {args.prev_date}): "
              f"{diff_pct:+.1f}%  ({cagr:+.1f}% CAGR)")


if __name__ == "__main__":
    main()
