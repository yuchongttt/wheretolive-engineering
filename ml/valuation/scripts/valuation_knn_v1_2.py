#!/usr/bin/env python3
"""Spatial-kNN valuation model PoC v1.2.

Base method (from v1): for each subject, take the K nearest comparable
sales (same flat/house bucket, bedrooms ±1, Haversine distance),
time-adjust each price to the reference date with the local 5-year CAGR
from Land Registry repeat sales, and report the weighted P25/P50/P75
£/sqm × subject area. v1.1 added a 2006+ training pool and an
exp(-years_ago / tau) recency weight.

Changes vs v1.1:
  - SO/RtB outlier filter: drop candidates whose time-adjusted £/sqft
    is < (--floor-ratio, default 0.5) × the median £/sqft of the K=50
    nearest type+beds-matched sales. Catches shared-ownership,
    Right-to-Buy, family transfers without needing an LR `category` flag.
  - Also skip evaluating SUBJECTS that are themselves abnormal
    (their own ppsf < floor_ratio × local median): we don't want to
    score the model on SO sales the customer wouldn't ask about.

Usage:
    venv/bin/python3 scripts/valuation_knn_v1_2.py --outcode E15 \
        [--k 20] [--tau 3] [--min-year 2006] [--floor-ratio 0.5]
"""
from __future__ import annotations

import argparse
import math
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "data" / "evaluations.db"
SOLD_DB = Path(__file__).resolve().parents[1] / "data" / "sold.db"
TODAY = datetime.now().date()


def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def type_bucket(t):
    if not t:
        return None
    t = t.lower()
    if 'flat' in t or 'apartment' in t or 'maisonette' in t or 'penthouse' in t or 'studio' in t:
        return 'flat'
    if 'house' in t or 'terrac' in t or 'detached' in t or 'semi' in t or 'bungalow' in t or 'cottage' in t or 'mews' in t:
        return 'house'
    return None


def area_cagr_5y(conn, lat, lng, radius=1.0, min_n=30):
    """Median 5y annualised growth from LR repeat-sales."""
    lat_d = radius / 69.0
    lng_d = radius / (69.0 * math.cos(math.radians(lat)))
    rows = conn.execute(
        "SELECT first_price, last_price, first_date, last_date FROM lr_repeat_sales "
        "WHERE latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ? "
        "AND last_date > '2019' AND first_price > 0 AND last_price > 0",
        [lat - lat_d, lat + lat_d, lng - lng_d, lng + lng_d],
    ).fetchall()
    cagrs = []
    for fp, lp, fd, ld in rows:
        try:
            fdt = datetime.fromisoformat(str(fd).split()[0])
            ldt = datetime.fromisoformat(str(ld).split()[0])
            years = (ldt - fdt).days / 365.25
            if years < 0.5:
                continue
            c = ((lp / fp) ** (1 / years) - 1) * 100
            if -20 < c < 30:
                cagrs.append(c)
        except Exception:
            continue
    if len(cagrs) < min_n:
        return None
    cagrs.sort()
    return cagrs[len(cagrs) // 2]


def time_adjust(price, sold_date, cagr_pct, ref_date=None):
    """Project sold_price to ref_date using annualised cagr."""
    if cagr_pct is None:
        return price
    try:
        sd = datetime.fromisoformat(sold_date).date()
    except Exception:
        return price
    rd = ref_date or TODAY
    years = (rd - sd).days / 365.25
    return price * ((1 + cagr_pct / 100) ** years)


def load_training_pool(conn, outcode, min_year):
    """One row per (rm_uuid, sold_date, sold_price) — only complete-features
    with sold_date >= min_year-01-01.
    """
    return conn.execute("""
        SELECT
          p.rm_uuid, p.full_address, p.postcode,
          p.latitude, p.longitude,
          p.bedrooms, p.floor_area_sqm, p.property_type,
          t.sold_date, t.sold_price
        FROM sold.rm_sold_properties p
        JOIN sold.rm_sold_transactions t ON t.rm_uuid = p.rm_uuid
        WHERE p.outcode = ?
          AND p.bedrooms IS NOT NULL
          AND p.bedrooms > 0  -- beds=0/NULL are mislabel-prone; excluded from training. NOTE: current production model still needs a retrain to benefit.
          AND p.floor_area_sqm IS NOT NULL
          AND p.latitude IS NOT NULL
          AND p.property_type IS NOT NULL
          AND t.sold_date >= ?
    """, (outcode.upper(), f"{min_year}-01-01")).fetchall()


def local_ppsf_median(subject, pool, k_local=50):
    """Median raw £/sqft of the k_local nearest type+beds-matched sales.
    Used as the reference for SO/RtB outlier detection.
    """
    sb_lat, sb_lng = subject['lat'], subject['lng']
    sb_beds = subject['beds']
    sb_type = subject['type_bucket']
    cands = []
    for r in pool:
        if type_bucket(r['property_type']) != sb_type:
            continue
        if abs(r['bedrooms'] - sb_beds) > 1:
            continue
        if r['latitude'] is None or r['floor_area_sqm'] in (None, 0):
            continue
        d = haversine_miles(sb_lat, sb_lng, r['latitude'], r['longitude'])
        ppsf = r['sold_price'] / r['floor_area_sqm']
        if 0 < ppsf < 50000:
            cands.append((d, ppsf))
    if len(cands) < 10:
        return None
    cands.sort(key=lambda x: x[0])
    ppsfs = sorted(p for _, p in cands[:k_local])
    return ppsfs[len(ppsfs) // 2]


def predict(conn, subject, pool, k=20, exclude_uuid=None, cagr_cache=None,
            tau=4.0, trim=False, ref_date=None, floor_ratio=0.5,
            local_median_cache=None):
    """Return dict with low/mid/high estimate + diagnostic.

    subject: dict with lat, lng, beds, sqft, type_bucket
    pool: list of training rows
    """
    sb_lat, sb_lng = subject['lat'], subject['lng']
    sb_beds = subject['beds']
    sb_type = subject['type_bucket']

    # Filter: same bucket, beds ±1
    candidates = []
    for r in pool:
        if r['rm_uuid'] == exclude_uuid:
            continue
        if type_bucket(r['property_type']) != sb_type:
            continue
        if abs(r['bedrooms'] - sb_beds) > 1:
            continue
        if r['latitude'] is None:
            continue
        dist = haversine_miles(sb_lat, sb_lng, r['latitude'], r['longitude'])
        candidates.append((dist, r))

    if len(candidates) < 5:
        return {'low': None, 'mid': None, 'high': None,
                'n_comps': len(candidates), 'reason': 'too_few_comps'}

    candidates.sort(key=lambda x: x[0])
    top_k = candidates[:k]

    # Time-adjust each candidate's £/sqft to today
    cagr_key = (round(sb_lat, 3), round(sb_lng, 3))
    if cagr_cache is not None and cagr_key in cagr_cache:
        cagr = cagr_cache[cagr_key]
    else:
        cagr = area_cagr_5y(conn, sb_lat, sb_lng)
        if cagr_cache is not None:
            cagr_cache[cagr_key] = cagr

    # Compute the local ppsf median once per subject (for SO/RtB floor)
    if floor_ratio is not None and floor_ratio > 0:
        if local_median_cache is not None:
            lm_key = (subject['type_bucket'], sb_beds,
                      round(sb_lat, 3), round(sb_lng, 3))
            if lm_key in local_median_cache:
                local_median = local_median_cache[lm_key]
            else:
                local_median = local_ppsf_median(subject, pool)
                local_median_cache[lm_key] = local_median
        else:
            local_median = local_ppsf_median(subject, pool)
        floor_ppsf = (local_median * floor_ratio) if local_median else 0
    else:
        floor_ppsf = 0

    ppsf_values = []
    weights = []
    n_dropped_outlier = 0
    rd = ref_date or TODAY
    for dist, r in top_k:
        price_today = time_adjust(r['sold_price'], r['sold_date'], cagr, ref_date=rd)
        ppsf = price_today / r['floor_area_sqm']
        if ppsf <= 0 or ppsf > 50000:
            continue
        # Floor filter: drop candidates whose ppsf is implausibly low
        # (likely SO 25-40% / RtB / family transfer / short lease).
        if floor_ppsf and ppsf < floor_ppsf:
            n_dropped_outlier += 1
            continue
        try:
            sd = datetime.fromisoformat(r['sold_date']).date()
            yrs_ago = max(0.0, (rd - sd).days / 365.25)
        except Exception:
            yrs_ago = 0.0
        w = (1.0 / (dist + 0.05)) * math.exp(-yrs_ago / tau)
        ppsf_values.append(ppsf)
        weights.append(w)

    if not ppsf_values:
        return {'low': None, 'mid': None, 'high': None,
                'n_comps': 0, 'reason': 'all_clipped'}

    if trim and len(ppsf_values) >= 8:
        # Drop top/bottom 10% of ppsf (outlier guard: SO, RtB, short lease, auction)
        paired_sorted = sorted(zip(ppsf_values, weights))
        cut = max(1, len(paired_sorted) // 10)
        paired_sorted = paired_sorted[cut:-cut]
        ppsf_values = [p for p, _ in paired_sorted]
        weights = [w for _, w in paired_sorted]

    # Weighted percentiles (sort by ppsf, cumulative weight, interpolate)
    paired = sorted(zip(ppsf_values, weights))
    total_w = sum(w for _, w in paired)
    cum = 0
    p25 = p50 = p75 = None
    for ppsf, w in paired:
        cum += w
        f = cum / total_w
        if p25 is None and f >= 0.25:
            p25 = ppsf
        if p50 is None and f >= 0.5:
            p50 = ppsf
        if p75 is None and f >= 0.75:
            p75 = ppsf
            break
    p25 = p25 or paired[0][0]
    p50 = p50 or paired[len(paired)//2][0]
    p75 = p75 or paired[-1][0]

    return {
        'low':  int(p25 * subject['sqft']),
        'mid':  int(p50 * subject['sqft']),
        'high': int(p75 * subject['sqft']),
        'n_comps': len(top_k),
        'cagr_5y_pct': round(cagr, 2) if cagr else None,
        'median_ppsf': round(p50, 2),
    }


def evaluate_loocv(conn, pool, outcode, k=20, tau=4.0, trim=False,
                   eval_min_year=2020, floor_ratio=0.5, skip_abnormal_subjects=True):
    """Leave-one-out CV on sales >= eval_min_year."""
    recent = [r for r in pool if r['sold_date'] >= f"{eval_min_year}-01-01"]
    print(f"[eval] {len(recent)} sales >={eval_min_year} to evaluate "
          f"(k={k}, tau={tau}, trim={trim}, floor_ratio={floor_ratio})", flush=True)

    cagr_cache: dict = {}
    local_median_cache: dict = {}
    errors = []
    no_pred = 0
    skipped_abnormal = 0
    for r in recent:
        subject = {
            'lat': r['latitude'], 'lng': r['longitude'],
            'beds': r['bedrooms'], 'sqft': r['floor_area_sqm'],
            'type_bucket': type_bucket(r['property_type']),
        }
        if subject['type_bucket'] is None:
            no_pred += 1
            continue
        # Skip subjects whose own ppsf is implausibly low — these are
        # SO/RtB/family-transfer sales and not the use case we serve.
        if skip_abnormal_subjects and floor_ratio:
            lm = local_ppsf_median(subject, pool)
            if lm and (r['sold_price'] / r['floor_area_sqm']) < lm * floor_ratio:
                skipped_abnormal += 1
                continue
        try:
            ref = datetime.fromisoformat(r['sold_date']).date()
        except Exception:
            ref = TODAY
        result = predict(conn, subject, pool, k=k, exclude_uuid=r['rm_uuid'],
                         cagr_cache=cagr_cache, tau=tau, trim=trim, ref_date=ref,
                         floor_ratio=floor_ratio, local_median_cache=local_median_cache)
        if result['mid'] is None:
            no_pred += 1
            continue
        actual = r['sold_price']
        predicted = result['mid']
        ape = abs(predicted - actual) / actual
        err_signed = (predicted - actual) / actual
        errors.append((ape, err_signed, predicted, actual, r))

    if not errors:
        print("[eval] no predictions made")
        return

    errors.sort(key=lambda x: x[0])
    apes = [e[0] for e in errors]
    signed = [e[1] for e in errors]
    print()
    print(f"[result] predictions: {len(errors)} / {len(recent)} ({len(errors)*100/len(recent):.0f}%) · no_pred={no_pred} · skipped_abnormal={skipped_abnormal}")
    print(f"[result] MAPE:        {statistics.mean(apes)*100:.1f}%")
    print(f"[result] median APE:  {statistics.median(apes)*100:.1f}%")
    print(f"[result] within 10%:  {sum(1 for a in apes if a < 0.10)*100/len(apes):.0f}%")
    print(f"[result] within 20%:  {sum(1 for a in apes if a < 0.20)*100/len(apes):.0f}%")
    print(f"[result] bias (med):  {statistics.median(signed)*100:+.1f}%   (negative = under-predicting)")

    print("\n[sample] best 5:")
    for ape, signed_e, pred, actual, r in errors[:5]:
        print(f"  ape={ape*100:5.1f}% pred=£{pred//1000}k actual=£{int(actual)//1000}k  {r['property_type']} {r['bedrooms']}bed {int(r['floor_area_sqm'])}m²  {r['full_address'][:50]}")

    print("\n[sample] worst 5:")
    for ape, signed_e, pred, actual, r in errors[-5:]:
        print(f"  ape={ape*100:5.1f}% pred=£{pred//1000}k actual=£{int(actual)//1000}k  {r['property_type']} {r['bedrooms']}bed {int(r['floor_area_sqm'])}m²  {r['full_address'][:50]}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--outcode", default="E15")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--tau", type=float, default=4.0,
                   help="time decay constant in years (lower = more recency-biased)")
    p.add_argument("--min-year", type=int, default=2006,
                   help="training pool starts at this year")
    p.add_argument("--eval-min-year", type=int, default=2020,
                   help="LOOCV evaluates sales from this year on")
    p.add_argument("--trim", action="store_true",
                   help="drop ppsf top/bottom 10%% before percentile compute")
    p.add_argument("--floor-ratio", type=float, default=0.5,
                   help="exclude comparables with ppsf below this ratio of local median (0 disables)")
    p.add_argument("--keep-abnormal-subjects", action="store_true",
                   help="don't skip evaluating low-ppsf SO/RtB subjects")
    args = p.parse_args()

    conn = sqlite3.connect(DB)

    conn.execute(f"ATTACH DATABASE '{SOLD_DB}' AS sold")
    conn.row_factory = sqlite3.Row
    pool = load_training_pool(conn, args.outcode, args.min_year)
    print(f"[init] {len(pool)} training rows for {args.outcode} "
          f"(sold_date >= {args.min_year}, complete features)", flush=True)
    print(f"[init] k={args.k} tau={args.tau} trim={args.trim}", flush=True)

    t0 = time.time()
    evaluate_loocv(conn, pool, args.outcode, k=args.k,
                   tau=args.tau, trim=args.trim, eval_min_year=args.eval_min_year,
                   floor_ratio=args.floor_ratio,
                   skip_abnormal_subjects=not args.keep_abnormal_subjects)
    print(f"\n[done] elapsed {time.time()-t0:.0f}s", flush=True)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
