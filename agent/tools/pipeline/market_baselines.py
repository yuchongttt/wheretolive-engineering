#!/usr/bin/env python3
"""Controlled market baselines — make the wrong seasonality answer structurally impossible.

**Why this exists** (2026-08-18): asked "which month has the most London
listings / the best prices" and working from raw sales data, the chat agent
wrote a query over 2019-2025 with `AVG(price)` grouped by month,
saw completion-volume spikes in March/June/September/December, and **invented
an industry explanation: "UK law firms prefer batching completions at quarter
end"**. The real cause: the three stamp-duty-holiday deadlines (2021-03-31 /
06-30 / 09-30) and the 2025-03-31 threshold change happen to fall at quarter
ends — June 2021 alone ran at **3.22x** its neighbouring months.

It got more than the number wrong; it invented causation. So contamination is
**removed**, not documented for the model to be careful about: documentation
relies on the model reading and obeying it, which is probabilistic; removal
is structural.

Three layers, each solving a different problem:

  1. lr_month_quality — automatic detection of distorted months (not a hand
     list). Over 31 years / 372 months it flags 11, zero false positives, all
     nameable events: the 2009 financial crisis, the March 2016 surcharge
     rush, the April/May 2020 lockdown, the three 2021 stamp-duty deadlines +
     the following-month hangover, the March 2025 threshold deadline.
     **Automatic detection strictly beats enumeration**: a hand blacklist
     would drop 4 whole years (48 months), this drops 11 — and it caught 2009,
     which I hadn't listed, plus the **reverse hangover the month after** each
     deadline.

  2. v_lr_clean — the clean view. Distorted months + category 'B'
     (repossessions / transfers not at market value) + trailing months whose
     filing is incomplete simply do not exist here. What the agent can't see,
     it can't make a story about.

  3. lr_seasonality — the correct primitive, materialised. **Removing rows
     does not remove the methodological trap**: even with every row clean,
     `AVG(price) GROUP BY month` in a rising market still counts trend as
     seasonality (I fell for it myself: 6.6% was really 4.3%). So detrending
     (centred 12-month rolling baseline) + medians + per-property-type split
     must be precomputed and sitting there.

     This is not freezing an answer back into code: it is a **primitive**, not
     an answer — the agent can slice it by type or combine it with CAGR to
     answer "should I wait?", uses we never anticipated.

Anything materialised goes stale (clean_ppsf_premium silently going stale with
no schedule is a mistake we already made), so there is a staleness gate: past
MAX_AGE_DAYS, consumers must report "baseline is stale" instead of answering
as normal.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Data directory holding evaluations.db: $WTL_DATA_DIR, else ../data (next to the tools).
DATA_DIR = Path(os.environ.get("WTL_DATA_DIR") or Path(__file__).resolve().parents[1] / "data")
DB_PATH = DATA_DIR / "evaluations.db"

# Distortion test: the month's sale count / the median of the neighbouring 12
# months. Thresholds calibrated on real data — 1.40/0.65 flag only 11 of 372
# months, all of them nameable (see the module docstring).
SPIKE_RATIO, COLLAPSE_RATIO = 1.40, 0.65
# Trailing months have no "next 6 months" to compare against, only the past;
# HMLR filing lags by months, so use a more conservative threshold here —
# better to use a few months less than to read incomplete filing as a crash.
TAIL_RATIO = 0.85
WINDOW = 6                 # centred-window radius (months)
MIN_TYPE_SALES = 200       # a type-month below this does not enter the index
MAX_AGE_DAYS = 45          # older than this = stale

TYPES = ("F", "T", "S", "D")

# Detection is automatic (the safety net); naming is manual (optional
# annotation). Why separate: automatic detection catches events we never
# listed (that's how 2009 was found), while inventing a name for an anomaly we
# don't actually recognise would import the very fabrication we're guarding
# against into the DB. Unrecognised → leave NULL; consumers then say "cause
# unknown".
KNOWN_EVENTS = {
    "2009-01": "financial-crisis trough",
    "2009-02": "financial-crisis trough",
    "2016-03": "rush before the 3% additional-property surcharge (1 Apr 2016)",
    "2020-04": "first COVID lockdown — market effectively closed",
    "2020-05": "first COVID lockdown — market effectively closed",
    "2021-03": "stamp-duty holiday original deadline (31 Mar 2021)",
    "2021-06": "stamp-duty holiday extended deadline (30 Jun 2021)",
    "2021-07": "hangover month after the 30 Jun 2021 deadline",
    "2021-09": "stamp-duty taper ended (30 Sep 2021)",
    "2025-03": "rush before the 1 Apr 2025 threshold change",
    "2025-04": "hangover month after the Apr 2025 threshold change",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS lr_month_quality (
  ym TEXT PRIMARY KEY, n INTEGER NOT NULL, ratio REAL,
  is_distorted INTEGER NOT NULL DEFAULT 0, reason TEXT, event TEXT,
  computed_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS lr_seasonality (
  property_type TEXT NOT NULL, month INTEGER NOT NULL,
  price_index REAL NOT NULL, volume_index REAL NOT NULL,
  n_months INTEGER NOT NULL, n_sales INTEGER NOT NULL, computed_at TEXT NOT NULL,
  PRIMARY KEY (property_type, month));
"""

CLEAN_VIEW = """
DROP VIEW IF EXISTS v_lr_clean;
CREATE VIEW v_lr_clean AS
  SELECT t.* FROM lr_transactions t
  WHERE t.category = 'A'
    AND substr(t.date, 1, 7) NOT IN
        (SELECT ym FROM lr_month_quality WHERE is_distorted = 1);
"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _monthly_counts(conn) -> list[tuple[str, int]]:
    return conn.execute(
        "SELECT substr(date,1,7) ym, COUNT(*) FROM lr_transactions"
        " WHERE category='A' AND date IS NOT NULL GROUP BY ym ORDER BY ym").fetchall()


def compute_quality(conn) -> list[tuple]:
    rows = _monthly_counts(conn)
    yms = [r[0] for r in rows]
    cnt = dict(rows)
    out = []
    for i, ym in enumerate(yms):
        lo, hi = i - WINDOW, i + WINDOW
        if lo >= 0 and hi < len(yms):
            win = [cnt[yms[j]] for j in range(lo, hi + 1) if j != i]
            base = statistics.median(win)
            ratio = cnt[ym] / base if base else None
            if ratio is None:
                flag, reason = 0, None
            elif ratio > SPIKE_RATIO:
                flag, reason = 1, "volume_spike"
            elif ratio < COLLAPSE_RATIO:
                flag, reason = 1, "volume_collapse"
            else:
                flag, reason = 0, None
        elif hi >= len(yms) and i >= 12:
            # Trailing edge: can only compare with the past. Low = incomplete filing, not a market signal.
            win = [cnt[yms[j]] for j in range(i - 12, i)]
            base = statistics.median(win)
            ratio = cnt[ym] / base if base else None
            if ratio is not None and ratio < TAIL_RATIO:
                flag, reason = 1, "incomplete_filing"
            else:
                flag, reason = 0, None
        else:
            ratio, flag, reason = None, 0, None
        event = KNOWN_EVENTS.get(ym) if flag else None
        out.append((ym, cnt[ym], ratio, flag, reason, event))
    return out


def compute_seasonality(conn, min_type_sales: int = MIN_TYPE_SALES) -> list[tuple]:
    """Detrend against a centred 12-month rolling baseline + medians + per
    property type. Computed after removing distorted months."""
    bad = {r[0] for r in conn.execute(
        "SELECT ym FROM lr_month_quality WHERE is_distorted=1")}
    rows = conn.execute(
        "SELECT substr(date,1,7), price, property_type FROM lr_transactions"
        " WHERE category='A' AND date IS NOT NULL").fetchall()

    prices: dict[tuple[str, str], list[int]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    for ym, price, pt in rows:
        counts[ym] += 1
        prices[(ym, pt if pt in TYPES else "O")].append(price)
        prices[(ym, "ALL")].append(price)
    med = {k: statistics.median(v) for k, v in prices.items()}
    yms = sorted(counts)

    seas: dict[tuple[str, int], list[float]] = defaultdict(list)
    vol: dict[tuple[str, int], list[float]] = defaultdict(list)
    nsale: dict[tuple[str, int], int] = defaultdict(int)

    for pt in ("ALL",) + TYPES:
        for i, ym in enumerate(yms):
            if ym in bad or i < WINDOW or i + WINDOW >= len(yms):
                continue
            here = med.get((ym, pt))
            if here is None or len(prices[(ym, pt)]) < min_type_sales:
                continue
            month = int(ym[5:7])
            win = [med[(yms[j], pt)] for j in range(i - WINDOW, i + WINDOW + 1)
                   if j != i and yms[j] not in bad and (yms[j], pt) in med]
            wc = [counts[yms[j]] for j in range(i - WINDOW, i + WINDOW + 1)
                  if j != i and yms[j] not in bad]
            if len(win) < 10 or not wc:
                continue
            seas[(pt, month)].append(here / statistics.median(win))
            vol[(pt, month)].append(counts[ym] / (sum(wc) / len(wc)))
            nsale[(pt, month)] += len(prices[(ym, pt)])

    stamp = _now()
    out = []
    for (pt, month), vals in sorted(seas.items()):
        out.append((pt, month, round(statistics.median(vals), 4),
                    round(statistics.median(vol[(pt, month)]), 4),
                    len(vals), nsale[(pt, month)], stamp))
    return out


def refresh(db_path: Path | str = DB_PATH,
            min_type_sales: int = MIN_TYPE_SALES) -> dict:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        stamp = _now()
        q = compute_quality(conn)
        conn.execute("DELETE FROM lr_month_quality")
        conn.executemany(
            "INSERT INTO lr_month_quality (ym,n,ratio,is_distorted,reason,event,"
            "computed_at) VALUES (?,?,?,?,?,?,?)", [(*r, stamp) for r in q])
        conn.commit()
        conn.executescript(CLEAN_VIEW)
        s = compute_seasonality(conn, min_type_sales)
        conn.execute("DELETE FROM lr_seasonality")
        conn.executemany(
            "INSERT INTO lr_seasonality (property_type,month,price_index,volume_index,"
            "n_months,n_sales,computed_at) VALUES (?,?,?,?,?,?,?)", s)
        conn.commit()
        return {"months": len(q), "distorted": sum(1 for r in q if r[3]),
                "named": sum(1 for r in q if r[5]),
                "seasonality_rows": len(s), "computed_at": stamp}
    finally:
        conn.close()


def staleness_days(db_path: Path | str = DB_PATH) -> float:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT MAX(computed_at) FROM lr_seasonality").fetchone()
    except sqlite3.OperationalError:
        return float("inf")
    finally:
        conn.close()
    if not row or not row[0]:
        return float("inf")
    then = datetime.strptime(row[0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - then).total_seconds() / 86400


def is_stale(db_path: Path | str = DB_PATH) -> bool:
    return staleness_days(db_path) > MAX_AGE_DAYS


def exclusion_note(db_path: Path | str = DB_PATH) -> str:
    """Consumers must say what was removed — otherwise clean data turns back into an untraceable claim."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT ym, reason, ratio FROM lr_month_quality WHERE is_distorted=1"
            " ORDER BY ym").fetchall()
        total = conn.execute("SELECT COUNT(*) FROM lr_month_quality").fetchone()[0]
    except sqlite3.OperationalError:
        return ""
    finally:
        conn.close()
    if not rows:
        return ""
    listed = ", ".join(f"{ym}(×{r:.2f})" for ym, _, r in rows if r)
    return (f"Excluded {len(rows)} of {total} months whose completion volume "
            f"broke from neighbouring months — policy deadlines and shocks "
            f"distort both counts and prices: {listed}. Also excluded: "
            f"Land Registry category B (repossessions / transfers not at "
            f"market value) and trailing months whose filing is incomplete.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Refresh the controlled market baselines")
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--check", action="store_true", help="only report staleness")
    args = ap.parse_args(argv)
    if args.check:
        d = staleness_days(args.db)
        print(f"baseline age {d:.1f} days — {'STALE' if d > MAX_AGE_DAYS else 'fresh'}"
              f" (gate {MAX_AGE_DAYS} days)")
        return 1 if d > MAX_AGE_DAYS else 0
    t0 = time.time()
    res = refresh(args.db)
    print(f"refresh done in {time.time()-t0:.1f}s: {res}")
    print(exclusion_note(args.db))
    return 0


if __name__ == "__main__":
    sys.exit(main())
