#!/usr/bin/env python3
"""Retention sweep for ops.db skill_runs.

skill_runs grows ~1M rows/month: per-listing enrich atoms (enrich-listings-
with-epc/lr, enrich-ex-council, enrich-planning-flags, resolve-floor-area, …)
log one row per listing per hourly ingest run. That volume bloated ops.db to
~750MB and is the raw material for slow admin queries.

Policy:
  - HIGH-FREQUENCY skills (all-time run count > HIGH_FREQ_THRESHOLD): keep the
    trailing RETAIN_DAYS (45) days, delete older. Sizing: ~26.5k rows/day as of
    2026-07 -> steady state ~1.2M rows / ~830MB incl. indexes; 60d would be 1.1GB.
  - Everything else (monthly/weekly jobs, workflows, one-offs) is kept forever
    — structurally untouched by the DELETE predicate.

Guards (mirroring nightly_db_retention.py's spirit — the failure mode here is
a typo'd cutoff nuking recent history the admin panel needs):
  - refuses RETAIN_DAYS < MIN_RETAIN_DAYS (45): the admin Skills panel builds
    its workflow composition from a 30-day window; 45 keeps a wide margin.
  - the DELETE predicate itself only matches high-frequency skills AND rows
    older than the cutoff — low-frequency jobs cannot be touched by design.
  - post-delete assertion: total rows inside the window must be unchanged.

VACUUM: --vacuum reclaims file space, but only when freelist > VACUUM_MIN_FRAC
of the file (a full VACUUM rewrites 750MB and briefly locks writers — the
20/60s dispatchers ride it out via busy_timeout, so run it in the quiet
weekly window, not ad hoc at midday).

Usage:
  python3 ops_retention.py --dry-run
  python3 ops_retention.py            # delete pass only
  python3 ops_retention.py --vacuum   # weekly launchd form

(SkillRun — the run ledger that writes ops.db skill_runs — comes from the main
application and is not part of this repo.)
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

# Application checkout root (holds data/, scripts/).
REPO = Path(os.environ.get("WTL_ROOT", "/opt/wheretolive"))
DB = REPO / "data" / "ops.db"

sys.path.insert(0, str(REPO / "scripts"))
from skill_report import SkillRun  # noqa: E402

RETAIN_DAYS = 45
MIN_RETAIN_DAYS = 45          # admin composition window (30d) + margin
HIGH_FREQ_THRESHOLD = 5_000   # all-time runs above this = per-listing atom noise
VACUUM_MIN_FRAC = 0.20        # VACUUM only if >20% of pages are free


def sweep(conn: sqlite3.Connection, retain_days: int, dry_run: bool) -> dict:
    if retain_days < MIN_RETAIN_DAYS:
        raise SystemExit(
            f"REFUSING: retain_days={retain_days} < {MIN_RETAIN_DAYS} — the admin "
            "Skills panel reads a 30-day composition window from this table."
        )
    cutoff = f"-{retain_days} days"

    hf = [
        (name, n)
        for name, n in conn.execute(
            "SELECT skill_name, COUNT(*) FROM skill_runs GROUP BY skill_name"
        )
        if n > HIGH_FREQ_THRESHOLD
    ]
    hf_names = [name for name, _ in hf]
    print(f"[ops retention] high-frequency skills (> {HIGH_FREQ_THRESHOLD} runs):")
    for name, n in sorted(hf, key=lambda x: -x[1]):
        print(f"  {name}: {n}")
    if not hf_names:
        print("  none — nothing to do")
        return {"deleted": 0}

    ph = ",".join("?" * len(hf_names))
    in_window_before = conn.execute(
        "SELECT COUNT(*) FROM skill_runs WHERE started_at >= datetime('now', ?)",
        (cutoff,),
    ).fetchone()[0]
    old = conn.execute(
        f"SELECT COUNT(*) FROM skill_runs WHERE skill_name IN ({ph}) "
        "AND started_at < datetime('now', ?)",
        (*hf_names, cutoff),
    ).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM skill_runs").fetchone()[0]
    print(f"\n  would delete {old}/{total} rows older than {retain_days}d "
          f"(keeps every low-frequency job forever)")
    if dry_run or old == 0:
        return {"deleted": 0, "would_delete": old, "total": total, "dry_run": dry_run}

    conn.execute(
        f"DELETE FROM skill_runs WHERE skill_name IN ({ph}) "
        "AND started_at < datetime('now', ?)",
        (*hf_names, cutoff),
    )
    conn.commit()

    in_window_after = conn.execute(
        "SELECT COUNT(*) FROM skill_runs WHERE started_at >= datetime('now', ?)",
        (cutoff,),
    ).fetchone()[0]
    # New rows may land between the two counts (dispatchers write constantly),
    # so "unchanged or grew" is the invariant — shrinkage means the predicate bit
    # into the window and that is a stop-the-world bug.
    if in_window_after < in_window_before:
        raise RuntimeError(
            f"post-delete assertion failed: in-window rows {in_window_before} -> "
            f"{in_window_after}. Predicate touched recent history!"
        )
    remaining = conn.execute("SELECT COUNT(*) FROM skill_runs").fetchone()[0]
    print(f"  deleted {old}, remaining {remaining}")
    return {"deleted": old, "remaining": remaining, "total_before": total}


def maybe_vacuum(conn: sqlite3.Connection, force: bool = False) -> dict:
    page_count = conn.execute("PRAGMA page_count").fetchone()[0]
    freelist = conn.execute("PRAGMA freelist_count").fetchone()[0]
    frac = freelist / page_count if page_count else 0.0
    print(f"\n[vacuum] freelist {freelist}/{page_count} pages ({frac*100:.0f}%)")
    if not force and frac < VACUUM_MIN_FRAC:
        print(f"  below {VACUUM_MIN_FRAC*100:.0f}% threshold — skipping")
        return {"vacuumed": False, "free_frac": round(frac, 3)}
    t0 = time.time()
    conn.execute("VACUUM")
    dt = time.time() - t0
    size_mb = DB.stat().st_size / 1e6
    print(f"  VACUUM done in {dt:.1f}s, file now {size_mb:.0f}MB")
    return {"vacuumed": True, "vacuum_s": round(dt, 1), "file_mb": round(size_mb)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--vacuum", action="store_true", help="VACUUM if freelist > threshold")
    ap.add_argument("--retain-days", type=int, default=RETAIN_DAYS)
    args = ap.parse_args()

    conn = sqlite3.connect(DB, timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")

    if args.dry_run:
        sweep(conn, args.retain_days, dry_run=True)
        if args.vacuum:
            print("\n(dry-run: skipping vacuum)")
        conn.close()
        return 0

    with SkillRun("ops-retention", invoked_by="launchd") as run:
        run.summary.update(sweep(conn, args.retain_days, dry_run=False))
        if args.vacuum:
            run.summary.update(maybe_vacuum(conn))
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
