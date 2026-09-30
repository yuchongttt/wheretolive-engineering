#!/usr/bin/env python3
"""Prune stale launchd stdio logs under ~/Library/Logs/wheretolive.

Since 2026-07-13 every launchd unit's Standard{Out,Error}Path lives in
~/Library/Logs (one directory per app on this host).
Retired or renamed units leave behind files nothing writes anymore; this sweep
deletes *.log / *.err not modified in RETAIN_DAYS. Actively-written files always
carry a fresh mtime and are structurally untouchable — this is a garbage
collector for dead files, NOT a rotator of live ones (launchd holds the fd on
live logs; truncating or moving those is a deploy concern, never retention's).

App-level logs inside the repo's logs/ dir are out of scope on purpose: mixed
one-off script outputs live there and mtime alone can't tell "retired unit's
log" from "rarely-run backfill record someone still wants".

Guards (mirroring ops_retention.py's spirit):
  - non-recursive: only files DIRECTLY under the log dir
  - suffix allowlist (.log / .err) — never touches state files parked there
  - refuses RETAIN_DAYS < MIN_RETAIN_DAYS (30)

Usage:
  python3 launchd_logs_retention.py --dry-run
  python3 launchd_logs_retention.py            # weekly launchd form (Sun 04:50)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

DIRS = [
    Path.home() / "Library" / "Logs" / "wheretolive",
]
RETAIN_DAYS = 90
MIN_RETAIN_DAYS = 30
SUFFIXES = {".log", ".err"}


def sweep(retain_days: int, dry_run: bool) -> int:
    if retain_days < MIN_RETAIN_DAYS:
        raise SystemExit(
            f"REFUSING: retain_days={retain_days} < {MIN_RETAIN_DAYS} — a typo'd "
            "cutoff must not eat logs the next incident post-mortem needs."
        )
    cutoff = time.time() - retain_days * 86400
    deleted = 0
    for d in DIRS:
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if not f.is_file() or f.suffix not in SUFFIXES:
                continue
            st = f.stat()
            if st.st_mtime >= cutoff:
                continue
            age_days = int((time.time() - st.st_mtime) / 86400)
            print(f"[logs retention] {'would delete' if dry_run else 'delete'} "
                  f"{f} ({st.st_size} bytes, idle {age_days}d)")
            if not dry_run:
                f.unlink()
            deleted += 1
    print(f"[logs retention] {'candidates' if dry_run else 'deleted'}: {deleted} "
          f"(retain={retain_days}d, dirs={len(DIRS)})")
    return deleted


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--retain-days", type=int, default=RETAIN_DAYS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sweep(args.retain_days, args.dry_run)


if __name__ == "__main__":
    main()
