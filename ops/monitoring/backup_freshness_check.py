#!/usr/bin/env python3
"""Backup freshness watchdog. Silent when the last verified backup is recent;
alerts (Telegram) when the success marker is missing or older than the threshold."""
import os, sys, time, subprocess, shlex, calendar
from pathlib import Path

# Application checkout root (holds data/, logs/, venv/).
ROOT = Path(os.environ.get("WTL_ROOT", "/opt/wheretolive"))
HERE = Path(__file__).resolve().parent
DEFAULT_MARKER = ROOT / "logs" / "db-backup-last-success"


def check(now_epoch, marker_path, max_age_h):
    p = Path(marker_path)
    if not p.exists():
        return "backup watchdog: success marker MISSING — no verified backup on record"
    try:
        ts = p.read_text().strip()
        marker_epoch = calendar.timegm(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ"))
    except Exception as e:
        return f"backup watchdog: unreadable marker ({e!r})"
    age_h = (now_epoch - marker_epoch) / 3600.0
    if age_h > max_age_h:
        return f"backup watchdog: last backup is STALE — {age_h:.1f}h old (>{max_age_h}h)"
    return None


def main():
    marker = os.environ.get("WTL_MARKER", str(DEFAULT_MARKER))
    # 2026-07-27: 26→48h. Backup runs daily at 05:20, this watchdog daily at 09:10 — one
    # miss leaves the marker ~28h old (tolerated: transient failures don't alert), two
    # misses in a row ~52h (>48 → alert = a genuinely persistent problem). A single
    # failure is observed via stderr + the admin STATUS_JSON instead.
    max_age = float(os.environ.get("WTL_BACKUP_MAX_AGE_H", "48"))
    msg = check(time.time(), marker, max_age)
    if msg:
        notify = os.environ.get("WTL_NOTIFY_CMD",
                                f"{ROOT}/venv/bin/python3 {HERE}/notify_telegram.py")
        try:
            subprocess.run(shlex.split(notify) + [msg], timeout=30)
        except (subprocess.TimeoutExpired, OSError) as e:
            print(f"backup watchdog: notify failed ({e!r})", file=sys.stderr)
        print(msg, file=sys.stderr)
        return 1
    print("backup watchdog: fresh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
