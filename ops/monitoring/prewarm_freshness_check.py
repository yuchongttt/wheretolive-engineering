#!/usr/bin/env python3
"""Watchdog for xyz.wheretolive.prewarm-commute-hourly.

That job's failure mode is silence. If it stops, new listings stop entering
`postcode_hub_commute`, the candidate hit rate drifts down from 99.9%, and every
answer stays CORRECT — just slower, and more often shown as an estimate rather
than a verified door-to-door time. Nothing errors. An idle run deliberately
writes no skill_runs row (so quiet hours don't spam the Runs tab), which means
"no record for days" and "nothing to do for days" look identical from outside.

So the signal is not "did it run" but "is there work sitting undone". Pending
work with no successful run behind it means the queue is not draining. Pending
zero stays silent however long the gap — that is the legitimately quiet case,
and it is the common one now that the five in-use hubs are fully warm.

Silent when healthy; one Telegram line when not. Same shape as
backup_freshness_check.py.
"""
import os
import shlex
import subprocess
import sqlite3
import sys
import time
from pathlib import Path

# Application checkout root (holds data/, scripts/, chat-skills/, .venv/). The
# pending-work count below uses the app's commute-prewarm module
# (scripts/radar/prewarm_hub_commute.py), which is not part of this repo; only
# the pure `check()` verdict is exercised by the tests here.
ROOT = Path(os.environ.get("WTL_ROOT", "/opt/wheretolive"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "chat-skills"))

# Tolerate a whole day of quiet plus one missed run before calling it broken —
# the job runs hourly, so anything under this is a transient.
MAX_AGE_H = 48.0
# A postcode that failed once is re-queued until a second failure retires it, so
# a small residue is normal. ~385 new postcode x hub pairs arrive per day, so a
# genuinely stopped job clears this floor within hours.
MIN_PENDING = 50


def check(pending: int, last_success: float | None, now: float,
          max_age_h: float = MAX_AGE_H, min_pending: int = MIN_PENDING):
    """One alert line, or None when healthy."""
    if pending < min_pending:
        return None
    if last_success is None:
        return (f"prewarm watchdog: {pending:,} postcodes pending and prewarm-hub-commute has "
                f"NEVER recorded a successful run")
    age_h = (now - last_success) / 3600.0
    if age_h > max_age_h:
        return (f"prewarm watchdog: {pending:,} postcodes pending but the last "
                f"successful run was {age_h:.0f}h ago (>{max_age_h:.0f}h) — the "
                f"commute cache is falling behind new listings")
    return None


def _gather():
    from scripts.radar import prewarm_hub_commute as pw
    ev = sqlite3.connect(f"file:{ROOT}/data/evaluations.db?mode=ro", uri=True)
    hubs = pw.demanded_hubs(ev)
    active = pw.active_postcodes(ev)
    pending = sum(len(pw.band_postcodes(ev, h, pw.DEFAULT_MAX_MINUTES, active=active))
                  for h in hubs)
    ops = sqlite3.connect(f"file:{ROOT}/data/ops.db?mode=ro", uri=True)
    row = ops.execute(
        "SELECT MAX(strftime('%s', finished_at)) FROM skill_runs "
        "WHERE skill_name='prewarm-hub-commute' AND status='success'").fetchone()
    return pending, (float(row[0]) if row and row[0] else None)


def main() -> int:
    pending, last_success = _gather()
    msg = check(pending, last_success, time.time())
    if msg:
        notify = os.environ.get(
            "WTL_NOTIFY_CMD",
            f"{ROOT}/.venv/bin/python3 {HERE}/notify_telegram.py")
        try:
            subprocess.run(shlex.split(notify) + [msg], timeout=30)
        except (subprocess.TimeoutExpired, OSError) as e:
            print(f"prewarm watchdog: notify failed ({e!r})", file=sys.stderr)
        print(msg, file=sys.stderr)
        return 1
    print(f"prewarm watchdog: healthy (pending={pending:,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
