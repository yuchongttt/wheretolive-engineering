#!/usr/bin/env python3
"""Per-job freshness (is there a recent healthy signal?) and log-tail helpers."""
import json
import os
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
import watchdog_rules as R

# Application checkout root (holds data/ and logs/).
ROOT = Path(os.environ.get("WTL_ROOT", "/opt/wheretolive"))
LOGS = Path(os.environ.get("WATCHDOG_LOG_DIR") or ROOT / "logs")
OPS_DB = ROOT / "data" / "ops.db"


def _is_live_catchup(status, summary_json, started) -> bool:
    """A catch-up run still inside its time budget counts as fresh."""
    if status != "running":
        return False
    try:
        summary = json.loads(summary_json) if summary_json else {}
    except ValueError:
        return False
    if not isinstance(summary, dict) or summary.get("mode") != "catchup":
        return False
    budget = timedelta(minutes=R.CATCHUP_FRESH_MINUTES)
    return started >= datetime.now(timezone.utc) - budget


def job_fresh(label: str) -> bool:
    mins = R.JOB_FRESH_MINUTES.get(label, 70)
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=mins)
    skill = R.LEDGER_FRESH_JOBS.get(label)
    if skill:
        try:
            conn = sqlite3.connect(str(OPS_DB))
            row = conn.execute(
                "SELECT started_at, status, summary_json FROM skill_runs "
                "WHERE skill_name=? ORDER BY started_at DESC LIMIT 1", (skill,)).fetchone()
            conn.close()
            if not row:
                return False
            started = datetime.fromisoformat(row[0])
            if started >= cutoff:
                return True
            return _is_live_catchup(row[1], row[2], started)
        except Exception:
            return True   # unknown -> assume fresh (fail-safe: don't kickstart on doubt)
    log = LOGS / f"{label}.log"
    try:
        mtime = datetime.fromtimestamp(log.stat().st_mtime, tz=timezone.utc)
        return mtime >= cutoff
    except FileNotFoundError:
        return True


def log_tail(label: str, n=3) -> str:
    for name in (f"{label}.log", f"{label}.err"):
        p = LOGS / name
        if p.exists():
            lines = p.read_text(errors="replace").splitlines()[-n:]
            if lines:
                return " / ".join(lines)
    return "(no log)"
