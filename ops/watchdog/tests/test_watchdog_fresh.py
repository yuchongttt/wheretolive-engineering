"""A partitioned catch-up run (after an outage) of a ledger-tracked hourly job can
legitimately run for up to ~2 h. The watchdog (kickstart -k when not fresh
within 70 min and last exit != 0 — which is exactly the state after an outage)
must recognise it, or it kills the catch-up every time it runs. Ordinary runs
keep their original thresholds.

Ported from the private repo's catch-up monitor tests (watchdog_fresh cases
only); the job label and skill name are neutral stand-ins.
"""
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import watchdog_fresh  # noqa: E402
import watchdog_rules  # noqa: E402

LABEL, SKILL = "hourly-ingest", "ingest"

# skill_runs DDL as in the production ops.db (2026-09-25).
OPS_DDL = """
CREATE TABLE skill_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_name           TEXT NOT NULL,
    invocation_id        TEXT NOT NULL,
    parent_invocation_id TEXT,
    started_at           TEXT NOT NULL,
    finished_at          TEXT,
    status               TEXT NOT NULL,
    summary_json         TEXT,
    duration_s           REAL,
    error_message        TEXT,
    invoked_by           TEXT
);
"""


@pytest.fixture()
def ops_db(tmp_path, monkeypatch):
    path = tmp_path / "ops.db"
    c = sqlite3.connect(path)
    c.executescript(OPS_DDL)
    c.commit()
    c.close()
    monkeypatch.setattr(watchdog_fresh, "OPS_DB", path)
    monkeypatch.setitem(watchdog_rules.LEDGER_FRESH_JOBS, LABEL, SKILL)
    monkeypatch.setitem(watchdog_rules.JOB_FRESH_MINUTES, LABEL, 70)
    return path


def _run(path, minutes_ago, status, summary, finished_minutes_ago=None):
    now = datetime.now(timezone.utc)
    started = now - timedelta(minutes=minutes_ago)
    finished = (now - timedelta(minutes=finished_minutes_ago)).isoformat() \
        if finished_minutes_ago is not None else None
    c = sqlite3.connect(path)
    c.execute("INSERT INTO skill_runs (skill_name, invocation_id, started_at, finished_at, status, summary_json, invoked_by) "
              "VALUES (?, 'x', ?, ?, ?, ?, 'launchd')",
              (SKILL, started.isoformat(), finished, status, json.dumps(summary) if summary is not None else None))
    c.commit()
    c.close()


CATCHUP = {"mode": "catchup", "catchup_days": 7}


def test_catchup_running_90min_is_healthy(ops_db):
    _run(ops_db, 90, "running", CATCHUP)
    assert watchdog_fresh.job_fresh(LABEL) is True


def test_catchup_running_200min_is_stuck(ops_db):
    _run(ops_db, 200, "running", CATCHUP)
    assert watchdog_fresh.job_fresh(LABEL) is False


def test_finished_catchup_80min_ago_is_stale(ops_db):
    # once a catch-up has finished, the ordinary "missed hourly run" rule applies
    _run(ops_db, 80, "success", dict(CATCHUP, touched=500))
    assert watchdog_fresh.job_fresh(LABEL) is False


def test_plain_success_recent_is_fresh(ops_db):
    _run(ops_db, 10, "success", {"mode": "flat", "touched": 800})
    assert watchdog_fresh.job_fresh(LABEL) is True
