#!/usr/bin/env python3
"""ops.db ledger for the self-healing watchdog. One row per detect/fix/escalate/
recover event. Powers budget math, coalescing, and recovery detection."""
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = str(Path(os.environ.get("WTL_ROOT", "/opt/wheretolive")) / "data" / "ops.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS watchdog_actions (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    target TEXT NOT NULL,
    action TEXT NOT NULL,               -- 'detect' | 'fix' | 'escalate' | 'recover'
    mode TEXT NOT NULL,                 -- 'observe' | 'enforce'
    exit_code INTEGER,
    verified_recovered INTEGER,         -- 1/0/NULL
    note TEXT
);
CREATE INDEX IF NOT EXISTS idx_wd_target_ts ON watchdog_actions(rule_id, target, ts);
"""


def open_db(path: str = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.executescript(_SCHEMA)
    return conn


def _iso(ts):
    return (ts or datetime.now(timezone.utc)).isoformat()


def record(conn, rule_id, target, action, mode, exit_code, verified_recovered, note, ts=None):
    vr = None if verified_recovered is None else (1 if verified_recovered else 0)
    cur = conn.execute(
        "INSERT INTO watchdog_actions(ts,rule_id,target,action,mode,exit_code,verified_recovered,note)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (_iso(ts), rule_id, target, action, mode, exit_code, vr, note))
    conn.commit()
    return cur.lastrowid


def count_in_window(conn, rule_id, target, window_s, now):
    cutoff = now.timestamp() - window_s
    rows = conn.execute(
        "SELECT ts FROM watchdog_actions WHERE rule_id=? AND target=? AND action='fix'",
        (rule_id, target)).fetchall()
    return sum(1 for (ts,) in rows if datetime.fromisoformat(ts).timestamp() >= cutoff)


def last_open_problem(conn, rule_id, target):
    row = conn.execute(
        "SELECT id,ts,action,verified_recovered FROM watchdog_actions "
        "WHERE rule_id=? AND target=? ORDER BY id DESC LIMIT 1",
        (rule_id, target)).fetchone()
    if not row:
        return None
    return {"id": row[0], "ts": row[1], "action": row[2], "verified_recovered": row[3]}


def fixes_today(conn, target, now):
    day = now.date().isoformat()
    rows = conn.execute(
        "SELECT ts FROM watchdog_actions WHERE target=? AND action='fix'", (target,)).fetchall()
    return sum(1 for (ts,) in rows if ts[:10] == day)
