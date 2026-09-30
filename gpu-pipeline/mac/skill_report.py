#!/usr/bin/env python3
"""Skill invocation reporter — writes to data/ops.db `skill_runs` table.

Two usage modes:

Library (preferred for in-process tracking):
    from skill_report import SkillRun, dispatch

    with SkillRun("nightly-maintenance", invoked_by="launchd") as run:
        run.summary["params"] = {...}
        # do work; on exit row is auto-updated with status + duration
        # raising an exception sets status='failed' and error_message

    # Fire-and-forget for downstream daemon work:
    dispatch("embed-images-dino-siglip", parent=run.invocation_id,
             summary={"queued_image_count": 187})

CLI (for shell-based skill wrappers / cross-process tracking):
    python3 scripts/skill_report.py dispatch \\
        --skill embed-images-dino-siglip \\
        --parent <invocation-id> \\
        --summary '{"queued_image_count": 187}' \\
        --invoked-by nightly-maintenance

Schema auto-created on first import. ops.db is OPERATIONAL metadata —
safe to drop without touching evaluations.db.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
OPS_DB_PATH = REPO_ROOT / "data" / "ops.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS skill_runs (
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
CREATE INDEX IF NOT EXISTS idx_skill_runs_name_time ON skill_runs (skill_name, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_skill_runs_time ON skill_runs (started_at DESC);
CREATE INDEX IF NOT EXISTS idx_skill_runs_invocation ON skill_runs (invocation_id);
CREATE INDEX IF NOT EXISTS idx_skill_runs_parent ON skill_runs (parent_invocation_id);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _open() -> sqlite3.Connection:
    OPS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(OPS_DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


class SkillRun:
    """Context manager that tracks one skill invocation.

    On enter: inserts a row with status='running' and remembers the row id.
    On exit: updates with finished_at + duration + status (success/failed)
    and serialises whatever you set on ``self.summary`` as ``summary_json``.

    The row's ``invocation_id`` is a UUID — children can use it as
    ``parent_invocation_id`` to record workflow → atom call chains.
    """

    def __init__(
        self,
        skill_name: str,
        invoked_by: str = "manual",
        parent: Optional[str] = None,
    ):
        self.skill_name = skill_name
        self.invoked_by = invoked_by
        self.parent = parent
        self.invocation_id = str(uuid.uuid4())
        self.summary: dict = {}
        self._row_id: Optional[int] = None
        self._start: Optional[datetime] = None

    def __enter__(self) -> "SkillRun":
        self._start = datetime.now(timezone.utc)
        conn = _open()
        try:
            cur = conn.execute(
                "INSERT INTO skill_runs "
                "(skill_name, invocation_id, parent_invocation_id, started_at, status, invoked_by) "
                "VALUES (?, ?, ?, ?, 'running', ?)",
                (
                    self.skill_name,
                    self.invocation_id,
                    self.parent,
                    self._start.isoformat(),
                    self.invoked_by,
                ),
            )
            self._row_id = cur.lastrowid
            conn.commit()
        finally:
            conn.close()
        return self

    def flush_summary(self) -> None:
        """Write the current summary to the row while the run is still going,
        so monitors can tell what KIND of run is in progress (a long nightly
        job marks a catch-up this way). __exit__ overwrites it with the final
        summary."""
        if self._row_id is None:
            return
        conn = _open()
        try:
            conn.execute(
                "UPDATE skill_runs SET summary_json=? WHERE id=?",
                (json.dumps(self.summary, ensure_ascii=False, default=str), self._row_id),
            )
            conn.commit()
        finally:
            conn.close()

    def __exit__(self, exc_type, exc_value, tb):
        finished = datetime.now(timezone.utc)
        duration_s = (finished - self._start).total_seconds() if self._start else 0.0
        status = "failed" if exc_type else "success"
        error_message = f"{exc_type.__name__}: {exc_value}" if exc_type else None
        conn = _open()
        try:
            conn.execute(
                "UPDATE skill_runs SET finished_at=?, status=?, summary_json=?, duration_s=?, error_message=? "
                "WHERE id=?",
                (
                    finished.isoformat(),
                    status,
                    json.dumps(self.summary, ensure_ascii=False, default=str),
                    duration_s,
                    error_message,
                    self._row_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return False  # re-raise exceptions


def dispatch(
    skill_name: str,
    summary: Optional[dict] = None,
    parent: Optional[str] = None,
    invoked_by: str = "manual",
) -> str:
    """Record a fire-and-forget invocation.

    Use when a workflow triggers an async / daemon-backed skill and
    doesn't wait for completion. Status is recorded as 'dispatched',
    finished_at equals started_at, duration_s=0. Downstream daemon
    progress is tracked by the daemon's own status tables, NOT by
    updating this row.

    Returns the invocation_id (so the caller can log it / pass to children).
    """
    invocation_id = str(uuid.uuid4())
    now = _now_iso()
    conn = _open()
    try:
        conn.execute(
            "INSERT INTO skill_runs "
            "(skill_name, invocation_id, parent_invocation_id, started_at, finished_at, "
            " status, summary_json, duration_s, invoked_by) "
            "VALUES (?, ?, ?, ?, ?, 'dispatched', ?, 0, ?)",
            (
                skill_name,
                invocation_id,
                parent,
                now,
                now,
                json.dumps(summary or {}, ensure_ascii=False, default=str),
                invoked_by,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return invocation_id


# ============ CLI ============

def main() -> int:
    p = argparse.ArgumentParser(
        description="Skill invocation reporter — read/write data/ops.db skill_runs."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dispatch", help="Record a fire-and-forget invocation.")
    d.add_argument("--skill", required=True, help="Skill name (e.g. embed-images-dino-siglip)")
    d.add_argument("--parent", default=None, help="Parent invocation_id (UUID)")
    d.add_argument("--summary", default="{}", help="JSON summary dict")
    d.add_argument("--invoked-by", default="manual",
                   help="Source of invocation (e.g. launchd, agent, nightly-maintenance)")

    args = p.parse_args()
    if args.cmd == "dispatch":
        try:
            summary = json.loads(args.summary)
        except json.JSONDecodeError as e:
            print(f"ERROR: --summary must be valid JSON: {e}", file=sys.stderr)
            return 2
        inv_id = dispatch(args.skill, summary=summary, parent=args.parent, invoked_by=args.invoked_by)
        print(inv_id)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
