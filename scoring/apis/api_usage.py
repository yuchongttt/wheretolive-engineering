#!/usr/bin/env python3
"""API usage tracking and monthly quota enforcement."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional


_CONFIG_CACHE: Optional[dict] = None
_QUOTA_EXCEEDED_APIS: set[str] = set()


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _get_db_path() -> Path:
    return _get_repo_root() / "data" / "evaluations.db"


def _load_config() -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    config_path = _get_repo_root() / "config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            _CONFIG_CACHE = json.load(f)
    except FileNotFoundError:
        _CONFIG_CACHE = {}
    except json.JSONDecodeError:
        _CONFIG_CACHE = {}
    return _CONFIG_CACHE


def _get_month_key(now: Optional[datetime] = None) -> str:
    dt = now or datetime.utcnow()
    return dt.strftime("%Y-%m")


def _get_monthly_limit(api_name: str) -> Optional[int]:
    config = _load_config()
    limits = config.get("api_limits", {}) or {}
    api_cfg = limits.get(api_name, {}) or {}
    limit = api_cfg.get("monthly_limit")
    if isinstance(limit, int) and limit > 0:
        return limit
    return None


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS api_usage_monthly (
            api_name TEXT NOT NULL,
            year_month TEXT NOT NULL,
            count INTEGER NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (api_name, year_month)
        )
        """
    )


def get_quota_exceeded_apis() -> set[str]:
    return set(_QUOTA_EXCEEDED_APIS)


def clear_quota_exceeded_apis() -> None:
    _QUOTA_EXCEEDED_APIS.clear()


def record_api_usage(api_name: str, units: int = 1) -> None:
    """Record API usage without quota enforcement. For free/unlimited APIs."""
    if units <= 0:
        return

    db_path = _get_db_path()
    now = datetime.utcnow().replace(microsecond=0).isoformat()
    year_month = _get_month_key()

    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode = WAL")
        _ensure_table(conn)

        row = conn.execute(
            "SELECT count FROM api_usage_monthly WHERE api_name = ? AND year_month = ?",
            (api_name, year_month)
        ).fetchone()

        if row:
            conn.execute(
                "UPDATE api_usage_monthly SET count = ?, updated_at = ? WHERE api_name = ? AND year_month = ?",
                (row[0] + units, now, api_name, year_month)
            )
        else:
            conn.execute(
                "INSERT INTO api_usage_monthly (api_name, year_month, count, updated_at) VALUES (?, ?, ?, ?)",
                (api_name, year_month, units, now)
            )
        conn.commit()
        conn.close()
    except Exception:
        pass  # A failure to record usage must not affect the main flow


def try_acquire_api_quota(api_name: str, units: int = 1) -> bool:
    """
    Atomically check monthly quota and increment usage.

    Returns True if quota is available (and increments usage), False otherwise.
    If no limit is configured, always returns True and does not write usage.
    """
    if units <= 0:
        return True

    limit = _get_monthly_limit(api_name)
    db_path = _get_db_path()
    now = datetime.utcnow().replace(microsecond=0).isoformat()
    year_month = _get_month_key()

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode = WAL")
        _ensure_table(conn)

        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT count FROM api_usage_monthly WHERE api_name = ? AND year_month = ?",
            (api_name, year_month)
        ).fetchone()
        current = row[0] if row else 0
        if limit and current + units > limit:
            conn.execute("ROLLBACK")
            _QUOTA_EXCEEDED_APIS.add(api_name)
            return False

        if row:
            conn.execute(
                "UPDATE api_usage_monthly SET count = ?, updated_at = ? WHERE api_name = ? AND year_month = ?",
                (current + units, now, api_name, year_month)
            )
        else:
            conn.execute(
                "INSERT INTO api_usage_monthly (api_name, year_month, count, updated_at) VALUES (?, ?, ?, ?)",
                (api_name, year_month, units, now)
            )

        conn.execute("COMMIT")
        return True
    finally:
        conn.close()
