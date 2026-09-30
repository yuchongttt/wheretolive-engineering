#!/usr/bin/env python3
"""Shared health checks + Telegram sender for the in-host 5-minute monitor and
the self-healing watchdog. Extracted so both import ONE source of truth (no
drift). Stdlib-only — safe to import under system python.

(Public extract: the original module also carried data-pipeline-specific checks
that are not part of this repo; only the generic pieces are kept here.)"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

# Application checkout root (holds data/, logs/, venv/).
ROOT = Path(os.environ.get("WTL_ROOT", "/opt/wheretolive"))
OPS_DB = ROOT / "data" / "ops.db"

from wtl_tg import send_telegram

# Host to ping (the GPU box). Env override also used to test reachability failure.
HOST_TAILSCALE_IP = os.environ.get("WTL_MONITOR_HOST_IP", "gpu-box")


def tg(text: str) -> bool:
    """Best-effort alert send (single exit point: wtl_tg). Every message from this
    module is alert-level, so an [ALERT] prefix is added when none is present.
    No markdown — keep plain to avoid escape bugs."""
    if not text.startswith("["):
        text = "[ALERT] " + text
    try:
        send_telegram(text)
        print(f"[tg] {text[:80]}", flush=True)
        return True
    except Exception as e:
        print(f"[tg] send failed: {e}", file=sys.stderr, flush=True)
        return False


def check_host_reachable() -> bool:
    """ICMP ping, 1 packet, 3s timeout. Use full path — launchd PATH doesn't include /sbin."""
    try:
        r = subprocess.run(
            ["/sbin/ping", "-c", "1", "-W", "3000", HOST_TAILSCALE_IP],
            capture_output=True, timeout=5
        )
        return r.returncode == 0
    except Exception:
        return False


# ── sqlite health: WAL / checkpoint / stuck reader (2026-09-06) ─────────────
# The 2026-09-05 incident: a daemon pinned a WAL read snapshot for 34 h, the
# checkpoint froze at frame 779 while the WAL grew to 11.5 GB, and the only
# signal was `done=244768/779` in logs/wal-checkpoint.log every 5 minutes.
# These helpers read the same numbers straight from the wal-index (-shm) and
# turn them into a deduped alert. Stall is the primary signal; the WAL-size
# threshold is deliberately slow (3 ticks) because the weekly VACUUM pushes the
# whole DB through the WAL for a few minutes and must not page anyone.
import fcntl
import struct

EVAL_DB = ROOT / "data" / "evaluations.db"
SQLITE_STALL_TICKS = 6              # 6 × 5 min: backfill frozen for half an hour
SQLITE_STALL_MIN_FRAMES = 50_000    # ...while the WAL grew this much — an idle DB is not a stall
SQLITE_WAL_ALERT_BYTES = 4 * 2**30  # WAL file this big...
SQLITE_WAL_BIG_TICKS = 3            # ...for 15 consecutive minutes (a VACUUM drains in under 10)
SQLITE_LOCKED_ALERT = 10            # "database is locked" failures per 30 min


def _shm(db_path) -> Path:
    return Path(f"{db_path}-shm")


def read_wal_index(db_path) -> tuple[int, int] | None:
    """(mxFrame, nBackfill) from the wal-index header — None when there is no -shm.
    mxFrame is the last valid WAL frame; nBackfill how many are already in the DB file.
    nBackfill frozen while mxFrame climbs = a reader is pinning the WAL."""
    shm = _shm(db_path)
    if not shm.exists():
        return None
    b = shm.read_bytes()[:136]
    if len(b) < 136:
        return None
    return struct.unpack_from("<I", b, 16)[0], struct.unpack_from("<I", b, 96)[0]


def pinned_reader_pid(db_path) -> int | None:
    """PID holding any of the five WAL read locks (shm bytes 123..127), via
    F_GETLK — no lock is taken. macOS `lsof` cannot show byte-range locks, and
    a process cannot see its own locks this way, so run it from the monitor."""
    shm = _shm(db_path)
    if not shm.exists():
        return None
    fd = os.open(shm, os.O_RDWR)
    try:
        for off in range(123, 128):  # WAL_READ_LOCK(i) = 120 + 3 + i
            # struct flock on Darwin: l_start, l_len, l_pid, l_type, l_whence
            lk = struct.pack("qqihh", off, 1, 0, fcntl.F_WRLCK, 0)
            _, _, pid, typ, _ = struct.unpack("qqihh", fcntl.fcntl(fd, fcntl.F_GETLK, lk))
            if typ != fcntl.F_UNLCK:
                return pid
    finally:
        os.close(fd)
    return None


def count_locked_failures(ops_db=OPS_DB, minutes: int = 30) -> int:
    """skill_runs rows that failed with "database is locked" in the last N minutes."""
    try:
        conn = sqlite3.connect(f"file:{ops_db}?mode=ro", uri=True, timeout=5)
        n = conn.execute(
            "SELECT COUNT(*) FROM skill_runs WHERE status = 'failed' "
            "AND error_message LIKE '%database is locked%' "
            "AND started_at >= strftime('%Y-%m-%dT%H:%M:%f', 'now', ?)",
            (f"-{minutes} minutes",)).fetchone()[0]
        conn.close()
        return int(n)
    except Exception:
        return 0


def sqlite_stall_step(state: dict, mx_frame: int, n_backfill: int) -> dict:
    """Advance the stall counter by one tick. Counts a tick only when the WAL
    grew materially and nothing was backfilled; a WAL restart (mxFrame drops)
    or any backfill progress resets it; a quiet DB holds the count."""
    s = dict(state)
    last_mx, last_nb = s.get("sqlite_last_mxframe"), s.get("sqlite_last_backfill")
    ticks = s.get("sqlite_stall_ticks", 0)
    if last_mx is None or mx_frame < last_mx:
        ticks = 0
    elif n_backfill != last_nb:
        ticks = 0
    elif mx_frame - last_mx >= SQLITE_STALL_MIN_FRAMES:
        ticks += 1
    s.update({"sqlite_last_mxframe": mx_frame, "sqlite_last_backfill": n_backfill,
              "sqlite_stall_ticks": ticks})
    return s


def sqlite_wal_size_step(state: dict, wal_bytes: int) -> dict:
    s = dict(state)
    big = s.get("sqlite_wal_big_ticks", 0)
    s["sqlite_wal_big_ticks"] = big + 1 if wal_bytes >= SQLITE_WAL_ALERT_BYTES else 0
    s["sqlite_wal_bytes"] = wal_bytes
    return s


def sqlite_health_reason(state: dict, locked: int) -> str | None:
    """Alert text, or None while everything is under threshold."""
    stall = state.get("sqlite_stall_ticks", 0)
    big = state.get("sqlite_wal_big_ticks", 0)
    if stall >= SQLITE_STALL_TICKS:
        return f"checkpoint stalled for {stall * 5} min (backfill not advancing while the WAL keeps growing)"
    if big >= SQLITE_WAL_BIG_TICKS:
        gb = state.get("sqlite_wal_bytes", 0) / 2**30
        return (f"WAL file above {SQLITE_WAL_ALERT_BYTES / 2**30:.0f} GB for {big * 5} "
                f"consecutive minutes (now {gb:.1f} GB)")
    if locked >= SQLITE_LOCKED_ALERT:
        return f"{locked} 'database is locked' failures within 30 minutes"
    return None
