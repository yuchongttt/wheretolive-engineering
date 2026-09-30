"""sqlite health check for the in-host 5-minute monitor (2026-09-06).

The 2026-09-05 incident (listing-facts daemon pinned a WAL read snapshot for
34 h; checkpoint frozen at frame 779 while the WAL grew to 11.5 GB) produced a
signal every 5 minutes in logs/wal-checkpoint.log — `done=244768/779` — that
nobody read. These checks turn that signal into a Telegram alert:
  1. checkpoint stall: backfill not advancing while the WAL keeps growing
  2. WAL file large for several consecutive ticks (a weekly VACUUM pushes the
     whole DB through the WAL for a few minutes — that must NOT alert)
  3. a burst of "database is locked" failures in ops.db skill_runs
"""
import os
import pytest
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import monitor_checks as mc  # noqa: E402


def _wal_db(tmp_path: Path) -> Path:
    p = tmp_path / "e.db"
    c = sqlite3.connect(p)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA wal_autocheckpoint=0")          # nothing drains the WAL behind our back
    c.execute("CREATE TABLE t(x)")
    c.commit()
    c.close()
    return p


# ── 1. wal-index header ─────────────────────────────────────────────────────

def test_read_wal_index_reports_frames_not_yet_backfilled(tmp_path):
    p = _wal_db(tmp_path)
    c = sqlite3.connect(p)
    c.execute("PRAGMA wal_autocheckpoint=0")
    c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    for i in range(50):
        c.execute("INSERT INTO t VALUES (?)", (i,))
        c.commit()
    mx, nb = mc.read_wal_index(p)
    assert mx >= 50 and nb == 0                        # 50 commits appended, none backfilled
    c.execute("PRAGMA wal_checkpoint(PASSIVE)")
    mx2, nb2 = mc.read_wal_index(p)
    assert nb2 == mx2 == mx                            # fully drained


def test_read_wal_index_is_none_without_a_wal_index(tmp_path):
    p = tmp_path / "plain.db"
    sqlite3.connect(p).close()
    assert mc.read_wal_index(p) is None


# ── 2. who pins the WAL ─────────────────────────────────────────────────────

@pytest.mark.skipif(sys.platform != "darwin", reason="packs the Darwin struct flock layout")
def test_pinned_reader_pid_names_the_process_holding_a_read_snapshot(tmp_path):
    p = _wal_db(tmp_path)
    sqlite3.connect(p).execute("PRAGMA wal_checkpoint(TRUNCATE)")
    child = subprocess.Popen(
        [sys.executable, "-c",
         "import sqlite3,sys,time; c=sqlite3.connect(sys.argv[1]); c.execute('BEGIN'); "
         "c.execute('SELECT COUNT(*) FROM t').fetchone(); print('ready', flush=True); time.sleep(30)",
         str(p)],
        stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        assert mc.pinned_reader_pid(p) == child.pid
    finally:
        child.kill()
        child.wait()
    deadline = time.time() + 5
    while mc.pinned_reader_pid(p) is not None and time.time() < deadline:
        time.sleep(0.1)
    assert mc.pinned_reader_pid(p) is None            # lock died with the process


# ── 3. locked-failure burst ─────────────────────────────────────────────────

def test_count_locked_failures_counts_only_recent_database_is_locked_rows(tmp_path):
    ops = tmp_path / "ops.db"
    c = sqlite3.connect(ops)
    c.execute("CREATE TABLE skill_runs (skill_name TEXT, started_at TEXT, status TEXT, error_message TEXT)")
    rows = [
        ("a", "-10 minutes", "failed", "OperationalError: database is locked"),   # counts
        ("b", "-25 minutes", "failed", "OperationalError: database is locked"),   # counts
        ("c", "-2 hours",    "failed", "OperationalError: database is locked"),   # too old
        ("d", "-5 minutes",  "failed", "TimeoutError: vlm"),                      # other error
        ("e", "-5 minutes",  "success", None),                                    # not a failure
    ]
    for name, off, status, err in rows:
        c.execute("INSERT INTO skill_runs VALUES (?, strftime('%Y-%m-%dT%H:%M:%f', 'now', ?), ?, ?)",
                  (name, off, status, err))
    c.commit()
    c.close()
    assert mc.count_locked_failures(ops, minutes=30) == 2


# ── 4. stall detection across ticks (pure) ──────────────────────────────────

def test_stall_ticks_count_only_while_backfill_is_stuck_and_wal_grows():
    s = {}
    s = mc.sqlite_stall_step(s, mx_frame=1_000, n_backfill=100)          # baseline
    assert s["sqlite_stall_ticks"] == 0
    s = mc.sqlite_stall_step(s, mx_frame=1_200, n_backfill=100)          # idle DB, tiny growth
    assert s["sqlite_stall_ticks"] == 0
    s = mc.sqlite_stall_step(s, mx_frame=80_000, n_backfill=100)         # writes pile up, nothing drains
    assert s["sqlite_stall_ticks"] == 1
    s = mc.sqlite_stall_step(s, mx_frame=200_000, n_backfill=100)
    assert s["sqlite_stall_ticks"] == 2
    s = mc.sqlite_stall_step(s, mx_frame=200_050, n_backfill=200_000)    # checkpoint caught up
    assert s["sqlite_stall_ticks"] == 0


def test_a_wal_restart_does_not_look_like_a_stall():
    s = mc.sqlite_stall_step({}, mx_frame=2_800_000, n_backfill=2_800_000)
    s = mc.sqlite_stall_step(s, mx_frame=6, n_backfill=0)                 # WAL rewound after a full checkpoint
    assert s["sqlite_stall_ticks"] == 0


# ── 5. verdict (pure) ───────────────────────────────────────────────────────

def _state(stall=0, wal_big=0):
    return {"sqlite_stall_ticks": stall, "sqlite_wal_big_ticks": wal_big}

def test_verdict_is_quiet_below_every_threshold():
    assert mc.sqlite_health_reason(_state(stall=5, wal_big=2), locked=9) is None

def test_verdict_fires_on_a_half_hour_checkpoint_stall():
    assert "checkpoint" in mc.sqlite_health_reason(_state(stall=6), locked=0)

def test_verdict_fires_when_the_wal_stays_huge_but_not_for_a_passing_vacuum():
    assert mc.sqlite_health_reason(_state(wal_big=2), locked=0) is None      # VACUUM in flight
    assert "WAL" in mc.sqlite_health_reason(_state(wal_big=3), locked=0)

def test_verdict_fires_on_a_locked_burst():
    assert "locked" in mc.sqlite_health_reason(_state(), locked=10)


def test_wal_big_ticks_track_consecutive_large_files():
    s = mc.sqlite_wal_size_step({}, wal_bytes=5 * 2**30)
    s = mc.sqlite_wal_size_step(s, wal_bytes=5 * 2**30)
    assert s["sqlite_wal_big_ticks"] == 2
    s = mc.sqlite_wal_size_step(s, wal_bytes=10 * 2**20)
    assert s["sqlite_wal_big_ticks"] == 0
