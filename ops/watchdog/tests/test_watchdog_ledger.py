import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timezone, timedelta  # noqa: E402
import watchdog_ledger as L  # noqa: E402


def _now():
    return datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_record_and_count_window(tmp_path):
    conn = L.open_db(str(tmp_path / "ops.db"))
    now = _now()
    for i in range(3):
        L.record(conn, "launchd-job", "queue-status", "fix", "enforce", 0, True, f"try{i}",
                 ts=now - timedelta(minutes=i * 5))
    assert L.count_in_window(conn, "launchd-job", "queue-status", 30 * 60, now) == 3
    assert L.count_in_window(conn, "launchd-job", "queue-status", 8 * 60, now) == 2  # 0 and 5 min ago


def test_fixes_today_counts_only_fix_rows(tmp_path):
    conn = L.open_db(str(tmp_path / "ops.db"))
    now = _now()
    L.record(conn, "launchd-job", "queue-status", "fix", "enforce", 0, True, "x", ts=now)
    L.record(conn, "launchd-job", "queue-status", "detect", "observe", None, None, "x", ts=now)
    assert L.fixes_today(conn, "queue-status", now) == 1


def test_last_open_problem_returns_most_recent(tmp_path):
    conn = L.open_db(str(tmp_path / "ops.db"))
    now = _now()
    L.record(conn, "launchd-job", "queue-status", "fix", "enforce", 0, None, "x", ts=now)
    L.record(conn, "launchd-job", "queue-status", "recover", "enforce", 0, True, "y",
             ts=now + timedelta(minutes=2))
    assert L.last_open_problem(conn, "launchd-job", "queue-status")["action"] == "recover"
