import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timezone, timedelta  # noqa: E402
import watchdog_notify as N  # noqa: E402
import watchdog_ledger as L  # noqa: E402


def _now():
    return datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_fixed_message_contains_target_and_count():
    msg = N.fmt_fixed("queue-status", "startup crash (getpath)", 90, 2)
    assert "queue-status" in msg and "🔧" in msg and "#2 today" in msg


def test_escalate_message_has_hint_and_tail():
    msg = N.fmt_escalate("radar-matcher", "exit=78", "line1\nline2", "check wtl-pyrun.sh")
    assert "🔴" in msg and "radar-matcher" in msg and "check wtl-pyrun.sh" in msg and "line2" in msg


def test_coalesce_suppresses_second_tier1_within_hour(tmp_path):
    conn = L.open_db(str(tmp_path / "ops.db"))
    now = _now()
    assert N.should_send_tier1(conn, "queue-status", now) is True
    # Record a RECOVERY notification (that is when tier-1 is sent)
    L.record(conn, "launchd-job", "queue-status", "recover", "enforce", 0, True, "notified", ts=now)
    assert N.should_send_tier1(conn, "queue-status", now + timedelta(minutes=30)) is False
    assert N.should_send_tier1(conn, "queue-status", now + timedelta(minutes=61)) is True
