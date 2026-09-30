"""Counting rules for chat_telemetry: a 429 row refused by the daily cap is
neither an error nor a session.

Since 2026-09-24 the api/chat 429 branch writes one chat_telemetry row
(status='error', error_message starting with 'daily_limit_reached', session_id
NULL — the cap check runs before the body is parsed). Review pointed out that
chat_weekly_audit would count it in the error rate and invent a phantom NULL
session, and the daily review job would treat it as a real turn and wake
with nothing to trace. Both scripts share one rule set here.

Public copy: the private repo also pins DAILY_LIMIT_DENIED_PREFIX to the web
app's constant by reading src/lib/chat-daily-limit.ts; that test needs the web
app and is not included.
"""
import sqlite3

import chat_telemetry_counts as ctc


def _rows(*specs):
    """specs: (session_id, status, error_message) → sqlite3.Row list, so the
    helpers are exercised on the same row type the scripts hand them."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE t (session_id TEXT, status TEXT, error_message TEXT)")
    conn.executemany("INSERT INTO t VALUES (?,?,?)", specs)
    return conn.execute("SELECT * FROM t").fetchall()


def test_a_cap_denial_is_not_an_error():
    rows = _rows(("web-a", "ok", None),
                 (None, "error", "daily_limit_reached: 10/10"),
                 ("web-b", "error", "spawn failed"))
    assert ctc.count_errors(rows) == 1
    assert [ctc.is_cap_denial(r) for r in rows] == [False, True, False]


def test_a_null_session_is_not_a_session():
    rows = _rows(("web-a", "ok", None), (None, "error", "daily_limit_reached: 10/10"),
                 ("web-a", "ok", None))
    assert ctc.count_sessions(rows) == 1


def test_split_denials_keeps_order_and_counts():
    rows = _rows(("web-a", "ok", None), (None, "error", "daily_limit_reached: 3/3"),
                 ("web-b", "timeout", None))
    turns, denials = ctc.split_denials(rows)
    assert [r["session_id"] for r in turns] == ["web-a", "web-b"]
    assert len(denials) == 1


def test_rows_without_an_error_message_column_are_plain_turns():
    """Old DBs / test skeletons have no error_message column in chat_telemetry —
    must not crash; treated as plain turns."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE t (session_id TEXT, status TEXT)")
    conn.execute("INSERT INTO t VALUES ('web-a', 'error')")
    rows = conn.execute("SELECT * FROM t").fetchall()
    assert ctc.is_cap_denial(rows[0]) is False
    assert ctc.count_errors(rows) == 1
