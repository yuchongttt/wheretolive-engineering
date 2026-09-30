"""One caliber for counting chat_telemetry rows in the audit scripts.

Since 2026-09-24 api/chat's daily-cap branch writes a telemetry row for every
refused request (status='error', error_message 'daily_limit_reached: N/N',
session_id NULL because the cap check runs before the body is parsed). That
row is a *signal* — the cap ended a session — not a failure to diagnose:
it has no session to replay and nothing broke. Counted naively it inflates
the weekly error rate, conjures a phantom NULL session, and wakes the daily
review job for a turn it can never trace. chat_weekly_audit.py and the daily
review job (not in this extract) both read the rule from here.

DAILY_LIMIT_DENIED_PREFIX mirrors DAILY_LIMIT_DENIED in the web app's
src/lib/chat-daily-limit.ts — in the full repo a test reads that file and pins
the two together (not included here, as the web app is not part of this module).
"""
from __future__ import annotations

DAILY_LIMIT_DENIED_PREFIX = "daily_limit_reached"


def _field(row, name: str):
    """sqlite3.Row / dict tolerant getter that treats a missing column as
    None — older DBs and test skeletons predate error_message."""
    try:
        keys = row.keys()
    except AttributeError:
        return None
    return row[name] if name in keys else None


def is_cap_denial(row) -> bool:
    return str(_field(row, "error_message") or "").startswith(DAILY_LIMIT_DENIED_PREFIX)


def split_denials(rows):
    """(turns, denials) — order preserved."""
    turns, denials = [], []
    for r in rows:
        (denials if is_cap_denial(r) else turns).append(r)
    return turns, denials


def count_errors(rows) -> int:
    """Non-ok turns, cap denials excluded."""
    return sum(1 for r in rows
               if (_field(r, "status") or "ok") != "ok" and not is_cap_denial(r))


def count_sessions(rows) -> int:
    """Distinct real sessions — a NULL session_id is not a session."""
    return len({_field(r, "session_id") for r in rows if _field(r, "session_id")})
