#!/usr/bin/env python3
"""Pure message formatting + coalescing decisions for watchdog Telegram output.
Tiers: 🔧 fixed (FYI) / 🔴 escalate (needs you) / ✅ recovered-after-escalation.
Tier-4 (watchdog/mac down) is emitted by the gpu-box heartbeat checker."""
from datetime import datetime


def _mins(s):
    return f"{round(s / 60)} min" if s >= 90 else f"{int(s)}s"


def fmt_fixed(target, reason, downtime_s, count_today):
    return (f"🔧 Auto-fixed\n{target}: {reason} → recovery confirmed\n"
            f"Down ~{_mins(downtime_s)} · fix #{count_today} today · no action needed")


def fmt_escalate(target, reason, log_tail, hint):
    return (f"🔴 Needs you\n{target}: auto-fix failed repeatedly, retries stopped\n"
            f"Symptom: {reason}\nRecent log: {log_tail}\nSuggested: {hint}")


def fmt_recovered(target, downtime_s):
    return f"✅ {target} back to normal (down {_mins(downtime_s)} in total)"


def should_send_tier1(conn, target, now, coalesce_s=3600):
    # Coalesce on prior RECOVERY notifications (tier-1 is sent at recovery), NOT
    # on 'fix' rows — the fix that triggers a recovery is always within the
    # window, so keying on 'fix' would suppress even the first message.
    rows = conn.execute(
        "SELECT ts FROM watchdog_actions WHERE target=? AND action='recover'", (target,)).fetchall()
    cutoff = now.timestamp() - coalesce_s
    return not any(datetime.fromisoformat(ts).timestamp() >= cutoff for (ts,) in rows)
