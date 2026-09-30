#!/usr/bin/env python3
"""Global daily LLM (Claude) cost alert (alert-only, NO cutoff).

Promotion safety net. Every LLM-backed route (chat / value-property /
ai-insight / radars-compile) logs its per-call cost to the `claude_usage`
table (lib/claude-usage.ts → logClaudeUsage). Per-user/IP daily quotas bound
any single caller, but there is no GLOBAL ceiling across all users — under a
promotion traffic spike total spend could run well past expectation before
anyone notices.

This job sums today's (UTC) Claude spend and pings Telegram when it crosses a
threshold. It does NOT stop anything (alert-only by design); flip the
single-route feature flags by hand if an alert fires and you want to pause.

To avoid spamming, it alerts once per integer multiple of the threshold per UTC
day: first crossing of $T, then $2T, $3T, … State in
data/claude_cost_alert_state.json.

Run via launchd hourly. Threshold via env CLAUDE_DAILY_COST_ALERT_USD
(default 50).
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# Application checkout root (holds data/, logs/, venv/).
ROOT = Path(os.environ.get("WTL_ROOT", "/opt/wheretolive"))
DB = ROOT / "data" / "evaluations.db"
STATE = ROOT / "data" / "claude_cost_alert_state.json"

from wtl_tg import send_telegram  # noqa: F401 — single home for creds/sending; re-exported for old importers

THRESHOLD = float(os.environ.get("CLAUDE_DAILY_COST_ALERT_USD", "50"))


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def fetch_today() -> tuple[float, int, list[tuple[str, int, float]]]:
    """Return (total_cost, total_calls, [(route, calls, cost), ...]) for today UTC."""
    # Read-only open: this is a monitor, it must never write evaluations.db.
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            """
            SELECT route, COUNT(*) AS calls, COALESCE(SUM(total_cost_usd), 0) AS cost
            FROM claude_usage
            WHERE ts >= date('now')
            GROUP BY route
            ORDER BY cost DESC
            """
        ).fetchall()
    finally:
        conn.close()
    by_route = [(r[0], int(r[1]), float(r[2])) for r in rows]
    total_cost = sum(c for _, _, c in by_route)
    total_calls = sum(n for _, n, _ in by_route)
    return total_cost, total_calls, by_route


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {}


def save_state(state: dict) -> None:
    try:
        STATE.write_text(json.dumps(state))
    except Exception:
        pass


def main() -> int:
    total_cost, total_calls, by_route = fetch_today()
    bucket = int(total_cost // THRESHOLD)  # 0 below threshold, 1 after $T, etc.

    day = today_utc()
    state = load_state()
    last_bucket = state.get("bucket", 0) if state.get("date") == day else 0

    if bucket > last_bucket:
        lines = [
            f"[ALERT]🚨 *Claude daily spend alert*",
            f"Today (UTC {day}) total *${total_cost:.2f}* / threshold ${THRESHOLD:.0f} · {total_calls} calls",
            "",
        ]
        for route, calls, cost in by_route:
            lines.append(f"· `{route}`: ${cost:.2f} ({calls})")
        lines.append("")
        lines.append("Alert only, nothing was stopped. To stem spend, turn off the per-route feature flags by hand.")
        try:
            send_telegram("\n".join(lines), parse_mode="Markdown")
        except Exception as e:
            print(f"[claude-cost-alert] telegram failed: {e}")
        save_state({"date": day, "bucket": bucket})
        print(f"[claude-cost-alert] ALERTED bucket={bucket} total=${total_cost:.2f}")
    else:
        # Keep state's date current so buckets reset cleanly at UTC midnight.
        save_state({"date": day, "bucket": max(last_bucket, bucket)})
        print(f"[claude-cost-alert] ok total=${total_cost:.2f} threshold=${THRESHOLD:.0f} bucket={bucket}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
