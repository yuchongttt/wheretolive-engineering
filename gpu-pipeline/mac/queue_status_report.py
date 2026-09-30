#!/usr/bin/env python3
"""Hourly Telegram report: per-queue throughput + backlog + failures.

Replaces the old outcode-expansion progress report. Pulls the SAME numbers the
admin "Queues" tab shows (GET /api/admin/queues?list=1 — single source of
truth) and sends a compact per-queue line every hour:

    <queue>: 1h ✓N · backlog M [· ✗F new failures]

- 1h processed = done_last_hour (rolling 1h window from the probe).
- backlog = current backlog the queue's worker still has to do.
- failures = NEW failures since the last report (delta of failed_lifetime,
  snapshotted in data/queue_status_snapshot.json) — answers "did anything fail?"
  without the noise of lifetime cumulative counts.

Run via launchd every 3600s.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "queue_status_snapshot.json"
API = "http://localhost:3000/api/admin/queues?list=1"
# VLM planned-maintenance window, shared with the pipeline monitor (see
# vlm-maintenance.sh): inside the window, new floorplan-vlm-queue failures are
# the expected product of a deliberate shutdown, so no alert.
VLM_MAINT_FILE = ROOT / "data" / "vlm_maintenance_until"


def vlm_in_maintenance() -> bool:
    try:
        return time.time() < float(VLM_MAINT_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return False

try:
    from wtl_tg import send_telegram  # noqa: E402 — the only home of the helper-bot credentials
except ImportError:  # public repo: the Telegram helper is not included
    def send_telegram(text, parse_mode=None):  # noqa: ARG001
        print(text)

# Display order + friendly labels. Grouped: enrich chain, then the
# floorplan/postcode mac queues, then GPU-box image queues.
# (Public repo: two listing-refresh queues from the private version are omitted.)
QUEUES = [
    ("enrich-lr-queue",        "LR match"),
    ("enrich-epc-queue",       "EPC match"),
    ("floorplan-vlm-queue",    "Floorplan VLM"),
    ("postcode-eval-queue",    "Postcode eval"),
    ("image-download-queue",   "Image download"),
    ("image-siglip-queue",     "SigLIP vectors"),
]


def admin_key() -> str:
    for env_name in (".env.local", ".env", ".env.production"):
        p = ROOT / "web" / env_name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.match(r"\s*ADMIN_KEY\s*=\s*[\"']?([^\"'\s]+)", line)
            if m:
                return m.group(1)
    return os.environ.get("ADMIN_KEY", "")


def fetch_queues() -> dict:
    # The Next.js server is briefly unreachable (~1-5s) during a deploy kickstart. This job
    # fires at :30 and used to crash + alert ("Connection refused") whenever a deploy
    # landed near the half hour. Retry over a kickstart window before giving up so a
    # transient restart doesn't page; only a genuine sustained outage alerts.
    # Catch OSError (covers URLError/ConnectionRefused at connect time AND
    # ConnectionResetError/TimeoutError raised mid-read when the server drops the
    # connection during a restart) plus http.client.HTTPException (RemoteDisconnected
    # / IncompleteRead). A bare ConnectionResetError is NOT a URLError, so the old
    # `except urllib.error.URLError` let read-phase resets escape the retry and page.
    req = urllib.request.Request(API, headers={"x-admin-key": admin_key()})
    last: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode())
        except (OSError, http.client.HTTPException) as e:
            last = e
            if attempt < 3:
                time.sleep(6)  # ride over the kickstart window (~1-5s)
    raise last if last else RuntimeError("fetch_queues failed")


def fmt(n) -> str:
    if n is None:
        return "?"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}k"
    return str(int(n))


def load_snapshot() -> dict:
    if SNAPSHOT.exists():
        try:
            return json.loads(SNAPSHOT.read_text())
        except Exception:  # noqa: BLE001
            return {}
    return {}


def save_snapshot(failed_by_q: dict) -> None:
    SNAPSHOT.write_text(json.dumps(
        {"ts": datetime.now(timezone.utc).isoformat(), "failed": failed_by_q}, indent=2))


def build_message(data: dict, prev_failed: dict) -> tuple[str, dict]:
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    qmap = {q["name"]: q for q in data.get("queues", [])}
    lines = [f"📊 *Queue status* · {now}", "_Each line: processed last 1h · current backlog · new failures_", ""]
    failed_now: dict = {}
    total_1h = 0
    any_fail = False
    for name, label in QUEUES:
        q = qmap.get(name)
        if not q:
            continue
        if q.get("probe_error"):
            lines.append(f"• {label}: ⚠️ probe failed")
            continue
        h1 = q.get("done_last_hour")
        backlog = q.get("backlog")
        failed_life = q.get("failed_lifetime")
        if isinstance(h1, (int, float)):
            total_1h += h1
        seg = f"• {label}: ✓{fmt(h1)} · backlog {fmt(backlog)}"
        # New failures this period (delta of lifetime cumulative).
        if failed_life is not None:
            failed_now[name] = failed_life
            prev = prev_failed.get(name)
            if prev is not None:
                delta = failed_life - prev
                if delta > 0:
                    seg += f" · ✗{fmt(delta)} new failures"
                    any_fail = True
        # Stale GPU-box probe (queue_stats.json not refreshed).
        age = q.get("stats_age_seconds")
        if isinstance(age, (int, float)) and age > 3600:
            seg += f" · ⏳data {int(age / 3600)}h stale"
        lines.append(seg)
    lines.append("")
    lines.append(f"Total processed last 1h ~{fmt(total_1h)}" + ("" if any_fail else " · no new failures this period ✅"))
    return "\n".join(lines), failed_now


def save_snapshot_full(state: dict) -> None:
    """Persist the full {failed, problem} state (superset of the old {failed})."""
    payload = {"ts": datetime.now(timezone.utc).isoformat()}
    payload.update(state or {})
    SNAPSHOT.write_text(json.dumps(payload, indent=2))


# A few sporadic failures self-heal (retry / picked up next round) and are not
# worth paging for. Only two situations count as a real problem:
#   1) a failure burst in one period (>= FAIL_BURST new in an hour, like a whole
#      batch dying)
#   2) new failures in FAIL_STREAK_RUNS consecutive periods (1h each) -- i.e.
#      it is not self-healing
# Before 2026-07-08 any delta>0 alerted, which caused "1 new failure -> back to
# normal" ping-pong spam.
FAIL_BURST = 50
FAIL_STREAK_RUNS = 3


def decide(prev_state: dict, data: dict) -> tuple[str | None, dict]:
    """Event-driven: return (message_or_None, new_state). Message is non-None
    ONLY on problem-onset (failure burst / sustained failure streak / GPU-box
    probe stale >1h) or problem-clear (a previously-flagged queue back to
    normal). Small transient failure deltas are silent -- they self-heal."""
    prev_fail = (prev_state or {}).get("failed", {})
    prev_problem = (prev_state or {}).get("problem", {})
    prev_streak = (prev_state or {}).get("streak", {})
    qmap = {q["name"]: q for q in data.get("queues", [])}
    label_of = dict(QUEUES)
    new_fail: dict = {}
    problem: dict = {}
    streak: dict = {}
    onset: list = []
    cleared: list = []
    for name, _label in QUEUES:
        q = qmap.get(name)
        if not q:
            continue
        fl = q.get("failed_lifetime")
        age = q.get("stats_age_seconds")
        stalled = isinstance(age, (int, float)) and age > 3600
        new_delta = (fl - prev_fail.get(name, fl)) if fl is not None else 0
        if fl is not None:
            new_fail[name] = fl
        run_streak = (prev_streak.get(name, 0) + 1) if new_delta > 0 else 0
        streak[name] = run_streak
        is_problem = stalled or new_delta >= FAIL_BURST or run_streak >= FAIL_STREAK_RUNS
        if name == "floorplan-vlm-queue" and vlm_in_maintenance():
            is_problem = False  # planned maintenance: new failures are expected
        problem[name] = is_problem
        if is_problem and not prev_problem.get(name):
            onset.append((name, new_delta, run_streak, stalled))
        if not is_problem and prev_problem.get(name):
            cleared.append(name)
    new_state = {"failed": new_fail, "problem": problem, "streak": streak}
    if not onset and not cleared:
        return None, new_state
    lines = []
    for name, delta, run_streak, stalled in onset:
        if delta >= FAIL_BURST:
            why = f"failure burst: {delta} within 1h"
        elif run_streak >= FAIL_STREAK_RUNS:
            why = f"new failures in {run_streak} consecutive periods (this period +{delta}), not self-healing"
        else:
            why = "data stale >1h"
        lines.append(f"🔴 {label_of.get(name, name)}: {why}")
    for name in cleared:
        lines.append(f"✅ {label_of.get(name, name)}: back to normal")
    return "\n".join(lines), new_state


def main() -> int:
    try:
        data = fetch_queues()
        msg, new_state = decide(load_snapshot(), data)
        save_snapshot_full(new_state)
        if msg:
            print(msg)
            send_telegram(msg if msg.startswith("[") else "[ALERT] " + msg,
                          parse_mode="Markdown")
            print("[ok] sent")
        else:
            print("[ok] healthy — silent (event-driven)")
        return 0
    except Exception as e:  # noqa: BLE001
        err = f"[ALERT]⚠️ Queue status report crashed: {type(e).__name__}: {e}"
        print(err)
        try:
            send_telegram(err)
        except Exception:  # noqa: BLE001
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
