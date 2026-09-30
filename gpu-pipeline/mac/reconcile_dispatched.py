#!/usr/bin/env python3
"""reconcile-dispatched — close out dispatched skill_runs by verifying
their downstream queue handled the work.

Fire-and-forget skills (embed-images-dino-siglip, etc.) write a row with
status='dispatched' and never update it. Without reconciliation, admin
Runs tab shows endless "dispatched" rows that look stuck.

This reconciler:
1. Finds dispatched skill_runs older than --min-age-minutes
2. For each, checks if the related queue is healthy (recent activity OR
   caught up) — if yes, mark as 'success' with verification note
3. If queue is stuck for hours, mark as 'failed' with the reason
4. Writes finished_at + summary_json so the row shows in completed list
   instead of pending dispatched

Designed to run on a cron schedule (every 10 min). Idempotent — only
touches rows still in dispatched state.

Verification rules per dispatched skill:
  embed-images-dino-siglip → check GPU-box queue stats:
    - If pending==0 AND recent embedding activity → success
    - If embeddings happened since dispatch start → success (downstream
      processed at least some, even if more queued)
    - If pending grows and no embedding activity → failed (daemon stuck)
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from skill_report import OPS_DB_PATH as OPS_DB  # noqa: E402

QUEUE_STATS_JSON = REPO_ROOT / "data" / "queue_stats.json"

# Map dispatched skill_name → list of queue_names that represent its
# downstream work. Used to look up health from queue_stats.json.
#
# List ONLY queues that currently have a probe actually reporting. Listing a
# retired queue makes every reconcile round fail: _queue_healthy can't find it
# -> "no probe data" -> the whole run is marked failed, even if every other
# queue is healthy. After DINO was retired on 2026-06-28 (embed daemon running
# DINO_ENABLED=0, probe no longer reporting image-dinov3-queue) this map was not
# updated, so embed-images-dino-siglip was falsely marked failed ~27 times a
# day, 192 times in seven days, all noise -- not noticed until 2026-07-27.
# Before adding a new queue, confirm it appears in data/queue_stats.json.
DISPATCH_TO_QUEUE = {
    "embed-images-dino-siglip": ["image-siglip-queue"],
}


def _load_linux_stats() -> dict:
    if not QUEUE_STATS_JSON.exists():
        return {}
    try:
        return json.loads(QUEUE_STATS_JSON.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _queue_healthy(stats: dict, queue_name: str, dispatch_iso: str) -> tuple[bool, str]:
    """Return (is_healthy, reason). Health rules:
    - backlog == 0: caught up
    - last_processed_at > dispatch_at: downstream embedded after dispatch
    - else: stuck"""
    q = stats.get(queue_name)
    if not q:
        return False, f"no probe data for {queue_name}"
    backlog = q.get("backlog")
    last_iso = q.get("last_processed_at")
    if backlog == 0:
        return True, f"{queue_name} caught up (backlog=0)"
    if last_iso and last_iso > dispatch_iso:
        return True, f"{queue_name} processed work after dispatch (last={last_iso})"
    age_h = ""
    if last_iso:
        try:
            last_dt = datetime.fromisoformat(last_iso.replace("Z", "+00:00"))
            age_h = f", idle {round((datetime.now(timezone.utc) - last_dt).total_seconds() / 3600, 1)}h"
        except ValueError:
            pass
    return False, f"{queue_name} backlog={backlog}, no progress since dispatch{age_h}"


def reconcile(min_age_minutes: int, dry_run: bool = False) -> dict:
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=min_age_minutes)).isoformat()
    linux_stats = _load_linux_stats()

    conn = sqlite3.connect(str(OPS_DB))
    conn.execute("PRAGMA busy_timeout = 30000")
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT id, skill_name, invocation_id, started_at FROM skill_runs "
        "WHERE status = 'dispatched' AND started_at < ?",
        (cutoff,),
    ).fetchall()
    stats = {"checked": len(rows), "marked_success": 0, "marked_failed": 0,
             "no_rule": 0, "unchanged": 0}
    for row_id, skill, inv_id, started in rows:
        queues = DISPATCH_TO_QUEUE.get(skill)
        if not queues:
            stats["no_rule"] += 1
            continue
        healthy = True
        reasons = []
        for q in queues:
            is_h, reason = _queue_healthy(linux_stats, q, started)
            reasons.append(reason)
            if not is_h:
                healthy = False
        new_status = "success" if healthy else "failed"
        summary_note = ("verified by downstream queue health: " if healthy
                        else "downstream queue did not finish in reasonable time: ") + "; ".join(reasons)
        if dry_run:
            print(f"  [would {new_status}] {skill} {inv_id[:8]}: {summary_note}")
            continue
        now = datetime.now(timezone.utc).isoformat()
        cur.execute(
            "UPDATE skill_runs SET status = ?, finished_at = ?, "
            "summary_json = json_set(COALESCE(summary_json, '{}'), '$.reconciled', ?) "
            "WHERE id = ?",
            (new_status, now, summary_note, row_id),
        )
        if new_status == "success":
            stats["marked_success"] += 1
        else:
            stats["marked_failed"] += 1
    conn.commit()
    conn.close()
    return stats


def main() -> int:
    p = argparse.ArgumentParser(
        description="Update fire-and-forget dispatched skill_runs based on "
                    "downstream queue health."
    )
    p.add_argument("--min-age-minutes", type=int, default=5,
                   help="Only reconcile rows older than this (default 5 min)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    s = reconcile(args.min_age_minutes, args.dry_run)
    print(f"checked={s['checked']} success={s['marked_success']} "
          f"failed={s['marked_failed']} no_rule={s['no_rule']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
