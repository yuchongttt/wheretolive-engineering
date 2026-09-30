#!/usr/bin/env python3
"""Probe GPU-box queue counts via SSH and write to data/queue_stats.json.

Why a separate script + file: the web app's Next.js process spawned by
launchd has no usable SSH agent and was emitting exit-255-silent for
ssh-from-node. Running this script from a normal shell context (cron or
ad-hoc) reuses the user's SSH agent/config and works reliably.

Output schema:
  {
    "probed_at": "2026-05-28T13:15:00Z",
    "stale_seconds": null,           // filled by reader
    "image-download-queue":    {"backlog": ..., "done_lifetime": ..., ...},
    "image-siglip-queue":      {...}
  }

Recommended: cron this every 60-120 seconds (or run it manually).
API treats stats >5 min old as warn, >30 min old as unhealthy.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "data" / "queue_stats.json"
OPS_DB = REPO_ROOT / "data" / "ops.db"
EVAL_DB = REPO_ROOT / "data" / "evaluations.db"
# Set GPU_HOST=user@<private-network IP>, not an ssh-config alias — every other producer
# (image enqueue, floorplan dispatch, decor drain) uses the explicit target,
# and depending on a per-host ~/.ssh/config alias made this the one fragile
# caller (would silently fail under launchd if the alias wasn't in scope).
SSH_TARGET = os.environ.get("GPU_HOST", "gpu-box")

REMOTE_PY = r"""
import sqlite3, json
c = sqlite3.connect("/data/ml/dataset/dataset.db").cursor()
def n(sql, *args):
    r = c.execute(sql, args).fetchone()
    return r[0] if r and r[0] is not None else 0
out = {}
# image-download-queue — covers all media types
out["image-download-queue"] = {
    "backlog": n("SELECT COUNT(*) FROM images WHERE status='pending'"),
    "done_lifetime": n("SELECT COUNT(*) FROM images WHERE status='done'"),
    "failed_lifetime": n("SELECT COUNT(*) FROM images WHERE status='failed'"),
    # Download rate uses completed_at (actual download time), not scraped_at.
    # datetime() wraps the ISO-8601 'T'-separated column so it compares correctly
    # against datetime('now') (space-separated). Without it, 'T'>' ' makes every
    # row dated today appear "recent" regardless of actual time.
    "done_last_10min": n("SELECT COUNT(*) FROM images WHERE status='done' AND datetime(completed_at) > datetime('now','-10 minutes')"),
    "done_last_hour": n("SELECT COUNT(*) FROM images WHERE status='done' AND datetime(completed_at) > datetime('now','-1 hour')"),
    "done_last_24h": n("SELECT COUNT(*) FROM images WHERE status='done' AND datetime(completed_at) > datetime('now','-24 hours')"),
    "last_processed_at": c.execute("SELECT MAX(completed_at) FROM images WHERE status='done'").fetchone()[0],
    "policy": "all media_types",
    "policy_scope": "active+sold+floorplan",
    "skipped_by_policy": 0,
}
# SigLIP daemon's scope: media_type IN ('photo','sold') (floorplan skipped).
# DINO (dinov3-l16-base) retired 2026-06-28 — no longer probed.
DAEMON_SCOPE = {
    "image-siglip-queue": ("siglip-lux128", ("photo","sold"), "photo + sold (skip floorplan)"),
}
for qname, (model, types, policy) in DAEMON_SCOPE.items():
    placeholders = ",".join("?" * len(types))
    backlog = n(f"SELECT COUNT(*) FROM images i WHERE i.status='done' AND i.media_type IN ({placeholders}) AND NOT EXISTS (SELECT 1 FROM image_embeddings e WHERE e.url=i.url AND e.model=?)",
                *types, model)
    # Show how many images are outside the daemon's scope (won't get processed by design)
    skipped = n(f"SELECT COUNT(*) FROM images i WHERE i.status='done' AND i.media_type NOT IN ({placeholders})", *types)
    done = n("SELECT COUNT(*) FROM image_embeddings WHERE model=?", model)
    rate10 = n("SELECT COUNT(*) FROM image_embeddings WHERE model=? AND datetime(created_at) > datetime('now','-10 minutes')", model)
    rate1 = n("SELECT COUNT(*) FROM image_embeddings WHERE model=? AND datetime(created_at) > datetime('now','-1 hour')", model)
    rate24 = n("SELECT COUNT(*) FROM image_embeddings WHERE model=? AND datetime(created_at) > datetime('now','-24 hours')", model)
    last = c.execute("SELECT MAX(created_at) FROM image_embeddings WHERE model=?", (model,)).fetchone()[0]
    out[qname] = {"backlog": backlog, "done_lifetime": done, "failed_lifetime": 0,
                  "done_last_10min": rate10, "done_last_hour": rate1, "done_last_24h": rate24,
                  "last_processed_at": last,
                  "policy": policy, "policy_scope": "+".join(types), "skipped_by_policy": skipped}
# aesthetic-score-queue — OneAlign aesthetic-scoring backlog (image_aesthetic_score).
# SigLIP2 classify/keep is the prerequisite (like the download step); this queue
# is the OneAlign scoring work. The worker (aesthetic_score_daemon.py) stamps
# scored_at on every processed row (aes>=0 real, -1 missing, -2 poison), so
# throughput is a REAL time-window count like the download/siglip queues — no
# Mac-side backlog-delta estimate. datetime(scored_at) wraps the column so it
# compares cleanly against datetime('now') regardless of 'T' vs ' ' separator.
try:
    _ab = n("SELECT COUNT(*) FROM image_aesthetic_score WHERE kept=1 AND aes IS NULL")
    _ad = n("SELECT COUNT(*) FROM image_aesthetic_score WHERE kept=1 AND aes IS NOT NULL AND aes >= 0")
    _af = n("SELECT COUNT(*) FROM image_aesthetic_score WHERE aes < 0")
    _ap = n("SELECT COUNT(*) FROM property_aesthetic_score")
    out["aesthetic-score-queue"] = {
        "backlog": _ab, "done_lifetime": _ad, "failed_lifetime": _af,
        "done_last_10min": n("SELECT COUNT(*) FROM image_aesthetic_score WHERE datetime(scored_at) > datetime('now','-10 minutes')"),
        "done_last_hour":  n("SELECT COUNT(*) FROM image_aesthetic_score WHERE datetime(scored_at) > datetime('now','-1 hour')"),
        "done_last_24h":   n("SELECT COUNT(*) FROM image_aesthetic_score WHERE datetime(scored_at) > datetime('now','-24 hours')"),
        "last_processed_at": c.execute("SELECT MAX(scored_at) FROM image_aesthetic_score").fetchone()[0],
        "policy": "kept photos; %d props aggregated" % _ap, "policy_scope": "SW+E+EC resale",
        "skipped_by_policy": 0,
    }
except Exception as _e:
    out["aesthetic-score-queue"] = {"backlog": None, "done_lifetime": None,
        "failed_lifetime": None, "done_last_10min": None, "done_last_hour": None,
        "done_last_24h": None, "last_processed_at": None, "probe_error": str(_e)[:200]}
print(json.dumps(out))
"""


def main() -> int:
    started = time.time()
    try:
        proc = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
             SSH_TARGET, "python3", "-"],
            input=REMOTE_PY,
            # 2026-06-13: 120→240s. Remote NOT-EXISTS join over images ×
            # image_embeddings (millions of rows each) normally takes 83-118s; the 20-outcode
            # batch grew the tables past the old 120s edge → persistent "ssh
            # timeout". StartInterval=300s leaves margin. Killing at 120s just
            # wasted a full heavy query every cycle.
            capture_output=True, text=True, timeout=240,
        )
    except subprocess.TimeoutExpired:
        print("[probe] ssh timeout", file=sys.stderr)
        return 1
    if proc.returncode != 0:
        print(f"[probe] ssh exit {proc.returncode}: {proc.stderr[:300]}", file=sys.stderr)
        return 1
    try:
        data = json.loads(proc.stdout.strip())
    except json.JSONDecodeError as e:
        print(f"[probe] json parse: {e}; stdout head: {proc.stdout[:200]}", file=sys.stderr)
        return 1
    probed_at = datetime.now(timezone.utc).isoformat()
    # aesthetic-score-queue throughput now comes from real scored_at windows in
    # REMOTE_PY (the worker stamps scored_at per row), same as every other queue —
    # the old Mac-side backlog-delta estimate is gone.
    out = {
        "probed_at": probed_at,
        "probe_duration_s": round(time.time() - started, 2),
        **data,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(OUT_PATH)
    backlog_total = sum((v.get("backlog") or 0) for v in data.values())
    print(f"[probe] OK in {out['probe_duration_s']}s, total backlog {backlog_total}", flush=True)

    # Also snapshot Mac queues + write history rows for sparklines.
    try:
        _snapshot_history(data)
    except Exception as e:  # noqa: BLE001
        print(f"[probe] history snapshot failed (non-fatal): {e}", file=sys.stderr)
    return 0


def _snapshot_history(linux_data: dict) -> None:
    """Append one history row per queue to ops.db queue_stats_history.
    Combines GPU-box stats (from probe) with Mac queues (queried fresh
    via evaluations.db) into one chronological snapshot row per queue.

    Public-repo note: the private version also snapshots the Mac-side data
    pipeline queues (enrichment, postcode evaluation, listing refresh, ...)
    here. Those belong to components not included in this repo and were
    removed; only the floorplan-VLM queue (part of this module) is kept."""
    import sqlite3
    ts = datetime.now(timezone.utc).isoformat()
    rows = []
    # GPU box: use the data we just probed
    for qname, qdata in linux_data.items():
        rows.append((ts, qname,
                     qdata.get("backlog"), qdata.get("done_last_hour"),
                     qdata.get("done_lifetime")))
    # Mac queues: query fresh
    e = sqlite3.connect(str(EVAL_DB))
    # floorplan-vlm-queue
    cutoff = e.execute("SELECT cutoff_at FROM queue_config WHERE queue_name='floorplan-vlm-queue'").fetchone()
    cutoff = cutoff[0] if cutoff else None
    fp_backlog = e.execute("""
        SELECT COUNT(DISTINCT i.property_id)
        FROM rm_sales_images i
        JOIN rm_sales_overview o ON o.id=i.property_id
        WHERE i.type='floorplan' AND i.url IS NOT NULL
          AND o.delisted_date IS NULL
          AND (? IS NULL OR i.scraped_at > ?)
          AND NOT EXISTS (SELECT 1 FROM floorplan_vlm_results r
                          WHERE r.rm_uuid=i.property_id
                            AND r.model_version LIKE 'qwen3.6-27b-autoround-v6.6%'
                            AND (r.ok=1
                                 OR (r.ok=0 AND r.processed_at > datetime('now','-24 hours'))))
    """, (cutoff, cutoff)).fetchone()[0]
    fp_done = e.execute("SELECT COUNT(*) FROM floorplan_vlm_results WHERE model_version LIKE 'qwen3.6-27b-autoround-v6.6%' AND ok=1").fetchone()[0]
    fp_rate1h = e.execute("SELECT COUNT(*) FROM floorplan_vlm_results WHERE model_version LIKE 'qwen3.6-27b-autoround-v6.6%' AND ok=1 AND processed_at > datetime('now','-1 hour')").fetchone()[0]
    rows.append((ts, "floorplan-vlm-queue", fp_backlog, fp_rate1h, fp_done))
    e.close()

    o = sqlite3.connect(str(OPS_DB))
    o.execute("PRAGMA busy_timeout = 30000")
    o.executemany(
        "INSERT OR IGNORE INTO queue_stats_history "
        "(ts, queue_name, backlog, done_last_hour, done_lifetime) "
        "VALUES (?,?,?,?,?)", rows)
    # Keep only last 7 days; older rows trimmed every probe (cheap)
    o.execute("DELETE FROM queue_stats_history WHERE ts < datetime('now','-7 days')")
    o.commit()
    o.close()


if __name__ == "__main__":
    sys.exit(main())
