#!/usr/bin/env python3
"""Push DINOv3-L embedding daemon stats to mac admin every minute."""
import json, os, sqlite3, sys, urllib.request
from datetime import datetime, timezone, timedelta

DB    = "/data/ml/dataset/dataset.db"
API   = os.environ.get("MAC_ADMIN_URL", "http://mac-admin:3000")
KEY   = os.environ.get("WTL_ADMIN_KEY", "")
HOST  = os.environ.get("WTL_HOST_ID",   "gpu-box")
URL   = f"{API}/api/admin/embedding-worker"
MODEL = "dinov3-l16-base"

# DINO embed daemon's pull_batch filters to media_type='photo' — it
# skips floorplans (2k) and sold-property images (33k). Match that
# filter here so `pending` means "eligible but not yet embedded" and a
# new "skipped" stat surfaces what the daemon will never touch.
DINO_ELIGIBLE_TYPES = ("photo",)

if not KEY:
    print("missing WTL_ADMIN_KEY", file=sys.stderr); sys.exit(1)

conn = sqlite3.connect(DB)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA busy_timeout=5000")

# Universe split into "eligible" (DINO will embed) vs "skipped" (daemon
# filter excludes — sold-property images, floorplans). pending is now
# eligible - embedded, never negative even on backlog gulps. embedded
# filters `url NOT LIKE 'local:%'` to drop synthetic m1a training
# variants (rows that live in image_embeddings but aren't from the
# image cache).
total_dl = conn.execute("SELECT COUNT(*) FROM images WHERE status='done' AND local_path IS NOT NULL").fetchone()[0]
eligible = conn.execute(
    "SELECT COUNT(*) FROM images WHERE status='done' AND local_path IS NOT NULL AND media_type='photo'"
).fetchone()[0]
skipped  = total_dl - eligible
embedded = conn.execute("SELECT COUNT(*) FROM image_embeddings WHERE model=? AND url NOT LIKE 'local:%'", (MODEL,)).fetchone()[0]
pending  = max(0, eligible - embedded)

# Throughput by created_at — last 60s and last 10min finished embeddings.
_now  = datetime.now(timezone.utc)
_t1m  = (_now - timedelta(seconds=60)).isoformat()
_t10m = (_now - timedelta(seconds=600)).isoformat()
count_1m  = conn.execute("SELECT COUNT(*) FROM image_embeddings WHERE model=? AND created_at >= ?", (MODEL, _t1m)).fetchone()[0]
count_10m = conn.execute("SELECT COUNT(*) FROM image_embeddings WHERE model=? AND created_at >= ?", (MODEL, _t10m)).fetchone()[0]

# Window adds — how many image_embeddings rows were created in each window.
# Lets the admin UI compare embed-rate vs downloader-rate at the same window.
added_1h  = conn.execute("SELECT COUNT(*) FROM image_embeddings WHERE model=? AND created_at >= datetime('now','-1 hour')", (MODEL,)).fetchone()[0]
added_24h = conn.execute("SELECT COUNT(*) FROM image_embeddings WHERE model=? AND created_at >= datetime('now','-1 day')", (MODEL,)).fetchone()[0]
added_7d  = conn.execute("SELECT COUNT(*) FROM image_embeddings WHERE model=? AND created_at >= datetime('now','-7 days')", (MODEL,)).fetchone()[0]

# Two-level rate + ETA — 1m for "is it actually running right now",
# 10m for steady-state planning. 1m is noisy when batches are bursty
# (one 64-row batch every ~10s = either 0 or 6.4 RPS instantaneously);
# 10m smooths that.
rps_1m  = count_1m / 60
rps_10m = count_10m / 600
eta_1m  = int(pending / rps_1m)  if rps_1m  > 0 else None
eta_10m = int(pending / rps_10m) if rps_10m > 0 else None

stats = {
    "model":            MODEL,
    "total_universe":   total_dl,
    "eligible":         eligible,
    "skipped":          skipped,
    "embedded":         embedded,
    "pending":          pending,
    # min(100, ...) — embedded can briefly exceed eligible right after a
    # backfill reclassifies images from 'photo' to a skipped media_type
    # (those embeddings still exist in image_embeddings; harmless).
    "coverage_pct":     round(min(100.0, 100.0 * embedded / max(eligible, 1)), 2),
    # Legacy aliases retained so older client builds don't crash.
    "rate_1m":          count_1m,
    "rate_10m_avg_rps": round(rps_10m, 2),
    "eta_seconds":      eta_10m,
    # New two-level metrics.
    "rate_1m_avg_rps":  round(rps_1m, 2),
    "rate_10m_count":   count_10m,
    "eta_seconds_1m":   eta_1m,
    "eta_seconds_10m":  eta_10m,
    "added_1h":         added_1h,
    "added_24h":        added_24h,
    "added_7d":         added_7d,
}
payload = {"host_id": HOST, "reported_at": _now.isoformat(), "stats": stats}
req = urllib.request.Request(
    URL, data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "x-admin-key": KEY},
)
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode()
        if '"ok":true' not in body:
            print(f"[report] unexpected response: {body}", file=sys.stderr); sys.exit(1)
except Exception as e:
    print(f"[report] POST failed: {e}", file=sys.stderr); sys.exit(1)
