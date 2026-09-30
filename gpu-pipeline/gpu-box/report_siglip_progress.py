#!/usr/bin/env python3
"""Push SigLIP-2 luxury embedding daemon stats to mac admin every minute.
Sister to report_embed_progress.py. Same shape, different model + endpoint.
"""
import json, os, sqlite3, sys, urllib.request
from datetime import datetime, timezone, timedelta

DB    = "/data/ml/dataset/dataset.db"
API   = os.environ.get("MAC_ADMIN_URL", "http://mac-admin:3000")
KEY   = os.environ.get("WTL_ADMIN_KEY", "")
HOST  = os.environ.get("WTL_HOST_ID",   "gpu-box")
URL   = f"{API}/api/admin/siglip-worker"
MODEL = "siglip-lux128"

if not KEY:
    print("missing WTL_ADMIN_KEY", file=sys.stderr); sys.exit(1)

conn = sqlite3.connect(DB)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA busy_timeout=5000")

# Total downloaded (universe), how many SigLIP can actually embed, and
# how many it already has. Match embed_daemon_combined.py's pull_batch
# filter exactly: media_type IN ('photo','sold') — daemon skips
# floorplans (line art breaks the SigLIP embedding distribution). Without
# this filter, ~2k perpetually-stuck floorplans show as "pending" in the
# admin UI even though the daemon will never touch them.
total_dl = conn.execute("SELECT COUNT(*) FROM images WHERE status='done' AND local_path IS NOT NULL").fetchone()[0]
eligible = conn.execute(
    "SELECT COUNT(*) FROM images WHERE status='done' AND local_path IS NOT NULL AND media_type IN ('photo','sold')"
).fetchone()[0]
skipped  = total_dl - eligible  # floorplans, mostly
embedded = conn.execute("SELECT COUNT(*) FROM image_embeddings WHERE model=? AND url NOT LIKE 'local:%'", (MODEL,)).fetchone()[0]
pending  = max(0, eligible - embedded)

# Throughput by created_at — last 60s and last 10min finished embeddings.
_now  = datetime.now(timezone.utc)
_t1m  = (_now - timedelta(seconds=60)).isoformat()
_t10m = (_now - timedelta(seconds=600)).isoformat()
count_1m  = conn.execute("SELECT COUNT(*) FROM image_embeddings WHERE model=? AND created_at >= ?", (MODEL, _t1m)).fetchone()[0]
count_10m = conn.execute("SELECT COUNT(*) FROM image_embeddings WHERE model=? AND created_at >= ?", (MODEL, _t10m)).fetchone()[0]

added_1h  = conn.execute("SELECT COUNT(*) FROM image_embeddings WHERE model=? AND created_at >= datetime('now','-1 hour')", (MODEL,)).fetchone()[0]
added_24h = conn.execute("SELECT COUNT(*) FROM image_embeddings WHERE model=? AND created_at >= datetime('now','-1 day')", (MODEL,)).fetchone()[0]
added_7d  = conn.execute("SELECT COUNT(*) FROM image_embeddings WHERE model=? AND created_at >= datetime('now','-7 days')", (MODEL,)).fetchone()[0]

# Bonus: how many of the just-embedded rows are interior (is_interior=1)
# in image_luxury_score_v2. Lets the admin show the indoor:outdoor split
# without a separate endpoint.
try:
    interior_pct = conn.execute("""
        SELECT 100.0 * SUM(is_interior) / NULLIF(COUNT(*), 0)
        FROM image_luxury_score_v2
        WHERE computed_at >= datetime('now','-1 day')
    """).fetchone()[0]
    interior_pct = round(interior_pct, 1) if interior_pct is not None else None
except sqlite3.OperationalError:
    interior_pct = None  # table doesn't exist yet (first hour of daemon)

# Two-level rate + ETA (1m for "alive right now", 10m for steady-state).
rps_1m  = count_1m / 60
rps_10m = count_10m / 600
eta_1m  = int(pending / rps_1m)  if rps_1m  > 0 else None
eta_10m = int(pending / rps_10m) if rps_10m > 0 else None

stats = {
    "model":            MODEL,
    "total_universe":   total_dl,
    # SigLIP skips floorplans (see comment at filter above), so eligible
    # < total_universe by ~2k. Coverage is relative to eligible, not
    # universe, otherwise it asymptotes to ~99.9% forever.
    "eligible":         eligible,
    "skipped":          skipped,
    "embedded":         embedded,
    "pending":          pending,
    "coverage_pct":     round(100.0 * embedded / max(eligible, 1), 2),
    # Legacy aliases.
    "rate_1m":          count_1m,
    "rate_10m_avg_rps": round(rps_10m, 2),
    "eta_seconds":      eta_10m,
    # Two-level rate + ETA.
    "rate_1m_avg_rps":  round(rps_1m, 2),
    "rate_10m_count":   count_10m,
    "eta_seconds_1m":   eta_1m,
    "eta_seconds_10m":  eta_10m,
    "added_1h":         added_1h,
    "added_24h":        added_24h,
    "added_7d":         added_7d,
    "interior_pct_24h": interior_pct,
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
