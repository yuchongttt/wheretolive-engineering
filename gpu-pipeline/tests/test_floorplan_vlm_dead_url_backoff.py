"""A property whose images can never be fetched must drop out of the claim
window, otherwise it blocks the whole backfill queue.

Background (2026-08-17 investigation): on 08-14 queue_config.cutoff_at was
removed to start backfilling 26.8k active listings. It ran for two days
(08-14: 2487 properties / 08-15: 1617) and then collapsed to 28 on 08-16. The
daemon had not crashed, vLLM returned 200, the mirror server was alive, inference
time was a normal 45s, and 24,556 properties were still queued — every proxy
metric was green.

The real cause was in the log: 398 of the last 400 ticks were "0 ok / 10 fail",
re-claiming the same 10 properties every 10 seconds, all reporting
`HTTP Error 404`. Those 10 had no row at all in floorplan_vlm_results.

The chain:
 1. Source image URLs can change over time. The DB held URLs ingested in
    May; the listing was still live, but the stored image URL now returned 404.
 2. process_batch classified `HTTP Error 404` as transient — together with
    500/502/503/timeout — so it took the `if not transient:` branch that skips
    insert_result: **no row written**.
 3. claim_active_batch's backoff criterion is "an ok=0 row exists with
    processed_at within 24h". No row -> the backoff never applies -> the next
    tick claims it again as-is.
 4. Add `ORDER BY MAX(scraped_at) DESC` + `--limit 10`: a property with dead URLs
    is pinned at the head of the queue permanently. It's a ratchet — usable
    slots = 10 - N, and when N reaches 10 throughput is zero. The backfill
    frontier got stuck at 2026-05-23.

So the invariant to guard is: **a failure to fetch the image must leave a trace;
only a VLM-server flake may leave no trace and retry immediately**. Both go
through the same except, and this allow-list is what tells them apart.

The line is drawn at "image vs server" rather than "404 vs 5xx" because the
costs of the two misjudgements differ by orders of magnitude: mistaking an image-host
blip for permanent costs this one property a day (the backfill is measured in
days anyway); mistaking a permanent 404 for a blip costs **the whole queue**.
And msg is truncated to 200 characters and lists only the first 3 URLs, so
judging per-URL permanence from the string was never reliable anyway. So every
image-fetch failure writes a row and the 24h backoff catches it uniformly; the
only thing that genuinely needs an immediate retry is the VLM server itself
(vLLM returns 503 while restarting, and a 24h backoff then would idle the whole
queue for a day).

A sample of 25 properties at the stalled frontier: 24 were 200 on both the
local mirror and the source URL — the backlog itself was clean, just blocked by these few at
the head of the queue, so this test guards throughput, not data quality.

(Public repo: the listing id and image URL below are synthetic.)
"""
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mac"))

import floorplan_vlm_analyze as fva  # noqa: E402

_UID = "10000001"
_DEAD_URL = "https://images.example.com/fp/10000001/0.jpeg"

# This is exactly what process_property raises when every candidate image fails
# (see call_vlm_multi in dispatcher.py).
_DEAD = f"all 1 URLs dead: [('{_DEAD_URL}', 'HTTP Error 404: Not Found')]"
_DEAD_5XX = f"all 1 URLs dead: [('{_DEAD_URL}', 'HTTP Error 503: Service Unavailable')]"
# vLLM itself flaking (it returns 503 on :8200 while restarting) — unrelated to
# whether the image can be fetched.
_VLM_FLAKY = "HTTP Error 503: Service Unavailable"


@pytest.fixture()
def conn():
    """Minimal DB sufficient for claim_active_batch + insert_result."""
    c = sqlite3.connect(":memory:")
    c.executescript("""
        CREATE TABLE rm_sales_overview (id TEXT PRIMARY KEY, delisted_date TEXT);
        CREATE TABLE rm_sales_images (
            property_id TEXT, idx INTEGER, url TEXT, type TEXT, scraped_at TEXT);
        CREATE TABLE queue_config (queue_name TEXT PRIMARY KEY, cutoff_at TEXT);
        CREATE TABLE floorplan_vlm_results (
            rm_uuid TEXT NOT NULL, floorplan_idx INTEGER NOT NULL,
            floorplan_url TEXT, model_version TEXT NOT NULL,
            total_sqft INTEGER, bedrooms INTEGER, bathrooms INTEGER,
            room_count INTEGER, layout_class TEXT, confidence TEXT,
            rooms_json TEXT, raw_response TEXT, inference_ms INTEGER,
            ok INTEGER NOT NULL DEFAULT 0, error TEXT,
            processed_at TEXT NOT NULL DEFAULT (datetime('now')),
            total_sqm REAL, reception_rooms INTEGER, ensuite_count INTEGER,
            floors INTEGER, has_garden INTEGER, has_garage INTEGER,
            has_balcony INTEGER, has_terrace INTEGER, has_fireplace INTEGER,
            total_windows INTEGER, total_doors INTEGER, adjacencies_json TEXT,
            raw_extraction_json TEXT, room_counts_json TEXT,
            bathroom_layout_json TEXT, per_floor_json TEXT, features_json TEXT,
            has_conservatory INTEGER,
            PRIMARY KEY (rm_uuid, floorplan_idx, model_version));
    """)
    c.execute("INSERT INTO queue_config VALUES ('floorplan-vlm-queue', NULL)")
    c.execute("INSERT INTO rm_sales_overview VALUES (?, NULL)", (_UID,))
    c.execute("INSERT INTO rm_sales_images VALUES (?, 0, ?, 'floorplan', ?)",
              (_UID, _DEAD_URL, "2026-05-23T17:01:53+00:00"))
    c.commit()
    return c


def _drain_once(conn, monkeypatch, err_msg):
    """Run one tick; process_property raises err_msg (no network, no VLM)."""
    def _boom(uid, fps):
        raise RuntimeError(err_msg)

    monkeypatch.setattr(fva, "process_property", _boom)
    batch = fva.claim_active_batch(conn, 10)
    assert batch, "precondition: this property should be in the queue to begin with"
    return fva.process_batch(conn, batch, concurrency=1)


class TestDeadImageUrl:
    """404 = the image URL is gone; this image is never coming back."""

    def test_failure_is_persisted(self, conn, monkeypatch):
        _drain_once(conn, monkeypatch, _DEAD)
        row = conn.execute(
            "SELECT ok, error FROM floorplan_vlm_results WHERE rm_uuid = ?",
            (_UID,)).fetchone()
        assert row is not None, "no row -> no basis for backoff -> re-claimed as-is next tick"
        assert row[0] == 0

    def test_drops_out_of_the_claim_window(self, conn, monkeypatch):
        """This is the direct criterion for the 08-16 collapse."""
        _drain_once(conn, monkeypatch, _DEAD)
        assert fva.claim_active_batch(conn, 10) == []


    def test_image_host_5xx_also_drops_out(self, conn, monkeypatch):
        """Can't fetch the image means can't fetch it — no branching on HTTP
        code; see the module docstring for why."""
        _drain_once(conn, monkeypatch, _DEAD_5XX)
        assert fva.claim_active_batch(conn, 10) == []


class TestVlmServerFlake:
    """A 503 while vLLM restarts should be retried on the next tick — the fix
    above must not hit it, or every vLLM restart would idle the whole queue for
    a day."""

    def test_stays_in_the_queue(self, conn, monkeypatch):
        _drain_once(conn, monkeypatch, _VLM_FLAKY)
        assert fva.claim_active_batch(conn, 10), "a server flake must not be backed off for 24h"

    def test_leaves_no_row(self, conn, monkeypatch):
        _drain_once(conn, monkeypatch, _VLM_FLAKY)
        assert conn.execute(
            "SELECT COUNT(*) FROM floorplan_vlm_results").fetchone()[0] == 0
