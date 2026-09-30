#!/usr/bin/env python3
"""floorplan-vlm-analyze — one-shot batch processor that runs the Qwen3.6-27B
v6.6 VLM over a queue of floorplan images.

Reuses the production dispatcher's helpers (`call_vlm_multi`,
`insert_result`, `process_property`, `extract_room_area`, etc.) so prompts
and post-processing stay in lockstep with `mac/floorplan_vlm/dispatcher.py`
(the existing always-on launchd dispatcher that handles sold.db backfill).

Difference vs that dispatcher:
- One-shot: claims a batch, processes, exits — invokable as a skill
- Two queue sources:
    * `--source active` (default) — active listings in rm_sales_overview
      with floorplan URLs in rm_sales_images that haven't been processed
      at the current MODEL_VERSION
    * `--source sold` — falls back to the original sold.db backfill queue
    * `--source both` — active first, then top up from sold
- `--ids` mode: process specific rm_uuids
- Tracked in skill_runs via SkillRun
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent / "floorplan_vlm"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import the production dispatcher's helpers. Prompts, model version,
# post-processing, all stay in lockstep. dispatcher.py defaults
# WTL_VLM_MODEL_VER to v6.6 (matches launchd plist).
import dispatcher  # noqa: E402
from dispatcher import (  # noqa: E402
    AGGREGATED_IDX,
    CONCURRENCY,
    DB,
    MODEL_VERSION,
    VLM_URL,
    _derive_bathrooms,
    _derive_bedrooms,
    _derive_receptions,
    insert_result,
    process_property,
)
from skill_report import SkillRun  # noqa: E402

SOLD_DB = REPO_ROOT / "data" / "sold.db"


def claim_active_batch(
    conn: sqlite3.Connection, n_properties: int,
) -> list[tuple[str, list[tuple[int, str]]]]:
    """Queue from active listings (rm_sales_overview + rm_sales_images),
    honoring the floorplan-vlm-queue cutoff watermark.

    queue_config.cutoff_at is a single timestamp configured per queue —
    only properties with at least one floorplan URL ingested AFTER cutoff
    are eligible. This intentionally skips legacy backlog (e.g. 57652
    pre-2026-05-28 properties) so the queue starts clean for new arrivals.
    Newest-floorplan-first ordering so today's listings get processed
    sooner than older queued ones."""
    cutoff_row = conn.execute(
        "SELECT cutoff_at FROM queue_config WHERE queue_name = 'floorplan-vlm-queue'",
    ).fetchone()
    cutoff = cutoff_row[0] if cutoff_row else None

    rows = conn.execute(
        """
        SELECT i.property_id, i.idx, i.url
        FROM rm_sales_images i
        JOIN rm_sales_overview o ON o.id = i.property_id
        WHERE i.type = 'floorplan'
          AND i.url IS NOT NULL
          AND o.delisted_date IS NULL
          AND i.property_id IN (
            SELECT i2.property_id FROM rm_sales_images i2
            JOIN rm_sales_overview o2 ON o2.id = i2.property_id
            WHERE i2.type = 'floorplan' AND i2.url IS NOT NULL
              AND o2.delisted_date IS NULL
              AND (? IS NULL OR i2.scraped_at > ?)
              -- Skip if already succeeded at v6.6, OR if it failed within the
            -- last 24h. The PK (rm_uuid, idx, model_version) means each
            -- failed attempt OVERWRITES the prior ok=0 row, so we can't
            -- count attempts — instead we back off via the failure
            -- timestamp. Bad images (parse fail forever) re-fail then
            -- re-skip 24h, never blocking the queue. Transient failures
            -- retry next day.
            AND NOT EXISTS (
              SELECT 1 FROM floorplan_vlm_results r
              WHERE r.rm_uuid = i2.property_id
                AND r.model_version LIKE 'qwen3.6-27b-autoround-v6.6%'
                AND (r.ok = 1
                     OR (r.ok = 0 AND r.processed_at > datetime('now', '-24 hours')))
            )
            GROUP BY i2.property_id
            ORDER BY MAX(i2.scraped_at) DESC
            LIMIT ?
          )
        ORDER BY i.property_id, i.idx
        """,
        (cutoff, cutoff, n_properties),
    ).fetchall()
    grouped: dict[str, list[tuple[int, str]]] = {}
    for uid, idx, url in rows:
        grouped.setdefault(uid, []).append((idx, url))
    return list(grouped.items())


def claim_by_ids(
    conn: sqlite3.Connection, ids: list[str],
) -> list[tuple[str, list[tuple[int, str]]]]:
    """Override queue with explicit property_ids. Pulls floorplans from
    rm_sales_images (active) OR sold.db (whichever has rows)."""
    grouped: dict[str, list[tuple[int, str]]] = {}
    placeholders = ",".join("?" * len(ids))
    # Active
    rows = conn.execute(
        f"SELECT property_id, idx, url FROM rm_sales_images "
        f"WHERE type='floorplan' AND url IS NOT NULL "
        f"  AND property_id IN ({placeholders}) "
        f"ORDER BY property_id, idx",
        ids,
    ).fetchall()
    for uid, idx, url in rows:
        grouped.setdefault(uid, []).append((idx, url))
    # Fill gaps from sold.db
    missing = [i for i in ids if i not in grouped]
    if missing:
        rows = conn.execute(
            f"SELECT rm_uuid, idx, url FROM sold.rm_sold_floorplans "
            f"WHERE url IS NOT NULL AND rm_uuid IN ({','.join('?' * len(missing))}) "
            f"ORDER BY rm_uuid, idx",
            missing,
        ).fetchall()
        for uid, idx, url in rows:
            grouped.setdefault(uid, []).append((idx, url))
    return list(grouped.items())


def process_batch(
    conn: sqlite3.Connection,
    batch: list[tuple[str, list[tuple[int, str]]]],
    concurrency: int,
) -> tuple[int, int, list[dict]]:
    """Mirrors dispatcher.process_one_tick but parameterised. Returns
    (ok, err, errors-summary-list)."""
    if not batch:
        return 0, 0, []
    total_imgs = sum(len(fps) for _, fps in batch)
    print(f"[claim] {len(batch)} properties ({total_imgs} images) | "
          f"model={MODEL_VERSION} -> {VLM_URL} (conc={concurrency})",
          flush=True)
    ok = err = 0
    errors: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(process_property, uid, fps): uid for uid, fps in batch}
        for fut in as_completed(futures):
            uid = futures[fut]
            try:
                uid, vlm, joined_url = fut.result()
                n_img = vlm.get("n_images_used", 1)
                insert_result(conn, uid, AGGREGATED_IDX, joined_url, vlm)
                # Commit each property immediately so other writers (postcode
                # drain, probe, ingest jobs) aren't blocked for ~12 min until
                # the whole batch finishes. This is critical when multiple
                # workers share evaluations.db.
                conn.commit()
                if vlm.get("parsed"):
                    p = vlm["parsed"]
                    ok += 1
                    bd = _derive_bedrooms(p) or "-"
                    ba = _derive_bathrooms(p) or "-"
                    rc = _derive_receptions(p) or "-"
                    print(f"  [ok ] {uid[:8]} imgs={n_img} "
                          f"{p.get('total_sqft','-')}sf {bd}bd/{ba}ba/{rc}rec "
                          f"{p.get('layout_class','-')} {p.get('confidence','-')} "
                          f"({vlm.get('inference_ms','-')}ms)", flush=True)
                else:
                    err += 1
                    errors.append({"id": uid, "kind": "parse_fail",
                                   "raw_head": (vlm.get("raw") or "")[:120]})
                    print(f"  [err] {uid[:8]} parse fail", flush=True)
            except Exception as e:  # noqa: BLE001
                err += 1
                msg = str(e)[:200]
                errors.append({"id": uid, "kind": "exception", "err": msg})
                # "all N URLs dead" (dispatcher.py) means every candidate image
                # failed — local mirror AND the image URL. That is never worth an
                # immediate retry, so it must leave an ok=0 row: claim_active_batch's
                # 24h backoff keys on that row existing. Without one the property is
                # re-claimed every tick, and since the queue is newest-first
                # it squats at the head forever — 10 such properties starve the whole
                # backfill (2026-08-17: 398 of the last 400 ticks were "0 ok / 10
                # fail" on the same 10 rows, all floorplan image hashes rotated at the
                # source since the rows were ingested in May, backfill frontier
                # frozen at 2026-05-23).
                #
                # We don't sub-classify by HTTP code here: msg is truncated to 200
                # chars and lists only the first 3 URLs, so per-URL permanence isn't
                # recoverable from it. The asymmetry settles it anyway — treating a
                # image-host blip as permanent costs this one property a day, treating a
                # permanent 404 as transient costs the entire queue.
                dead_images = "URLs dead" in msg
                transient = not dead_images and any(s in msg for s in (
                    "HTTP Error 404", "HTTP Error 500", "HTTP Error 502",
                    "HTTP Error 503", "ConnectionError", "Connection refused",
                    "timed out", "Read timeout", "URLError",
                    # str(URLError) renders as "<urlopen error ...>" (the class
                    # name never appears) — e.g. DNS blips like
                    # "[Errno 8] nodename nor servname provided" 2026-06-04.
                    "urlopen error",
                ))
                if not transient:
                    insert_result(conn, uid, AGGREGATED_IDX, "", None,
                                  f"skill_exception: {msg}")
                    conn.commit()
                print(f"  [exc] {uid[:8]}: {msg}", flush=True)
    return ok, err, errors[:10]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Qwen3.6-27B v6.6 VLM over a queue of property floorplans."
    )
    p.add_argument("--limit", type=int, default=30,
                   help="Max properties per run (default 30 ≈ 30 min at ~60s each)")
    p.add_argument("--source", choices=["active", "sold", "both"], default="active",
                   help="Which queue to drain (default active: new listings)")
    p.add_argument("--ids", default=None,
                   help="Comma-separated rm_uuids (overrides queue)")
    p.add_argument("--concurrency", type=int, default=CONCURRENCY,
                   help=f"Parallel VLM HTTP requests (default {CONCURRENCY})")
    p.add_argument("--daemon", action="store_true",
                   help="Run forever as a daemon: drain --limit per tick, then "
                        "sleep --idle-sleep (queue empty) or --active-sleep "
                        "(queue had work). Each tick is its own SkillRun.")
    p.add_argument("--active-sleep", type=int, default=10,
                   help="Sleep between non-empty ticks (seconds, default 10)")
    p.add_argument("--idle-sleep", type=int, default=120,
                   help="Sleep when queue empty (seconds, default 120)")
    p.add_argument("--parent", default=None)
    p.add_argument("--invoked-by", default="manual")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.daemon:
        return _run_daemon(args)
    _run_once(args)
    return 0  # one-shot exit code stays 0; _run_once now returns work-count


def _run_daemon(args) -> int:
    """Loop forever: drain a batch, write SkillRun, sleep, repeat.

    Skill_runs gets one row per tick — admin sees daemon's recent batch
    history same as any manual invocation. Sleep is shorter when queue
    had work (10s) vs idle (120s) so we react quickly but don't hammer
    the DB when nothing's happening."""
    invoked_by = args.invoked_by if args.invoked_by != "manual" else "daemon"
    tick = 0
    print(f"[daemon] floorplan-vlm-analyze starting; limit={args.limit}, "
          f"idle={args.idle_sleep}s, active={args.active_sleep}s", flush=True)
    while True:
        tick += 1
        # Build a per-tick args namespace so each tick is its own SkillRun
        from copy import copy
        tick_args = copy(args)
        tick_args.daemon = False
        tick_args.invoked_by = invoked_by
        n_done = _run_once(tick_args)
        # Sleep short while ANY queue (active OR sold backfill) still has work,
        # long only when both are fully drained. Basing this on whether the tick
        # actually processed properties — rather than re-querying the active-only
        # _check_backlog() — keeps the sold backfill running back-to-back under
        # --source both instead of idling idle_sleep between every batch.
        sleep_s = args.active_sleep if n_done > 0 else args.idle_sleep
        print(f"[daemon] tick {tick} done, processed={n_done}, "
              f"sleep {sleep_s}s", flush=True)
        time.sleep(sleep_s)
        # Loop never exits; launchd KeepAlive restarts on crash


def _check_backlog() -> int:
    conn = sqlite3.connect(str(DB))
    try:
        conn.execute(f"ATTACH DATABASE '{SOLD_DB}' AS sold")
    except Exception:
        pass
    row = conn.execute("""
        SELECT COUNT(DISTINCT i.property_id)
        FROM rm_sales_images i
        JOIN rm_sales_overview o ON o.id = i.property_id
        WHERE i.type = 'floorplan' AND i.url IS NOT NULL
          AND o.delisted_date IS NULL
          AND i.scraped_at > COALESCE(
            (SELECT cutoff_at FROM queue_config WHERE queue_name='floorplan-vlm-queue'),
            '0000')
          AND NOT EXISTS (SELECT 1 FROM floorplan_vlm_results r
                          WHERE r.rm_uuid = i.property_id
                            AND r.model_version LIKE 'qwen3.6-27b-autoround-v6.6%'
                            AND (r.ok = 1
                                 OR (r.ok = 0 AND r.processed_at > datetime('now', '-24 hours'))))
    """).fetchone()
    conn.close()
    return row[0] if row else 0


def _run_once(args) -> int:
    started = time.time()
    with SkillRun("floorplan-vlm-analyze",
                  invoked_by=args.invoked_by, parent=args.parent) as run:
        run.summary["params"] = {
            "limit": args.limit, "source": args.source,
            "ids": args.ids, "concurrency": args.concurrency,
            "model_version": MODEL_VERSION,
            "vlm_url": VLM_URL,
        }
        conn = sqlite3.connect(str(DB))
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute(f"ATTACH DATABASE '{SOLD_DB}' AS sold")

        if args.ids:
            ids = [s.strip() for s in args.ids.split(",") if s.strip()]
            batch = claim_by_ids(conn, ids)
        else:
            batch = []
            if args.source in ("active", "both"):
                batch.extend(claim_active_batch(conn, args.limit))
            if args.source in ("sold", "both") and len(batch) < args.limit:
                top_up = dispatcher.claim_property_batch(conn, args.limit - len(batch))
                batch.extend(top_up)

        run.summary["queue_size"] = len(batch)
        if not batch:
            run.summary["final"] = "queue empty, no floorplans to analyse"
            print(run.summary["final"], flush=True)
            conn.close()
            return 0

        ok, err, errors = process_batch(conn, batch, args.concurrency)
        conn.close()
        elapsed = time.time() - started
        run.summary["stats"] = {"ok": ok, "failed": err}
        run.summary["errors"] = errors
        run.summary["elapsed_s"] = round(elapsed, 1)
        run.summary["final"] = (
            f"VLM analysis done | {ok} properties ok / {err} failed | "
            f"took {round(elapsed/60, 1)}m | model={MODEL_VERSION}"
        )
        print(run.summary["final"], flush=True)
    return len(batch)  # properties processed this tick; drives daemon sleep pacing


if __name__ == "__main__":
    sys.exit(main())
