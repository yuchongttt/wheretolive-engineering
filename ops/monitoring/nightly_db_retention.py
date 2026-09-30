#!/usr/bin/env python3
"""Retention sweep for evaluations.db.

Drops:
  - dim_* rows whose data_version != current CACHE_VERSION
  - audit rows older than AUDIT_KEEP_DAYS

Has a sanity guard: refuses to delete > MAX_DELETE_FRACTION of any single
table in one pass (catches typo'd version constants that would wipe a table).

Usage:
  python3 nightly_db_retention.py [--dry-run] [--vacuum]

(Public extract: the original also swept a pipeline job-queue DB; that branch
is omitted here. SkillRun and CACHE_VERSION come from the main application.)
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

# Application checkout root (holds data/, scripts/, cache/).
REPO = Path(os.environ.get("WTL_ROOT", "/opt/wheretolive"))
DB = REPO / "data" / "evaluations.db"

sys.path.insert(0, str(REPO / "scripts"))
from skill_report import SkillRun  # noqa: E402 — run ledger (ops.db skill_runs), from the main app

# Import CACHE_VERSION from the master config rather than hard-coding it,
# so a future scorer bump (s7 -> s8) automatically migrates retention too.
sys.path.insert(0, str(REPO))
from cache.config import CACHE_VERSION  # noqa: E402

DIM_TABLES = [
    "dim_demographics",
    "dim_safety",
    "dim_schools",
    "dim_commute",
    "dim_transit",
    "dim_price_analysis",
]

AUDIT_TABLES = [
    ("rm_sales_insert_audit", "inserted_at"),
    ("rm_sales_delete_audit", "deleted_at"),
]

AUDIT_KEEP_DAYS = 30
MAX_DELETE_FRACTION = 0.50  # refuse if a single DELETE would wipe > 50%


def cleanup_old_dim_versions(conn: sqlite3.Connection, current: str, dry_run: bool) -> tuple[int, int]:
    """Delete dim_* rows where data_version != current. Returns (would_delete, would_keep)."""
    total_deleted = 0
    total_kept = 0
    print(f"\n[dim retention] current version = {current}")
    for table in DIM_TABLES:
        total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        current_rows = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE data_version = ?",
            (current,),
        ).fetchone()[0]
        old = total - current_rows
        if old == 0:
            print(f"  {table}: clean (all {total} rows on {current})")
            total_kept += total
            continue
        frac = old / total
        if frac > MAX_DELETE_FRACTION:
            print(
                f"  {table}: REFUSING — would delete {old}/{total} "
                f"({frac*100:.0f}% > {MAX_DELETE_FRACTION*100:.0f}% guard). "
                f"Is CACHE_VERSION '{current}' correct?"
            )
            total_kept += total
            continue
        if dry_run:
            print(f"  {table}: would delete {old}/{total} ({frac*100:.0f}%)")
        else:
            conn.execute(f"DELETE FROM {table} WHERE data_version != ?", (current,))
            print(f"  {table}: deleted {old}, kept {current_rows}")
        total_deleted += old
        total_kept += current_rows
    return total_deleted, total_kept


def cleanup_old_audit(conn: sqlite3.Connection, days: int, dry_run: bool) -> tuple[int, int]:
    """Delete audit rows older than `days`. Returns (would_delete, would_keep)."""
    total_deleted = 0
    total_kept = 0
    print(f"\n[audit retention] keep last {days} days")
    for table, col in AUDIT_TABLES:
        total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        old = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {col} < datetime('now', ?)",
            (f"-{days} days",),
        ).fetchone()[0]
        if old == 0:
            print(f"  {table}: clean (all {total} rows within window)")
            total_kept += total
            continue
        # Audit retention is more permissive — old audit data IS expected to be wiped.
        # No sanity guard here.
        if dry_run:
            print(f"  {table}: would delete {old}/{total} ({old/total*100:.0f}%)")
        else:
            conn.execute(
                f"DELETE FROM {table} WHERE {col} < datetime('now', ?)",
                (f"-{days} days",),
            )
            print(f"  {table}: deleted {old}, kept {total - old}")
        total_deleted += old
        total_kept += total - old
    return total_deleted, total_kept


def vacuum(conn: sqlite3.Connection) -> float:
    """Run VACUUM. Returns elapsed seconds."""
    t0 = time.time()
    conn.execute("VACUUM")
    return time.time() - t0


CHAT_PURGE_DAYS = 7


def cleanup_deleted_chats(conn: sqlite3.Connection, days: int, dry_run: bool) -> tuple[int, int]:
    """Hard-delete chat conversations the user soft-deleted >`days` ago, plus
    their messages. Soft delete (deleted_at) hides a conversation immediately;
    we keep the real rows `days` more (recover-from-accident + audit window),
    then purge here. Returns (conversations_deleted, messages_deleted)."""
    print(f"\n[chat retention] purge soft-deleted conversations older than {days}d")
    try:
        ids = [r[0] for r in conn.execute(
            "SELECT conversation_id FROM chat_conversations "
            "WHERE deleted_at IS NOT NULL AND deleted_at < datetime('now', ?)",
            (f"-{days} days",),
        ).fetchall()]
    except sqlite3.OperationalError:
        print("  chat_conversations / deleted_at not present — skip")
        return 0, 0
    if not ids:
        print("  nothing to purge")
        return 0, 0
    # Count messages that will go (session_id == conversation_id).
    ph = ",".join("?" * len(ids))
    msg_n = conn.execute(
        f"SELECT COUNT(*) FROM chat_messages WHERE session_id IN ({ph})", ids
    ).fetchone()[0]
    if dry_run:
        print(f"  would purge {len(ids)} conversations + {msg_n} messages")
        return len(ids), msg_n
    conn.execute(f"DELETE FROM chat_messages WHERE session_id IN ({ph})", ids)
    conn.execute(f"DELETE FROM chat_conversations WHERE conversation_id IN ({ph})", ids)
    print(f"  purged {len(ids)} conversations + {msg_n} messages")
    return len(ids), msg_n


COMPILE_JOB_KEEP_DAYS = 30


def cleanup_radar_orphans(conn: sqlite3.Connection, dry_run: bool) -> tuple[int, int, int]:
    """Radar GC (added 2026-07-20 — 495/711 radar_match rows were orphans).

    DELETE /api/radars/[id] clears a radar's own matches, but historical
    delete paths left radar_match/radar_run rows pointing at radar ids that no
    longer exist (no FK enforcement). Orphans are dead by definition — nothing
    can ever read them — so no delete-fraction guard here. Also drops
    radar_compile_job rows older than COMPILE_JOB_KEEP_DAYS (the client only
    ever restores jobs <30 min old).
    Returns (match_deleted, run_deleted, compile_deleted)."""
    print("\n[radar retention] orphaned match/run rows + old compile jobs")
    deleted = []
    for table in ("radar_match", "radar_run"):
        try:
            old = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE radar_id NOT IN (SELECT id FROM radar)"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            print(f"  {table}: not present — skip")
            deleted.append(0)
            continue
        if old == 0:
            print(f"  {table}: clean (no orphans)")
        elif dry_run:
            print(f"  {table}: would delete {old} orphaned rows")
        else:
            conn.execute(f"DELETE FROM {table} WHERE radar_id NOT IN (SELECT id FROM radar)")
            print(f"  {table}: deleted {old} orphaned rows")
        deleted.append(old)
    try:
        old_jobs = conn.execute(
            "SELECT COUNT(*) FROM radar_compile_job WHERE created_at < datetime('now', ?)",
            (f"-{COMPILE_JOB_KEEP_DAYS} days",),
        ).fetchone()[0]
        if old_jobs == 0:
            print("  radar_compile_job: clean (all within window)")
        elif dry_run:
            print(f"  radar_compile_job: would delete {old_jobs} rows > {COMPILE_JOB_KEEP_DAYS}d")
        else:
            conn.execute(
                "DELETE FROM radar_compile_job WHERE created_at < datetime('now', ?)",
                (f"-{COMPILE_JOB_KEEP_DAYS} days",),
            )
            print(f"  radar_compile_job: deleted {old_jobs} rows > {COMPILE_JOB_KEEP_DAYS}d")
    except sqlite3.OperationalError:
        print("  radar_compile_job: not present — skip")
        old_jobs = 0
    return deleted[0], deleted[1], old_jobs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen, don't change DB")
    parser.add_argument("--vacuum", action="store_true", help="Run VACUUM after deletes")
    parser.add_argument("--keep-days", type=int, default=AUDIT_KEEP_DAYS)
    parser.add_argument("--parent", default=None,
                        help="Parent invocation_id when called by a workflow")
    parser.add_argument("--invoked-by", default="manual",
                        help="Source of invocation (e.g. nightly-maintenance)")
    args = parser.parse_args()

    if not DB.exists():
        print(f"ERROR: {DB} not found", file=sys.stderr)
        return 1

    with SkillRun("db-retention", invoked_by=args.invoked_by, parent=args.parent) as srun:
        srun.summary["params"] = {"dry_run": args.dry_run, "vacuum": args.vacuum, "keep_days": args.keep_days}
        conn = sqlite3.connect(str(DB))
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA journal_mode = WAL")

        size_before = DB.stat().st_size
        print(f"DB size before: {size_before / 1024 / 1024:.1f} MB")

        dim_del, dim_kept = cleanup_old_dim_versions(conn, CACHE_VERSION, args.dry_run)
        aud_del, aud_kept = cleanup_old_audit(conn, args.keep_days, args.dry_run)
        chat_conv_del, chat_msg_del = cleanup_deleted_chats(conn, CHAT_PURGE_DAYS, args.dry_run)
        radar_match_del, radar_run_del, radar_job_del = cleanup_radar_orphans(conn, args.dry_run)

        if not args.dry_run:
            conn.commit()

        if args.vacuum and not args.dry_run:
            # VACUUM rewrites the whole (~4.6GB) file under an EXCLUSIVE lock
            # (~70s, blocks the website + daemons). Only worth it when there's
            # real space to reclaim — i.e. we just deleted rows, or the freelist
            # is already a meaningful fraction of the file. In steady state
            # (no version bump) the freelist is ~0, so skip the pointless lock.
            free = conn.execute("PRAGMA freelist_count").fetchone()[0]
            total = conn.execute("PRAGMA page_count").fetchone()[0]
            frac = (free / total) if total else 0
            rows_deleted = dim_del + aud_del + chat_conv_del + chat_msg_del + radar_match_del + radar_run_del + radar_job_del
            if rows_deleted > 0 or frac > 0.05:
                print(f"\n[VACUUM] running (freelist {frac*100:.1f}%, {rows_deleted} rows deleted)...")
                elapsed = vacuum(conn)
                print(f"[VACUUM] done in {elapsed:.1f}s")
            else:
                print(f"\n[VACUUM] skipped — nothing to reclaim (freelist {frac*100:.1f}%, 0 rows deleted)")

        conn.close()

        size_after = DB.stat().st_size
        mb_before = size_before / 1024 / 1024
        mb_after = size_after / 1024 / 1024
        print(f"\nDB size after: {mb_after:.1f} MB (delta {mb_after - mb_before:+.1f} MB)")
        print(f"Deleted: {dim_del} dim rows + {aud_del} audit rows  "
              f"({'DRY RUN' if args.dry_run else 'COMMITTED'})")
        srun.summary["stats"] = {
            "dim_deleted": dim_del, "audit_deleted": aud_del,
            "radar_orphans_deleted": radar_match_del + radar_run_del,
            "radar_compile_jobs_deleted": radar_job_del,
            "mb_before": round(mb_before, 1), "mb_after": round(mb_after, 1),
        }
        srun.summary["final"] = (
            f"cleaned {dim_del} dim + {aud_del} audit rows | "
            f"{mb_before:.1f}→{mb_after:.1f} MB{' (DRY RUN)' if args.dry_run else ''}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
