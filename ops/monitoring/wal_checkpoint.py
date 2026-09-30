#!/usr/bin/env python3
"""PRAGMA wal_checkpoint(PASSIVE) on every Mac SQLite DB.

Scheduled every 5 min via xyz.wheretolive.wal-checkpoint.plist.

Was originally a bash script but launchd-spawned /bin/bash + /usr/bin/sqlite3
hit macOS TCC restrictions on ~/Documents access. Python via the project
venv works because that interpreter is already cleared (same pattern the
other dispatchers use).

PASSIVE mode: doesn't take exclusive lock, won't fail under load —
opportunistically writes whatever frames it can. Long-running readers
(valuation ML, epc_enrich) keep their snapshot; their frames stay in
WAL until they finish. But without this, WAL grew to 3.7 GB in a day.

PASSIVE alone never shrinks the WAL *file* — it stays at its high-water mark
(seen 2026-05-31: evaluations.db-wal stuck at 469 MB with only 4 live frames,
amplifying lock contention → "database is locked" on the live chat path). So
when the WAL file exceeds WAL_TRUNCATE_BYTES we additionally attempt a
TRUNCATE checkpoint to reclaim the file. TRUNCATE returns busy (no-op) if a
reader still holds an old snapshot — with timeout=5 it won't hang, so it's
safe under load and simply succeeds during the next quiet moment.
"""
import os
import sqlite3
import sys
import time
from pathlib import Path

WAL_TRUNCATE_BYTES = 64 * 1024 * 1024  # reclaim the file once it bloats past 64 MB
WAL_ADOPT_BYTES = 8 * 1024 * 1024      # any other data/*.db this bloated gets picked up too

# Application checkout root (holds data/, logs/).
ROOT = Path(os.environ.get("WTL_ROOT", "/opt/wheretolive"))
DBS = [
    ROOT / "data" / "evaluations.db",
    ROOT / "data" / "scrape_jobs.db",
    ROOT / "data" / "egress_log.db",
]
LOG = ROOT / "logs" / "wal-checkpoint.log"


def wal_bloated_dbs(data_dir: Path, min_bytes: int = WAL_ADOPT_BYTES) -> list[Path]:
    """data/*.db whose -wal has grown past min_bytes.

    The hardcoded DBS list above only ever named the three noisy writers, so any
    *other* DB that acquires a long-lived reader keeps its WAL forever: SQLite's
    auto-checkpoint only fires when the last connection closes, and the website
    never closes. Found 2026-09-07 — area_intel.db sat at a 143 MB WAL after the
    monthly crime ingest, on a DB the live /area + chat get_area_profile read.

    Size-gated on purpose: this runs every 5 min, so it must not open every DB in
    data/ just to find nothing. A small WAL is a healthy WAL — auto-checkpoint is
    coping — and only the bloated ones need the nudge.
    """
    out = []
    for wal in sorted(data_dir.glob("*.db-wal")):
        db = wal.with_name(wal.name[:-len("-wal")])
        if not db.exists():
            continue  # orphan WAL, DB already gone
        try:
            if wal.stat().st_size >= min_bytes:
                out.append(db)
        except OSError:
            continue
    return out


def targets() -> list[Path]:
    """The three fixed DBs (always, even with no WAL) plus adopted bloated ones."""
    fixed = {db.name for db in DBS}
    return DBS + [db for db in wal_bloated_dbs(ROOT / "data") if db.name not in fixed]


def checkpoint_one(db: Path) -> str:
    if not db.exists():
        return f"{db.name}: missing"
    try:
        conn = sqlite3.connect(str(db), timeout=5)
        # 0|N|M  → busy_flag | checkpointed_frames | total_frames_in_wal
        result = conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
        wal = db.with_suffix(".db-wal")
        wal_size = wal.stat().st_size if wal.exists() else 0
        trunc = ""
        if wal_size > WAL_TRUNCATE_BYTES:
            # File has bloated — try to reclaim it. Busy-tolerant, won't hang.
            tr = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            after = wal.stat().st_size if wal.exists() else 0
            trunc = f" TRUNC(busy={tr[0]} {wal_size//1048576}->{after//1048576}MB)"
        conn.close()
    except Exception as e:
        return f"{db.name}: err {e}"
    return f"{db.name}: busy={result[0]} done={result[1]}/{result[2]} wal={wal_size}B{trunc}"


def main() -> int:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        for db in targets():
            f.write(f"{ts}  {checkpoint_one(db)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
