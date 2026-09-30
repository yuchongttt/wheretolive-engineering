#!/usr/bin/env python3
"""backup_dbs.py — verified daily SQLite backup, Mac → gpu-box (~/wtl-backups, SSD root).

WHY PYTHON (and not the old backup_dbs.sh)
------------------------------------------
This repo lives under ~/Documents, which is TCC-protected on modern macOS. A
launchd job's access is decided per-*binary*: only binaries granted "Full Disk
Access" / "Documents folder" (System Settings › Privacy) may touch ~/Documents.
`/bin/bash` and the `sqlite3` CLI are NOT granted, so the previous plain-bash
job (backup_dbs.sh) failed under launchd with exit 126 / "authorization denied"
and only ever succeeded when run by hand from an FDA terminal. The Homebrew
`python3` this repo's venv points at IS granted (it's how every other launchd
job here reaches ~/Documents). So all ~/Documents I/O — enumerating data/,
VACUUM-reading the .db files, writing the marker/status — is done here, in this
Python process, via the sqlite3 module (in-process, not a spawned CLI). Pure
non-Documents work (rsync to gpu-box, remote ssh mkdir/mv/prune) is still
shelled out, byte-for-byte the same commands the bash version used.

Run via the standard launchd vehicle, exactly like the other jobs:
    wtl-pyrun.sh  venv/bin/python3  backup_dbs.py

Env-overridable knobs (used by tests) keep the same names as the old script.
"""
import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---- config (env-overridable for tests) ---------------------------------
HOME = os.environ.get("HOME", str(Path.home()))
# Application checkout root (holds data/, logs/, venv/); override with WTL_ROOT.
DEFAULT_ROOT = "/opt/wheretolive"
HERE = Path(__file__).resolve().parent


def _colon(name: str, default: str) -> str:
    """bash ${VAR:=default} — default when unset OR empty."""
    return os.environ.get(name) or default


def _plain(name: str, default: str) -> str:
    """bash ${VAR=default} — default only when unset; explicit '' is kept."""
    return os.environ.get(name, default)


def ROOT() -> str:        return _colon("WTL_ROOT", DEFAULT_ROOT)
def DATA_DIR() -> Path:   return Path(_colon("WTL_DATA_DIR", f"{ROOT()}/data"))
def STAGING() -> Path:    return Path(_colon("WTL_STAGING", f"{HOME}/.wtl-backup-staging"))
def BACKUP_SSH() -> str:  return _plain("WTL_BACKUP_SSH", "ssh -o BatchMode=yes -o ConnectTimeout=15 gpu-box")
def BACKUP_DEST() -> str: return _colon("WTL_BACKUP_DEST", "gpu-box:wtl-backups")
def RSYNC_TIMEOUT() -> str: return _colon("WTL_RSYNC_TIMEOUT", "600")
def KEEP_DAILY() -> str:  return _colon("WTL_KEEP_DAILY", "3")
def KEEP_WEEKLY() -> str: return _colon("WTL_KEEP_WEEKLY", "0")
def MARKER() -> Path:     return Path(_colon("WTL_MARKER", f"{ROOT()}/logs/db-backup-last-success"))
def LOCKDIR() -> Path:    return Path(_colon("WTL_LOCKDIR", f"{HOME}/.wtl-backup.lock.d"))
def LOCK_STALE_MIN() -> int: return int(_colon("WTL_LOCK_STALE_MIN", "360"))
def STATUS_JSON() -> Path: return Path(_colon("WTL_STATUS_JSON", str(MARKER().parent / "db-backup-status.json")))
def NOTIFY_CMD() -> str:  return _colon("WTL_NOTIFY_CMD", f"{ROOT()}/venv/bin/python3 {HERE}/notify_telegram.py")

ALLOW_LIST = ["evaluations", "leases", "sold", "ops", "planning_apps",
              "annotations", "monitors", "analytics", "security",
              # 2026-09-07: all three had been unclassified since they were created,
              # so they nagged daily via warn_unclassified() AND were never backed up.
              # All three hold user-facing state that cannot be regenerated:
              # bug_reports = reports users typed at us; newsletter = the subscriber
              # list (losing a signup would be silent);
              # reports = token -> generated report snapshot, and those tokens are
              # live URLs we handed out. Tiny (~0.9 MB combined).
              "bug_reports", "newsletter", "reports"]
DENY_LIST = ["scrape_jobs", "egress_log", "detail_staging", "host_metrics",
             "image_download", "embedding_worker", "gpu_tasks",
             "images", "chat_telemetry", "backup_nonres_20260615", "image_cache",
             "area_intel", "journey_cache"]


def _allow():
    ov = os.environ.get("WTL_ALLOW_OVERRIDE")
    return ov.split() if ov else list(ALLOW_LIST)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def notify(msg: str) -> None:
    try:
        subprocess.run(shlex.split(NOTIFY_CMD()) + [msg], check=False)
    except Exception:
        pass  # notify is best-effort — never affect the backup outcome


# ---- snapshot -----------------------------------------------------------
def snapshot_db(name: str) -> bool:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        print(f"snapshot_db: invalid name '{name}'", file=sys.stderr)
        return False
    src = DATA_DIR() / f"{name}.db"
    dst = STAGING() / f"{name}.db"
    if not src.is_file():
        print(f"snapshot_db: missing {src}", file=sys.stderr)
        return False
    try:
        dst.unlink()
    except FileNotFoundError:
        pass
    # VACUUM INTO — consistent, WAL-safe snapshot. Path is inlined (VACUUM takes
    # no bind params); single-quotes are doubled. name is regex-validated and
    # dst lives in our own staging dir, so this is safe.
    dst_sql = str(dst).replace("'", "''")
    try:
        con = sqlite3.connect(str(src))
        try:
            con.execute(f"VACUUM INTO '{dst_sql}'")
        finally:
            con.close()
    except sqlite3.Error as e:
        print(f"snapshot_db: VACUUM failed {name}: {e}", file=sys.stderr)
        return False
    try:
        con = sqlite3.connect(str(dst))
        try:
            ic = con.execute("PRAGMA integrity_check").fetchone()[0]
        finally:
            con.close()
    except sqlite3.Error:
        ic = "error"
    if ic != "ok":
        print(f"snapshot_db: integrity_check {name} -> {ic}", file=sys.stderr)
        return False
    return True


def classify_guard() -> bool:
    """Warn (never fail) if a data/ DB is in neither ALLOW nor DENY — so a new
    DB can't silently escape the backup set."""
    for f in sorted(DATA_DIR().glob("*.db")):
        name = f.stem
        if name not in ALLOW_LIST and name not in DENY_LIST:
            print(f"UNCLASSIFIED: {name}", file=sys.stderr)
            notify(f"backup: unclassified DB '{name}' — add to ALLOW or DENY in backup_dbs.py")
    return True


# ---- transfer / rotate (non-Documents; shelled out like the bash version) --
def remote(cmd: str) -> bool:
    """Run a shell snippet on the backup host (or locally when WTL_BACKUP_SSH='')."""
    ssh = BACKUP_SSH()
    if not ssh:
        r = subprocess.run(["bash", "-c", cmd])
    else:
        r = subprocess.run(shlex.split(ssh) + [cmd])
    return r.returncode == 0


def _base() -> str:
    """Strip a 'host:' prefix from WTL_BACKUP_DEST -> path used in remote() cmds."""
    dest = BACKUP_DEST()
    return dest.split(":", 1)[1] if ":" in dest else dest


def commit_transfer(date: str) -> bool:
    base = _base()
    dest = BACKUP_DEST()
    staging = STAGING()
    if not remote(f"mkdir -p '{base}/daily'"):
        return False
    if not remote(f"rm -rf '{base}/daily/{date}.partial'"):
        return False
    partial = f"{dest}/daily/{date}.partial/"
    r = subprocess.run(["rsync", "-a", f"--timeout={RSYNC_TIMEOUT()}",
                        f"{staging}/", partial])
    if r.returncode != 0:
        print("commit_transfer: rsync failed", file=sys.stderr)
        return False
    verify = subprocess.run(["rsync", "-a", "--checksum", "--dry-run",
                             "--itemize-changes", f"{staging}/", partial],
                            capture_output=True, text=True)
    diff = "\n".join(ln for ln in verify.stdout.splitlines()
                     if ln[:1] in "<>ch")
    if diff:
        print(f"commit_transfer: checksum mismatch:\n{diff}", file=sys.stderr)
        return False
    if not remote(f"rm -rf '{base}/daily/{date}' && "
                  f"mv '{base}/daily/{date}.partial' '{base}/daily/{date}'"):
        return False
    return True


def _prune_dir(parent: str, keep: str) -> bool:
    # BSD-safe (no `head -n -N`): sort oldest-first, delete the first (total-keep).
    # Runs on the remote (or locally) so it works over ssh identically.
    snippet = (
        f"cd '{parent}' 2>/dev/null || exit 0; "
        f"rm -rf ./*.partial 2>/dev/null || true; "
        f"n=$(ls -1d */ 2>/dev/null | wc -l | tr -d ' '); keep={keep}; rm=$((n - keep)); "
        f"if [ \"$rm\" -gt 0 ]; then "
        f"ls -1d */ 2>/dev/null | sed 's:/$::' | sort | head -n \"$rm\" | "
        f"while read -r d; do rm -rf \"$d\"; done; fi"
    )
    return remote(snippet)


def rotate() -> bool:
    base = _base()
    if not _prune_dir(f"{base}/daily", KEEP_DAILY()):
        return False
    if not _prune_dir(f"{base}/weekly", KEEP_WEEKLY()):
        return False
    return True


# ---- orchestration ------------------------------------------------------
def _acquire_lock() -> bool:
    """mkdir is atomic on POSIX -> cross-process lock (macOS has no flock).
    Returns True if we own the lock (caller must clean it up)."""
    lock = LOCKDIR()
    try:
        lock.mkdir()
        return True
    except FileExistsError:
        pass
    # Held. Break it only if stale (older than LOCK_STALE_MIN minutes).
    try:
        age_min = (time.time() - lock.stat().st_mtime) / 60.0
    except FileNotFoundError:
        age_min = 1e9
    if age_min > LOCK_STALE_MIN():
        print(f"run_backup: breaking stale lock {lock}", file=sys.stderr)
        shutil.rmtree(lock, ignore_errors=True)
        try:
            lock.mkdir()
            return True
        except OSError:
            print("run_backup: cannot acquire lock; skipping", file=sys.stderr)
            return False
    print(f"run_backup: another run holds the lock ({lock}); skipping", file=sys.stderr)
    return False


def run_backup() -> bool:
    if not _acquire_lock():
        return True  # a live run holds the lock; skipping is success (like bash)
    try:
        STAGING().mkdir(parents=True, exist_ok=True)
        date = _colon("WTL_BACKUP_DATE", datetime.now(timezone.utc).strftime("%Y-%m-%d"))

        for db in _allow():
            if not snapshot_db(db):
                # 2026-07-27: a single failure sends no TG (already recorded in stderr +
                # STATUS_JSON; persistent failure is caught by backup_freshness_check's
                # staleness watchdog, which only alerts on consecutive misses).
                print(f"run_backup FAILED: snapshot/integrity of {db}", file=sys.stderr)
                return False
        classify_guard()
        if not commit_transfer(date):
            print(f"run_backup FAILED: transfer/verify for {date}", file=sys.stderr)
            return False

        force = os.environ.get("WTL_FORCE_WEEKLY", "0")
        do_weekly = force == "1" or (
            force == "auto" and datetime.now(timezone.utc).isoweekday() == 7)
        if do_weekly:
            base = _base()
            week = datetime.now(timezone.utc).strftime("%G-W%V")
            if not remote(f"mkdir -p '{base}/weekly/{week}' && "
                          f"cp -l '{base}/daily/{date}/'* '{base}/weekly/{week}/'"):
                print(f"run_backup FAILED: weekly copy {week}", file=sys.stderr)
                return False

        if not rotate():
            print(f"run_backup FAILED: rotation for {date}", file=sys.stderr)
            return False

        MARKER().parent.mkdir(parents=True, exist_ok=True)
        MARKER().write_text(_now_iso() + "\n")
        print(f"run_backup: OK {date}")
        return True
    finally:
        shutil.rmtree(STAGING(), ignore_errors=True)
        shutil.rmtree(LOCKDIR(), ignore_errors=True)


def main() -> int:
    ok = run_backup()
    # status breadcrumb for the admin panel (best-effort; never affects exit code)
    try:
        STATUS_JSON().write_text(json.dumps({
            "last_run": _now_iso(),
            "result": "ok" if ok else "failed",
            "target": BACKUP_DEST(),
            "keep_daily": int(KEEP_DAILY()),
        }) + "\n")
    except Exception:
        pass
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
