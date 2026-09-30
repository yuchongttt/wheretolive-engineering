import importlib.util
import sqlite3
import re
from pathlib import Path

# Load backup_dbs.py as a module (this directory is not a package).
_SPEC = importlib.util.spec_from_file_location(
    "backup_dbs", Path(__file__).resolve().parents[1] / "backup_dbs.py")
bd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bd)


def _mkdb(path, rows=5):
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE t(x INTEGER)")
    con.executemany("INSERT INTO t VALUES(?)", [(i,) for i in range(rows)])
    con.commit(); con.close()


def _setenv(monkeypatch, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)


def test_snapshot_db_ok(tmp_path, monkeypatch):
    data = tmp_path / "data"; stage = tmp_path / "stage"
    data.mkdir(); stage.mkdir()
    _mkdb(data / "evaluations.db")
    _setenv(monkeypatch, {"WTL_DATA_DIR": str(data), "WTL_STAGING": str(stage)})
    assert bd.snapshot_db("evaluations") is True
    snap = stage / "evaluations.db"
    assert snap.exists()
    n = sqlite3.connect(str(snap)).execute("SELECT COUNT(*) FROM t").fetchone()[0]
    assert n == 5


def test_snapshot_db_corrupt_fails(tmp_path, monkeypatch):
    data = tmp_path / "data"; stage = tmp_path / "stage"
    data.mkdir(); stage.mkdir()
    (data / "evaluations.db").write_bytes(b"this is not a database" * 100)
    _setenv(monkeypatch, {"WTL_DATA_DIR": str(data), "WTL_STAGING": str(stage)})
    assert bd.snapshot_db("evaluations") is False


def test_snapshot_db_rejects_bad_name(tmp_path, monkeypatch):
    data = tmp_path / "data"; stage = tmp_path / "stage"
    data.mkdir(); stage.mkdir()
    _mkdb(data / "foo'bar.db")  # dangerous name
    _setenv(monkeypatch, {"WTL_DATA_DIR": str(data), "WTL_STAGING": str(stage)})
    assert bd.snapshot_db("foo'bar") is False


def test_classify_guard_warns_unknown(tmp_path, monkeypatch):
    data = tmp_path / "data"; data.mkdir()
    for n in ("evaluations.db", "scrape_jobs.db", "mystery.db"):
        _mkdb(data / n)
    sink = tmp_path / "notify.log"
    stub = tmp_path / "stub.sh"
    stub.write_text('#!/usr/bin/env bash\necho "$@" >> "%s"\n' % sink)
    stub.chmod(0o755)
    _setenv(monkeypatch, {"WTL_DATA_DIR": str(data), "WTL_NOTIFY_CMD": f"bash {stub}"})
    assert bd.classify_guard() is True
    logged = sink.read_text() if sink.exists() else ""
    assert "mystery" in logged
    assert "evaluations" not in logged and "scrape_jobs" not in logged


def _stage_two(tmp_path):
    stage = tmp_path / "stage"; stage.mkdir()
    _mkdb(stage / "evaluations.db"); _mkdb(stage / "sold.db")
    return stage


def test_commit_transfer_atomic_local(tmp_path, monkeypatch):
    stage = _stage_two(tmp_path)
    dest = tmp_path / "remote_backups"; dest.mkdir()
    _setenv(monkeypatch, {"WTL_STAGING": str(stage), "WTL_BACKUP_SSH": "",
                          "WTL_BACKUP_DEST": str(dest)})
    assert bd.commit_transfer("2026-07-01") is True
    day = dest / "daily" / "2026-07-01"
    assert (day / "evaluations.db").exists() and (day / "sold.db").exists()
    assert not (dest / "daily" / "2026-07-01.partial").exists()


def test_commit_transfer_failure_leaves_no_committed_dir(tmp_path, monkeypatch):
    stage = _stage_two(tmp_path)
    dest = tmp_path / "ro" / "backups"
    (tmp_path / "ro").mkdir(); (tmp_path / "ro").chmod(0o500)
    _setenv(monkeypatch, {"WTL_STAGING": str(stage), "WTL_BACKUP_SSH": "",
                          "WTL_BACKUP_DEST": str(dest)})
    ok = bd.commit_transfer("2026-07-01")
    (tmp_path / "ro").chmod(0o700)  # restore for cleanup
    assert ok is False
    assert not (dest / "daily" / "2026-07-01").exists()


def test_commit_transfer_rsync_failure_no_commit(tmp_path, monkeypatch):
    # dest writable (mkdir -p daily succeeds) but the rsync source is missing
    # -> rsync fails AFTER mkdir -> no committed daily/<date> dir.
    dest = tmp_path / "backups"; dest.mkdir()
    missing_stage = tmp_path / "does_not_exist"
    _setenv(monkeypatch, {"WTL_STAGING": str(missing_stage), "WTL_BACKUP_SSH": "",
                          "WTL_BACKUP_DEST": str(dest)})
    assert bd.commit_transfer("2026-07-01") is False
    assert not (dest / "daily" / "2026-07-01").exists()


def test_rotate_keeps_newest_daily(tmp_path, monkeypatch):
    base = tmp_path / "backups"; daily = base / "daily"; daily.mkdir(parents=True)
    for d in range(1, 11):  # 2026-07-01 .. 2026-07-10
        (daily / f"2026-07-{d:02d}").mkdir()
    (daily / "2026-07-11.partial").mkdir()  # stray partial
    _setenv(monkeypatch, {"WTL_BACKUP_SSH": "", "WTL_BACKUP_DEST": str(base),
                          "WTL_KEEP_DAILY": "7"})
    assert bd.rotate() is True
    kept = sorted(p.name for p in daily.iterdir())
    assert kept == [f"2026-07-{d:02d}" for d in range(4, 11)]  # 04..10 = 7 newest
    assert not any(p.name.endswith(".partial") for p in daily.iterdir())


def test_run_backup_end_to_end_local(tmp_path, monkeypatch):
    data = tmp_path / "data"; data.mkdir()
    for n in ("evaluations", "sold", "analytics"):
        _mkdb(data / f"{n}.db")
    dest = tmp_path / "backups"; dest.mkdir()
    marker = tmp_path / "marker"
    sink = tmp_path / "notify.log"
    stub = tmp_path / "notify_stub.sh"
    stub.write_text('#!/usr/bin/env bash\necho "$@" >> "%s"\n' % sink)
    stub.chmod(0o755)
    _setenv(monkeypatch, {
        "WTL_DATA_DIR": str(data), "WTL_STAGING": str(tmp_path / "stage"),
        "WTL_BACKUP_SSH": "", "WTL_BACKUP_DEST": str(dest),
        "WTL_MARKER": str(marker), "WTL_LOCKDIR": str(tmp_path / "lock.d"),
        "WTL_BACKUP_DATE": "2026-07-01", "WTL_FORCE_WEEKLY": "0",
        "WTL_ALLOW_OVERRIDE": "evaluations sold analytics",
        "WTL_NOTIFY_CMD": f"bash {stub}",
    })
    assert bd.run_backup() is True
    day = dest / "daily" / "2026-07-01"
    assert {p.name for p in day.iterdir()} == {"evaluations.db", "sold.db", "analytics.db"}
    assert re.match(r"\d{4}-\d\d-\d\dT", marker.read_text().strip())
    assert not (tmp_path / "stage").exists() or not any((tmp_path / "stage").iterdir())
    assert not sink.exists() or sink.read_text().strip() == ""  # silence = healthy


def test_run_backup_aborts_on_corrupt_no_marker(tmp_path, monkeypatch):
    data = tmp_path / "data"; data.mkdir()
    _mkdb(data / "evaluations.db")
    (data / "sold.db").write_bytes(b"garbage" * 200)  # corrupt
    dest = tmp_path / "backups"; dest.mkdir()
    marker = tmp_path / "marker"
    sink = tmp_path / "notify.log"
    stub = tmp_path / "notify_stub.sh"
    stub.write_text('#!/usr/bin/env bash\necho "$@" >> "%s"\n' % sink)
    stub.chmod(0o755)
    _setenv(monkeypatch, {
        "WTL_DATA_DIR": str(data), "WTL_STAGING": str(tmp_path / "stage"),
        "WTL_BACKUP_SSH": "", "WTL_BACKUP_DEST": str(dest),
        "WTL_MARKER": str(marker), "WTL_LOCKDIR": str(tmp_path / "lock.d"),
        "WTL_BACKUP_DATE": "2026-07-01", "WTL_FORCE_WEEKLY": "0",
        "WTL_ALLOW_OVERRIDE": "evaluations sold",
        "WTL_NOTIFY_CMD": f"bash {stub}",
    })
    assert bd.run_backup() is False
    assert not marker.exists()
    assert not (dest / "daily" / "2026-07-01").exists()
    # 2026-07-27: a single backup failure no longer sends TG (stderr instead; the
    # failure still shows up in STATUS_JSON + the watchdog). Silence = the notify
    # stub was never called. Persistent failure is caught by backup_freshness_check's
    # staleness watchdog.
    assert not sink.exists() or sink.read_text().strip() == ""


def test_run_backup_weekly_hardlink(tmp_path, monkeypatch):
    # Weeklies default OFF; opt in (FORCE_WEEKLY=1 + KEEP_WEEKLY=4) to exercise
    # the weekly-hardlink code path.
    data = tmp_path / "data"; data.mkdir()
    _mkdb(data / "evaluations.db")
    dest = tmp_path / "backups"; dest.mkdir()
    _setenv(monkeypatch, {
        "WTL_DATA_DIR": str(data), "WTL_STAGING": str(tmp_path / "stage"),
        "WTL_BACKUP_SSH": "", "WTL_BACKUP_DEST": str(dest),
        "WTL_MARKER": str(tmp_path / "marker"), "WTL_LOCKDIR": str(tmp_path / "lock.d"),
        "WTL_BACKUP_DATE": "2026-07-01", "WTL_FORCE_WEEKLY": "1", "WTL_KEEP_WEEKLY": "4",
        "WTL_ALLOW_OVERRIDE": "evaluations",
        "WTL_NOTIFY_CMD": "true",
    })
    assert bd.run_backup() is True
    weeklies = list((dest / "weekly").glob("*/evaluations.db"))
    assert len(weeklies) == 1  # hardlinked copy into weekly/<ISOWEEK>
