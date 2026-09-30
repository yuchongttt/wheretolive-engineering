"""wal_checkpoint's target DBs used to be a hard-coded three (problem found during
the 2026-09-07 disk audit): area_intel.db's WAL sat at 143 MB after the monthly
ingest on 2026-09-06 and was never reclaimed — it wasn't on that list, and the
website's long-lived read connections mean SQLite's own auto-checkpoint never
sees "the last connection closed". This is exactly the incident recorded in the
script docstring (WAL stuck at 469 MB → amplified lock contention → live
"database is locked"), recurring on a different DB.

Now: the three fixed DBs as before + automatically adopt any data/*.db whose WAL
has bloated past the threshold.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wal_checkpoint as w  # noqa: E402


def _mk(d, name, wal_bytes=0):
    """Create a .db (+ optional -wal). Content is irrelevant; size and pairing are under test."""
    (d / name).write_bytes(b"\0" * 1024)
    if wal_bytes:
        (d / (name + "-wal")).write_bytes(b"\0" * wal_bytes)
    return d / name


def test_adopts_db_whose_wal_bloated(tmp_path):
    """The area_intel.db case: not on the fixed list, but its WAL has bloated — must be adopted."""
    _mk(tmp_path, "area_intel.db", wal_bytes=w.WAL_ADOPT_BYTES + 1)
    assert [p.name for p in w.wal_bloated_dbs(tmp_path)] == ["area_intel.db"]


def test_ignores_small_wal(tmp_path):
    """Small WAL = healthy, don't open it — this runs every 5 min and must not connect to every DB."""
    _mk(tmp_path, "quiet.db", wal_bytes=1024)
    assert w.wal_bloated_dbs(tmp_path) == []


def test_ignores_wal_without_db(tmp_path):
    """An orphan -wal (DB already deleted) must not become a target."""
    (tmp_path / "ghost.db-wal").write_bytes(b"\0" * (w.WAL_ADOPT_BYTES + 1))
    assert w.wal_bloated_dbs(tmp_path) == []


def test_ignores_db_without_wal(tmp_path):
    _mk(tmp_path, "nowal.db")
    assert w.wal_bloated_dbs(tmp_path) == []


def test_targets_keeps_fixed_three_and_dedupes(tmp_path, monkeypatch):
    """The fixed three are always present (even with no WAL right now); adopted DBs must not duplicate them."""
    data = tmp_path / "data"
    data.mkdir()
    _mk(data, "evaluations.db", wal_bytes=w.WAL_ADOPT_BYTES + 1)  # fixed DB + bloated WAL
    _mk(data, "area_intel.db", wal_bytes=w.WAL_ADOPT_BYTES + 1)   # should be adopted
    monkeypatch.setattr(w, "ROOT", tmp_path)
    monkeypatch.setattr(w, "DBS", [data / "evaluations.db", data / "scrape_jobs.db",
                                   data / "egress_log.db"])

    names = [p.name for p in w.targets()]
    assert names[:3] == ["evaluations.db", "scrape_jobs.db", "egress_log.db"]
    assert names.count("evaluations.db") == 1
    assert "area_intel.db" in names
