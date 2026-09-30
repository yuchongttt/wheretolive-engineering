import importlib.util, time, calendar
from pathlib import Path

SPEC = Path(__file__).resolve().parents[1] / "backup_freshness_check.py"
_m = importlib.util.spec_from_file_location("bfc", SPEC)
bfc = importlib.util.module_from_spec(_m); _m.loader.exec_module(bfc)

def test_fresh_ok(tmp_path):
    mk = tmp_path / "marker"; mk.write_text("2026-07-01T05:20:00Z")
    now = calendar.timegm(time.strptime("2026-07-01T08:00:00Z", "%Y-%m-%dT%H:%M:%SZ"))
    assert bfc.check(now, str(mk), 26) is None

def test_stale_alerts(tmp_path):
    mk = tmp_path / "marker"; mk.write_text("2026-07-01T05:20:00Z")
    now = calendar.timegm(time.strptime("2026-07-02T08:00:00Z", "%Y-%m-%dT%H:%M:%SZ"))
    msg = bfc.check(now, str(mk), 26)
    assert msg and "stale" in msg.lower()

def test_missing_alerts(tmp_path):
    msg = bfc.check(1.0, str(tmp_path / "nope"), 26)
    assert msg and "missing" in msg.lower()

def test_main_survives_notify_failure(tmp_path, monkeypatch):
    mk = tmp_path / "marker"; mk.write_text("2020-01-01T00:00:00Z")  # very stale -> alerts
    monkeypatch.setenv("WTL_MARKER", str(mk))
    monkeypatch.setenv("WTL_NOTIFY_CMD", "/nonexistent/cmd_xyz_does_not_exist")
    assert bfc.main() == 1   # alerts, notify fails, still returns 1 without raising


def test_single_missed_backup_is_silent_by_default(tmp_path, monkeypatch):
    """One missed backup → marker ~28h old. Decision 2026-07-27: a single/transient
    failure must not alert; only persistent staleness does. The production default
    threshold must tolerate those 28h (i.e. be >28h), or the next morning's
    watchdog fires anyway."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 28 * 3600))
    mk = tmp_path / "marker"; mk.write_text(ts)
    monkeypatch.setenv("WTL_MARKER", str(mk))
    monkeypatch.delenv("WTL_BACKUP_MAX_AGE_H", raising=False)  # use the script's built-in default
    assert bfc.main() == 0   # silent


def test_two_missed_backups_alert_by_default(tmp_path, monkeypatch):
    """Two missed backups in a row → ~52h = a genuinely persistent problem → must
    alert. The default threshold must be <52h."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 52 * 3600))
    mk = tmp_path / "marker"; mk.write_text(ts)
    monkeypatch.setenv("WTL_MARKER", str(mk))
    monkeypatch.delenv("WTL_BACKUP_MAX_AGE_H", raising=False)
    monkeypatch.setenv("WTL_NOTIFY_CMD", "true")  # harmless, never hits real Telegram
    assert bfc.main() == 1   # alerts
