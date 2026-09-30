import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import heartbeat_check as H  # noqa: E402

FRESH = {"mac-mini": 30, "mac-watchdog": 30, "windows-3090": 30, "gpu-box": 30}


def test_all_fresh_no_alerts():
    alerts, down = H.evaluate(FRESH, set())
    assert alerts == [] and down == set()


def test_windows_stale_alerts_down_once():
    ages = dict(FRESH, **{"windows-3090": 2000})
    alerts, down = H.evaluate(ages, set())
    assert ("windows-3090", "down", "3090 WSL") in alerts
    assert "windows-3090" in down
    # already-flagged -> no repeat alert (dedup)
    alerts2, _ = H.evaluate(ages, down)
    assert alerts2 == []


def test_recovery_emits_up():
    down_prev = {"windows-3090"}
    alerts, down = H.evaluate(FRESH, down_prev)
    assert ("windows-3090", "up", "3090 WSL") in alerts
    assert down == set()


def test_missing_host_counts_as_down():
    ages = {k: v for k, v in FRESH.items() if k != "mac-watchdog"}
    alerts, down = H.evaluate(ages, set())
    assert "mac-watchdog" in down


def test_full_mac_down_suppresses_watchdog_double_ping():
    ages = dict(FRESH, **{"mac-mini": 2000, "mac-watchdog": 2000})
    alerts, down = H.evaluate(ages, set())
    ids = {hid for hid, _, _ in alerts}
    assert "mac-mini" in ids and "mac-watchdog" not in ids  # only the Mac alert, not both
    assert "mac-watchdog" not in down


# ── Mac unreachable: the one thing this checker must report, and previously couldn't ──
# What happened on 2026-08-27: the Mac's DNS/Tailscale was down for 33.5 hours;
# this script exited 1 every 5 minutes with `URLError: <urlopen error timed out>`
# and sent not a single alert. Reason: its data is read *from the Mac* (the
# host-metrics API runs on the Mac), so when the Mac died, it died with it.
# The TG credentials are local to gpu-box and its network was fine — the only
# thing missing was treating "cannot read" as an alert in its own right.

def test_api_unreachable_alerts_mac_down():
    alerts, down = H.evaluate({}, set(), api_ok=False)
    ids = {hid for hid, kind, _ in alerts if kind == "down"}
    assert ids == {"mac-mini"}, f"unreachable host-metrics means the Mac is down; got {ids}"
    assert "mac-mini" in down


def test_api_unreachable_does_not_invent_other_hosts_down():
    # Other hosts' state is read *through the Mac*: unreadable = unknown, not down.
    # Reporting "gpu-box unreachable" from gpu-box itself would be especially absurd.
    _alerts, down = H.evaluate({}, set(), api_ok=False)
    assert down == {"mac-mini"}, f"unknown must not be treated as down; got {down}"


def test_api_unreachable_carries_over_previously_down_hosts():
    # 3090 was already down before the outage → no "✅ recovered" while the Mac is unreachable.
    alerts, down = H.evaluate({}, {"windows-3090"}, api_ok=False)
    assert ("windows-3090", "up", "3090 WSL") not in alerts
    assert down == {"mac-mini", "windows-3090"}


def test_api_unreachable_dedups_like_the_normal_path():
    _a1, down = H.evaluate({}, set(), api_ok=False)
    alerts2, _d2 = H.evaluate({}, down, api_ok=False)
    assert alerts2 == [], f"a sustained outage must not re-alert every 5 minutes; got {alerts2}"


def test_mac_down_never_emits_a_watchdog_recovery():
    # Watchdog dies first, then the whole Mac goes dark: the old logic discarded
    # the watchdog from `down` and immediately sent "✅ Mac watchdog reporting
    # again" — announcing a recovery in the middle of an outage.
    ages = dict(FRESH, **{"mac-mini": 2000, "mac-watchdog": 2000})
    alerts, _down = H.evaluate(ages, {"mac-watchdog"})
    assert ("mac-watchdog", "up", "Mac watchdog (self-healing)") not in alerts, \
        f"must not report a watchdog recovery while the Mac is down; got {alerts}"


def test_main_survives_an_unreachable_api_and_sends_the_alert(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "STATE_FILE", tmp_path / "state.json")
    sent = []

    def boom():
        raise OSError("timed out")

    rc = H.main(fetch=boom, tg=sent.append)
    assert rc == 0, "an unreachable host-metrics API must still finish cleanly (old behaviour: exit 1)"
    assert any("unreachable" in m for m in sent), f"expected a Mac-down alert; got {sent}"
