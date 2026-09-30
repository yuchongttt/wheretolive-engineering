"""Outward-reachability rule — the 2026-08-27 shape.

That day the Mac stayed powered on but every outbound IPv4 send returned
ENETUNREACH for 33h32m (cloudflared: `sendmsg: network is unreachable`; the
hourly ingestion job failed identically). DNS went down with it: MagicDNS is the
primary resolver (unscoped, order 101000) and it can only answer NXDOMAIN with
no route upstream — the router's own resolver is present too, but scoped to en1
and equally cut off, so no resolver choice would have helped. Nothing on the box
noticed; it took a manual reboot. Googlebot crawled straight into Cloudflare
530s, which is what Search Console later reported as "Server error (5xx)".

These tests pin the two things that must hold: the detector must tell "our
tunnel broke" apart from "the host has no network" (different repairs), and it
must never act on a single tick — a routine cloudflared restart shows zero
connections for a second or two.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timedelta, timezone  # noqa: E402
import watchdog as W  # noqa: E402
import watchdog_ledger as L  # noqa: E402
import watchdog_rules as R  # noqa: E402


def _now():
    return datetime(2026, 9, 7, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------- detector

def test_parse_ha_connections_reads_the_gauge():
    text = ("# HELP cloudflared_tunnel_ha_connections Number of active ha connections\n"
            "# TYPE cloudflared_tunnel_ha_connections gauge\n"
            "cloudflared_tunnel_ha_connections 4\n"
            "cloudflared_tunnel_total_requests 142133\n")
    assert R.parse_ha_connections(text) == 4
    assert R.parse_ha_connections(text.replace(" 4", " 0")) == 0


def test_parse_ha_connections_none_when_unreadable():
    # Metrics port dead (cloudflared not running) or the gauge is gone.
    assert R.parse_ha_connections(None) is None
    assert R.parse_ha_connections("cloudflared_tunnel_total_requests 1\n") is None


def test_net_fault_healthy_when_tunnel_has_connections():
    assert R.net_fault(4, raw_internet_ok=True) is None
    # Tunnel up but our raw-IP probe failed: the probe is the doubtful one, not
    # the tunnel. Never act on that alone.
    assert R.net_fault(4, raw_internet_ok=False) is None


def test_net_fault_separates_cloudflared_from_host_network():
    # Tunnel down, host can still reach the public internet by raw IP.
    assert R.net_fault(0, raw_internet_ok=True) == "cloudflared"
    # 2026-08-27: tunnel down AND no egress at all. Restarting cloudflared
    # cannot fix a kernel with no route.
    assert R.net_fault(0, raw_internet_ok=False) == "host-network"
    # Metrics port unreachable counts as tunnel down.
    assert R.net_fault(None, raw_internet_ok=False) == "host-network"


def test_net_confirmed_needs_a_recent_prior_detection():
    now = _now()
    assert R.net_confirmed(None, now) is False                       # first sighting
    fresh = {"action": "detect", "ts": (now - timedelta(minutes=2)).isoformat()}
    assert R.net_confirmed(fresh, now) is True                       # second consecutive tick
    stale = {"action": "detect", "ts": (now - timedelta(days=2)).isoformat()}
    assert R.net_confirmed(stale, now) is False                      # old episode, start over
    closed = {"action": "recover", "ts": (now - timedelta(minutes=2)).isoformat()}
    assert R.net_confirmed(closed, now) is False                     # last episode ended


# ---------------------------------------------------------------- loop wiring

def _deps(**over):
    sent = []
    d = dict(
        # every launchd job healthy so only the net rule can fire
        job_state=lambda label: "-\t0\tx",
        job_fresh=lambda label: True,
        kickstart=lambda label: sent.append(("kickstart", label)) or 0,
        hetzner_ok=lambda: True,
        restart_worker=lambda: 0,
        send=lambda text: sent.append(("tg", text)),
        post_heartbeat=lambda: sent.append(("hb", None)),
        log_tail=lambda label: "",
        ha_connections=lambda: 0,
        raw_internet_ok=lambda: False,
        repair_net=lambda fault: sent.append(("repair", fault)) or 0,
        net_diag=lambda: "default=router@en1 gw=down wifi=on",
    )
    d.update(over)
    return d, sent


def test_first_tick_only_records_never_touches_the_network(tmp_path):
    conn = L.open_db(str(tmp_path / "ops.db"))
    deps, sent = _deps()
    W.run_once(conn, "enforce", _now(), deps)
    assert not any(a == "repair" for a, _ in sent)
    assert L.last_open_problem(conn, "net-wedged", "tunnel")["action"] == "detect"


def test_second_consecutive_tick_repairs_the_host_network(tmp_path):
    conn = L.open_db(str(tmp_path / "ops.db"))
    deps, sent = _deps()
    W.run_once(conn, "enforce", _now(), deps)
    W.run_once(conn, "enforce", _now() + timedelta(minutes=2), deps)
    assert ("repair", "host-network") in sent
    assert L.last_open_problem(conn, "net-wedged", "tunnel")["action"] == "fix"


def test_tunnel_down_but_internet_up_repairs_cloudflared_only(tmp_path):
    conn = L.open_db(str(tmp_path / "ops.db"))
    deps, sent = _deps(raw_internet_ok=lambda: True)
    W.run_once(conn, "enforce", _now(), deps)
    W.run_once(conn, "enforce", _now() + timedelta(minutes=2), deps)
    assert ("repair", "cloudflared") in sent


def test_observe_mode_never_executes_the_repair(tmp_path):
    conn = L.open_db(str(tmp_path / "ops.db"))
    deps, sent = _deps()
    W.run_once(conn, "observe", _now(), deps)
    W.run_once(conn, "observe", _now() + timedelta(minutes=2), deps)
    assert not any(a == "repair" for a, _ in sent)


def test_recovery_is_recorded_once_the_tunnel_is_back(tmp_path):
    conn = L.open_db(str(tmp_path / "ops.db"))
    deps, sent = _deps()
    W.run_once(conn, "enforce", _now(), deps)
    W.run_once(conn, "enforce", _now() + timedelta(minutes=2), deps)      # fix
    healthy, _ = _deps(ha_connections=lambda: 4, raw_internet_ok=lambda: True)
    W.run_once(conn, "enforce", _now() + timedelta(minutes=4), healthy)
    assert L.last_open_problem(conn, "net-wedged", "tunnel")["action"] == "recover"


def test_healthy_host_never_records_anything(tmp_path):
    conn = L.open_db(str(tmp_path / "ops.db"))
    deps, sent = _deps(ha_connections=lambda: 4, raw_internet_ok=lambda: True)
    W.run_once(conn, "enforce", _now(), deps)
    assert L.last_open_problem(conn, "net-wedged", "tunnel") is None
    assert not any(a == "repair" for a, _ in sent)


# ---------------------------------------------------------------- repair safety

class _FakeRun:
    """Stands in for subprocess.run. Records argv, can be told the radio's
    current power state, and can blow up on a chosen sub-command."""

    def __init__(self, power="On", raise_on=None):
        self.calls, self.power, self.raise_on = [], power, raise_on

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        if self.raise_on and self.raise_on in argv:
            raise RuntimeError("boom")

        class _R:
            returncode = 0
            stdout = f"Wi-Fi Power (en1): {self.power}\n"
        return _R()

    @property
    def power_ops(self):
        return [a[-1] for a in self.calls if "-setairportpower" in a]


def test_bounce_wifi_cycles_the_radio_when_it_is_on():
    run = _FakeRun(power="On")
    assert W._bounce_wifi(run=run, sleep=lambda s: None) == 0
    assert run.power_ops == ["off", "on"]


def test_bounce_wifi_always_ends_with_the_radio_on():
    # If anything blows up mid-cycle the radio must still come back. Leaving it
    # off strands the box in the very outage this repair exists to end, and once
    # the repair budget is spent nothing would come back to switch it on.
    run = _FakeRun(power="On", raise_on="off")
    W._bounce_wifi(run=run, sleep=lambda s: None)
    assert run.power_ops[-1] == "on"


def test_bounce_wifi_just_switches_on_when_the_radio_is_already_off():
    # Previous tick was killed between off and on: don't cycle again, recover.
    run = _FakeRun(power="Off")
    W._bounce_wifi(run=run, sleep=lambda s: None)
    assert run.power_ops == ["on"]


# ---------------------------------------------------------------- diagnostics

def test_fault_note_carries_the_network_snapshot(tmp_path):
    # 2026-08-27 is unreconstructable because nothing wrote down the routing
    # state while it was happening. Every fault row must carry that snapshot.
    conn = L.open_db(str(tmp_path / "ops.db"))
    deps, _ = _deps(net_diag=lambda: "default=router@en1 gw=down wifi=on")
    W.run_once(conn, "enforce", _now(), deps)
    assert "gw=down" in conn.execute(
        "SELECT note FROM watchdog_actions WHERE rule_id='net-wedged'").fetchone()[0]


def test_detect_only_episode_is_closed_so_the_two_tick_gate_still_applies(tmp_path):
    conn = L.open_db(str(tmp_path / "ops.db"))
    deps, sent = _deps()
    W.run_once(conn, "enforce", _now(), deps)                                  # detect
    healthy, _ = _deps(ha_connections=lambda: 4, raw_internet_ok=lambda: True)
    W.run_once(conn, "enforce", _now() + timedelta(minutes=2), healthy)        # blip cleared
    assert L.last_open_problem(conn, "net-wedged", "tunnel")["action"] == "recover"
    # A fresh fault inside the confirm window must start the gate over, not
    # inherit the stale detection and repair on its first tick.
    W.run_once(conn, "enforce", _now() + timedelta(minutes=4), deps)
    assert not any(a == "repair" for a, _ in sent)
