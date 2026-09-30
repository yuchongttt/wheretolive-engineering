#!/usr/bin/env python3
"""Self-healing watchdog loop. Runs one tick per launchd invocation.

Verify-after-act: a fix is only reported recovered once the NEXT tick confirms
health. Ships in observe mode (logs intended actions, executes nothing) until
WATCHDOG_MODE=enforce."""
import os
import socket
import sys
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import watchdog_ledger as L
import watchdog_rules as R
import watchdog_notify as N

# ssh target of the external Hetzner VPS that runs the wtl-worker systemd unit.
HETZNER = os.environ.get("WATCHDOG_REMOTE_HOST", "root@hetzner-box")

# cloudflared's own metrics page. 127.0.0.1 on purpose: it needs no DNS, and DNS
# is one of the casualties of the failure this rule exists for.
CF_METRICS_URL = "http://127.0.0.1:20241/metrics"
# Egress probes by literal IP, two independent operators. The question is "does
# this box have a route out", so one reachable host is enough.
EGRESS_PROBES = [("1.1.1.1", 443), ("9.9.9.9", 443)]
# en1 is the only live uplink — en0 (Ethernet) has no cable.
WIFI_IF = os.environ.get("WATCHDOG_WIFI_IF", "en1")
NET_REASON = {
    "cloudflared": "tunnel down (host network is fine)",
    "host-network": "host has lost its whole route to the internet",
}


def run_once(conn, mode, now, deps):
    actions = []
    # --- launchd jobs ---
    for label in R.ALLOWED_JOBS:
        out = deps["job_state"](label)
        fresh = deps["job_fresh"](label)
        if not R.job_unhealthy(label, out, fresh):
            _maybe_recover(conn, "launchd-job", label, now, deps, actions)
            continue
        _handle(conn, "launchd-job", label, mode, now, deps, actions,
                reason="startup crash / abnormal exit", fix=lambda label=label: deps["kickstart"](label))
    # --- hetzner worker ---
    if not deps["hetzner_ok"]():
        _handle(conn, "hetzner-worker", "wtl-worker", mode, now, deps, actions,
                reason="worker hung", fix=deps["restart_worker"])
    else:
        _maybe_recover(conn, "hetzner-worker", "wtl-worker", now, deps, actions)
    # --- outward reachability: is the site actually reachable from outside? ---
    fault = R.net_fault(deps["ha_connections"](), deps["raw_internet_ok"]())
    prev = L.last_open_problem(conn, "net-wedged", "tunnel")
    if fault is None:
        if prev and prev["action"] == "detect":
            # A blip that cleared before we acted. Close it, or the next fault
            # inside the confirm window would inherit this sighting and skip
            # straight to a repair on its own first tick.
            L.record(conn, "net-wedged", "tunnel", "recover", mode, 0, True,
                     "cleared before action", ts=now)
            actions.append("recover:tunnel")
        else:
            _maybe_recover(conn, "net-wedged", "tunnel", now, deps, actions)
    elif not R.net_confirmed(prev, now):
        # First sighting. Record it and do nothing — the next tick decides.
        L.record(conn, "net-wedged", "tunnel", "detect", mode, None, None,
                 f"unconfirmed: {fault} | {deps['net_diag']()}", ts=now)
        actions.append(f"detect:tunnel:{fault}")
    else:
        _handle(conn, "net-wedged", "tunnel", mode, now, deps, actions,
                reason=f"{NET_REASON[fault]} [{deps['net_diag']()}]",
                fix=lambda f=fault: deps["repair_net"](f))
    # --- heartbeat always ---
    deps["post_heartbeat"]()
    actions.append("heartbeat")
    return actions


def _budget(rule_id):
    for r in R.WHITELIST:
        if r["id"] == rule_id:
            return r["budget"]
    return (1, 1800)


def _handle(conn, rule_id, target, mode, now, deps, actions, reason, fix):
    n, window = _budget(rule_id)
    if mode == "observe":
        L.record(conn, rule_id, target, "detect", mode, None, None, f"would-fix: {reason}", ts=now)
        actions.append(f"observe:{target}")
        return
    if L.count_in_window(conn, rule_id, target, window, now) >= n:
        prev = L.last_open_problem(conn, rule_id, target)
        if not prev or prev["action"] != "escalate":
            deps["send"](N.fmt_escalate(target, reason, deps["log_tail"](target), _hint(rule_id)))
            L.record(conn, rule_id, target, "escalate", mode, None, None, "budget exhausted", ts=now)
            actions.append(f"escalate:{target}")
        return
    code = fix()
    L.record(conn, rule_id, target, "fix", mode, code, None, reason, ts=now)
    actions.append(f"fix:{target}")
    # NOTE: recovery is verified on the NEXT tick, not now.


def _maybe_recover(conn, rule_id, target, now, deps, actions):
    prev = L.last_open_problem(conn, rule_id, target)
    if not prev or prev["action"] == "recover":
        return
    if prev["action"] == "fix":
        # Telegram silenced (2026-08-05): tier-1 is the "auto-fixed" FYI — once it
        # has self-healed there is nothing for you to do; the message itself said
        # "no action needed". The ledger still records it (admin → Runs still shows
        # it); it just no longer pushes to the phone. Only tier-1 is silenced; the
        # ✅ recovery receipt in the escalate branch below stays — it closes a 🔴
        # you really did receive, i.e. a "recovered after being down a while"
        # message, which meets the bar for recovery receipts.
        print(f"[quiet] watchdog auto-fix of {target} verified recovered — not notifying", flush=True)
        L.record(conn, rule_id, target, "recover", "enforce", 0, True, "verified", ts=now)
        actions.append(f"recover:{target}")
    elif prev["action"] == "escalate":
        deps["send"](N.fmt_recovered(target, 3600))
        L.record(conn, rule_id, target, "recover", "enforce", 0, True, "post-escalation", ts=now)
        actions.append(f"recover-after-escalation:{target}")


def _hint(rule_id):
    if rule_id == "launchd-job":
        return "check that the wtl-pyrun.sh wrapper is installed and the venv is healthy"
    if rule_id == "net-wedged":
        # Precedent from 2026-08-28: a reboot cured the 33.5-hour outage; automation
        # does not dare press that button for you.
        return ("auto-repair did not cure it — this Mac most likely needs a reboot "
                "(that is how 2026-08-28 was recovered)")
    return f"ssh {HETZNER} systemctl status wtl-worker"


# ---- real dependency wiring ----
def _uid():
    return os.getuid()


def _job_state(label):
    out = subprocess.run(["launchctl", "list"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.endswith(f"xyz.wheretolive.{label}"):
            return line
    return f"-\t0\txyz.wheretolive.{label}"


def _kickstart(label):
    return subprocess.run(
        ["launchctl", "kickstart", "-k", f"gui/{_uid()}/xyz.wheretolive.{label}"]).returncode


def _hetzner_ok() -> bool:
    """SSH to Hetzner, check wtl-worker active. 5s connect + 5s exec.
    (In the private repo this lives in the shared monitor_checks module as
    check_hetzner_worker(); moved here because the public monitor_checks
    extract does not carry it.)"""
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             "-o", "StrictHostKeyChecking=no", HETZNER,
             "systemctl", "is-active", "wtl-worker"],
            capture_output=True, text=True, timeout=12
        )
        return r.stdout.strip() == "active"
    except Exception:
        return False


def _restart_worker():
    return subprocess.run(["ssh", HETZNER, "systemctl", "restart", "wtl-worker"]).returncode


def _ha_connections():
    """How many edge connections cloudflared currently holds. None = its metrics
    port didn't answer, which for our purposes is the same as zero."""
    try:
        with urllib.request.urlopen(CF_METRICS_URL, timeout=5) as r:
            return R.parse_ha_connections(r.read().decode("utf-8", "replace"))
    except Exception:
        return None


def _raw_internet_ok():
    """Can this box open a TCP connection to the public internet at all? Literal
    IPs only — on 2026-08-27 the resolver died along with the routing, so any
    hostname here would have made the probe agree with itself for the wrong
    reason."""
    for host, port in EGRESS_PROBES:
        try:
            socket.create_connection((host, port), timeout=4).close()
            return True
        except OSError:
            continue
    return False


NETWORKSETUP = "/usr/sbin/networksetup"


def _wifi_power(run):
    out = run([NETWORKSETUP, "-getairportpower", WIFI_IF],
              capture_output=True, text=True, timeout=15)
    return "On" if "On" in (getattr(out, "stdout", "") or "").split(":")[-1] else "Off"


def _bounce_wifi(run=None, sleep=None):
    """Tear the Wi-Fi interface down and back up. No sudo needed. Only reached
    after two consecutive ticks proved the box has no route out at all, so there
    is no working network here to break.

    Switching the radio on is unconditional and always last: leaving it off
    would strand the box in the very outage this repair exists to end, and once
    the repair budget is spent nothing comes back to switch it on. If a previous
    tick died between off and on, the radio is already off — then just switch it
    on rather than cycling it again."""
    run, sleep = run or subprocess.run, sleep or time.sleep
    try:
        if _wifi_power(run) == "On":
            run([NETWORKSETUP, "-setairportpower", WIFI_IF, "off"],
                capture_output=True, timeout=30)
            sleep(5)
    except Exception:
        pass  # whatever went wrong, the radio still has to come back
    r = run([NETWORKSETUP, "-setairportpower", WIFI_IF, "on"],
            capture_output=True, timeout=30)
    return getattr(r, "returncode", 1)


def _net_diag():
    """One line of routing state, captured at fault time. 2026-08-27 can no
    longer be explained because nothing wrote this down while it was happening —
    by the time anyone looked, macOS's unified log had rolled over. Read-only."""
    gw = iface = "?"
    try:
        out = subprocess.run(["/sbin/route", "-n", "get", "default"],
                             capture_output=True, text=True, timeout=5).stdout or ""
        for line in out.splitlines():
            k, _, v = line.strip().partition(":")
            if k == "gateway":
                gw = v.strip()
            elif k == "interface":
                iface = v.strip()
    except Exception:
        pass
    reach = "?"
    if gw != "?":
        try:
            reach = "up" if subprocess.run(
                ["/sbin/ping", "-c", "1", "-W", "2000", gw],
                capture_output=True, timeout=6).returncode == 0 else "down"
        except Exception:
            pass
    try:
        wifi = _wifi_power(subprocess.run).lower()
    except Exception:
        wifi = "?"
    return f"default={gw}@{iface} gw={reach} wifi={wifi}"


def _repair_net(fault):
    """cloudflared's own problem → restart it. Host has no route → rebuild the
    interface; restarting cloudflared against a routeless kernel is theatre."""
    return _kickstart("cloudflared") if fault == "cloudflared" else _bounce_wifi()


def main():
    # Telegram sender: the shared monitor_checks module in ../monitoring.
    sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "monitoring"))
    import monitor_checks as C
    from watchdog_heartbeat import post_heartbeat
    from watchdog_fresh import job_fresh, log_tail
    mode = os.environ.get("WATCHDOG_MODE", "observe")
    conn = L.open_db()
    deps = dict(job_state=_job_state, job_fresh=job_fresh, kickstart=_kickstart,
                hetzner_ok=_hetzner_ok, restart_worker=_restart_worker,
                send=C.tg, post_heartbeat=post_heartbeat, log_tail=log_tail,
                ha_connections=_ha_connections, raw_internet_ok=_raw_internet_ok,
                repair_net=_repair_net, net_diag=_net_diag)
    acted = run_once(conn, mode, datetime.now(timezone.utc), deps)
    print(f"[watchdog] mode={mode} actions={acted}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
