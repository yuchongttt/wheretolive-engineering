#!/usr/bin/env python3
"""Declarative whitelist + launchd-job health detection for the watchdog."""
import json
import os
from pathlib import Path

# The launchd allowlist is deployment data, so it lives in watchdog_jobs.json
# (override the path with WATCHDOG_JOBS_FILE) rather than in code. Each entry:
#   "<label>": {"fresh_minutes": N}                     -> freshness from the job's log mtime
#   "<label>": {"fresh_minutes": N, "ledger_skill": S}  -> freshness from the newest
#                                                          ops.db skill_runs row for skill S
# It is still an explicit allowlist — NOT "any launchd job". next-prod and the
# GPU/image daemons are deliberately never on it.
#
# History: changelog-poll was retired on 2026-07-27 together with the changelog
# pipeline; its unit was booted out,
# disabled and removed from LaunchAgents, so it was taken off the watch list.
# Leaving it on would not have caused false alarms (for a missing unit
# _job_state returns last_exit=0, i.e. healthy), but it would have been dead
# config that never fires and only confuses.
JOBS_FILE = Path(os.environ.get("WATCHDOG_JOBS_FILE")
                 or Path(__file__).resolve().with_name("watchdog_jobs.json"))


def load_jobs(path=JOBS_FILE):
    """-> (ALLOWED_JOBS, JOB_FRESH_MINUTES, LEDGER_FRESH_JOBS)."""
    jobs = json.loads(Path(path).read_text()).get("jobs", {})
    allowed = list(jobs)
    fresh = {label: int(spec.get("fresh_minutes", 70)) for label, spec in jobs.items()}
    ledger = {label: spec["ledger_skill"] for label, spec in jobs.items()
              if spec.get("ledger_skill")}
    return allowed, fresh, ledger


# ALLOWED_JOBS: labels the watchdog may kickstart.
# JOB_FRESH_MINUTES: per-job freshness budget (minutes) — how recently we expect a
#   healthy signal.
# LEDGER_FRESH_JOBS: label -> skill_runs.skill_name for jobs whose healthy signal is
#   a run-ledger row rather than a log file.
ALLOWED_JOBS, JOB_FRESH_MINUTES, LEDGER_FRESH_JOBS = load_jobs()


# Ledger-tracked jobs: a partitioned catch-up after an outage (a skill_runs row
# still 'running' with summary mode=catchup) may run up to ~2 h. It starts right
# after failed runs, so last_exit != 0 — without this the watchdog would
# `kickstart -k` it at 70 min, every time. Mirrors the monitor's own
# "catch-up running too long" threshold.
CATCHUP_FRESH_MINUTES = 180


def parse_launchctl(job_line: str) -> dict:
    """Parse one `launchctl list` row: '<pid>\\t<lastexit>\\t<label>'."""
    parts = job_line.strip().split("\t")
    if len(parts) < 3:
        return {"pid": None, "last_exit": 0}
    pid = None if parts[0] == "-" else int(parts[0])
    last_exit = int(parts[1]) if parts[1].lstrip("-").isdigit() else 0
    return {"pid": pid, "last_exit": last_exit}


def job_unhealthy(label: str, launchctl_out: str, fresh: bool) -> bool:
    """Down = non-zero last exit AND no recent healthy signal. A one-shot that
    ran fine (exit 0, no process) is healthy; a job that crashed but whose log
    is still fresh gets one more benefit-of-the-doubt cycle."""
    st = parse_launchctl(launchctl_out)
    return (st["last_exit"] != 0) and (not fresh)


# --- outward reachability ------------------------------------------------
# 2026-08-27 12:51Z→08-28 22:24Z the Mac stayed up but lost every route to the
# public internet (cloudflared: `sendmsg: network is unreachable`, 11,679 times;
# the hourly ingestion job failed the same way). DNS went with it — the box's
# only resolvers are Tailscale's MagicDNS. Nothing here noticed for 33h32m; a
# manual reboot ended it. Googlebot spent that window collecting Cloudflare 530s.
#
# Two ticks must agree before we touch anything: cloudflared reports zero
# connections for a second or two across its own restarts, and the watchdog
# ticks every 120s, so a single sighting proves nothing.
NET_CONFIRM_WINDOW_S = 15 * 60


def parse_ha_connections(metrics_text):
    """Pull `cloudflared_tunnel_ha_connections` out of cloudflared's local
    metrics page. None = couldn't read it (port dead / gauge absent), which the
    caller treats as "tunnel down". Deliberately local-only: 127.0.0.1 needs no
    DNS, and DNS is one of the things that dies in this failure."""
    for line in (metrics_text or "").splitlines():
        if line.startswith("cloudflared_tunnel_ha_connections "):
            try:
                return int(float(line.split()[1]))
            except (IndexError, ValueError):
                return None
    return None


def net_fault(ha_connections, raw_internet_ok):
    """Which repair the evidence points at, or None when healthy.

    'cloudflared'  — the tunnel is down but the host still reaches the public
                     internet by raw IP, so the fault is cloudflared's own.
    'host-network' — the tunnel is down AND raw-IP egress fails: the 2026-08-27
                     shape. Restarting cloudflared cannot fix a kernel with no
                     route; the interface has to be rebuilt.

    A failing egress probe on its own is never a fault — if the tunnel is
    carrying traffic, the probe is the thing that's wrong, not the network."""
    if ha_connections is not None and ha_connections > 0:
        return None
    return "cloudflared" if raw_internet_ok else "host-network"


def net_confirmed(prev, now, window_s=NET_CONFIRM_WINDOW_S):
    """True once a *recent* earlier tick saw the fault too. `prev` is the
    ledger's last row for this rule; a 'recover' row means the previous episode
    closed, and anything older than the window belongs to that old episode."""
    if not prev or prev["action"] == "recover":
        return False
    from datetime import datetime
    try:
        age = now.timestamp() - datetime.fromisoformat(prev["ts"]).timestamp()
    except (TypeError, ValueError):
        return False
    return age <= window_s


# Rules are declarative; the loop (watchdog.py) supplies runtime callables.
WHITELIST = [
    {"id": "launchd-job", "budget": (3, 30 * 60), "fix_kind": "kickstart",
     "targets": ALLOWED_JOBS},
    {"id": "hetzner-worker", "budget": (1, 30 * 60), "fix_kind": "ssh-restart-worker",
     "targets": ["wtl-worker"]},
    # 2 repairs per 30 min: one per rung (cloudflared restart, Wi-Fi bounce).
    # Past that the box needs a human — keep bouncing the interface and you just
    # keep it offline.
    {"id": "net-wedged", "budget": (2, 30 * 60), "fix_kind": "restart-net",
     "targets": ["tunnel"]},
]
