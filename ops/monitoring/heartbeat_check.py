#!/usr/bin/env python3
"""Runs on gpu-box (systemd timer, every 5 min). Reads the admin host-metrics
API and alerts via Telegram when any watched host's heartbeat goes stale — i.e.
its host_report / watchdog stopped posting, meaning the box (or that daemon) is
down. Event-driven: alerts once on the down-transition and once on recovery,
using a small state file so a sustained outage doesn't spam every 5 min.

Covers the mac-watchdog blind spot (a dead watchdog can't self-report) AND the
3090 WSL / gpu-box hosts (host_report going dark = the box died). TG creds
match the rest of the fleet. WTL_ADMIN_KEY is a secret — supply via env."""
import os
import json
import urllib.parse
import urllib.request
from pathlib import Path

API = os.environ.get("WTL_API", "http://localhost:3000")
ADMIN_KEY = os.environ.get("WTL_ADMIN_KEY", "")

# Deliberately self-contained credential reading: this script runs on gpu-box
# (systemd) and cannot import the Mac's wtl_tg.py.
# The token is read from
# ~/.config/wtl/telegram.env (or env vars). That file must exist on gpu-box
# (chmod 600), otherwise this script fails at startup — on purpose: a heartbeat
# alerter that silently stops working blinds the whole monitoring net.
CRED_FILE = Path(os.environ.get(
    "WTL_TG_ENV", str(Path.home() / ".config" / "wtl" / "telegram.env")))


def _cred(key: str) -> str:
    val = os.environ.get(key)
    if val:
        return val
    try:
        for line in CRED_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k.strip() == key:
                    return v.strip()
    except OSError as e:
        raise SystemExit(f"[heartbeat] cannot read credential file {CRED_FILE}: {e}") from e
    raise SystemExit(f"[heartbeat] {key} missing from {CRED_FILE}")


TG_TOKEN = _cred("WTL_TG_TOKEN")
TG_CHAT = _cred("WTL_TG_CHAT")
STATE_FILE = Path(os.environ.get("WTL_HEARTBEAT_STATE",
                                 str(Path.home() / ".config" / "wtl" / "heartbeat_state.json")))

# (host_id, stale_seconds, human label). host_report posts ~every 60s; the
# watchdog every 120s — 600s/900s thresholds tolerate a few missed posts.
WATCHED = [
    ("mac-mini", 600, "Mac host_report"),
    ("mac-watchdog", 600, "Mac watchdog (self-healing)"),
    ("windows-3090", 900, "3090 WSL"),
    ("gpu-box", 900, "gpu-box host_report"),
]


def evaluate(ages, prev_down, api_ok=True):
    """Pure. Given {host_id: age_seconds} and the set of already-flagged hosts,
    return (alerts, new_down). alerts = list of (host_id, 'down'|'up', label).
    A host is down if missing or older than its threshold. If the whole Mac is
    down (mac-mini stale), suppress the mac-watchdog alert — the mac-mini one is
    the actionable signal, no need to double-ping.

    `api_ok=False` = host-metrics could not be reached at all. That API RUNS ON
    THE MAC, so failing to reach it IS the mac-mini down signal — and the only
    thing it tells us. Every other host's state is read THROUGH the Mac, so it
    becomes unknown, not down: carry it over from prev_down rather than inventing
    either verdict. (Reporting "gpu-box unreachable" from gpu-box itself would be
    absurd, and a spurious ✅ during an outage is worse than silence.)

    Why this branch exists: on 2026-08-27 the Mac's DNS/Tailscale died for 33.5h
    and this checker — the one named "alerts if Mac goes dark" — exited 1 every
    5 minutes on the fetch and never said a word."""
    label_of = {hid: label for hid, _s, label in WATCHED}
    prev_down = set(prev_down or [])
    if not api_ok:
        down = prev_down | {"mac-mini"}
    else:
        down = set()
        for hid, stale_s, _label in WATCHED:
            age = ages.get(hid)
            if age is None or age > stale_s:
                down.add(hid)
    if "mac-mini" in down:
        down.discard("mac-watchdog")
        # …and drop it from the comparison too, or the discard above reads as a
        # recovery and we announce "✅ watchdog reporting again" mid-outage.
        prev_down = prev_down - {"mac-watchdog"}
    alerts = [(hid, "down", label_of[hid]) for hid in sorted(down - prev_down)]
    alerts += [(hid, "up", label_of.get(hid, hid)) for hid in sorted(prev_down - down)]
    return alerts, down


def _fetch_ages(opener=urllib.request.urlopen):
    req = urllib.request.Request(f"{API}/api/admin/host-metrics",
                                 headers={"x-admin-key": ADMIN_KEY})
    with opener(req, timeout=15) as r:
        data = json.loads(r.read().decode())
    return {h["host_id"]: h.get("age_seconds") for h in data.get("hosts", [])}


def _load_state():
    try:
        return set(json.loads(STATE_FILE.read_text()).get("down", []))
    except Exception:
        return set()


def _save_state(down):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({"down": sorted(down)}))


def _tg(text):
    d = urllib.parse.urlencode({"chat_id": TG_CHAT, "text": text}).encode()
    urllib.request.urlopen(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                           data=d, timeout=10).read()


def main(fetch=None, tg=None):
    """`fetch`/`tg` are injectable for tests; production uses the module defaults.

    A failed fetch must NOT abort: unreachable-Mac is the single most important
    thing this checker exists to report, and our TG credentials + network are
    local to gpu-box, so we can still send even when the Mac is gone."""
    fetch = fetch or _fetch_ages
    tg = tg or _tg
    try:
        ages, api_ok = fetch(), True
    except Exception as e:            # network, HTTP, malformed JSON — all mean "no data"
        ages, api_ok = {}, False
        print(f"[heartbeat-check] host-metrics unreachable ({e}) — treating as mac-mini down")
    alerts, down = evaluate(ages, _load_state(), api_ok=api_ok)
    for hid, kind, label in alerts:
        if kind == "down":
            tg(f"🔴🔴 {label} unreachable — host heartbeat stopped, {hid} may be down")
        else:
            tg(f"✅ {label} reporting again")
    _save_state(down)
    print(f"[heartbeat-check] api_ok={api_ok} down={sorted(down)} "
          f"alerts={[(h,k) for h,k,_ in alerts]} ages={ages}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
