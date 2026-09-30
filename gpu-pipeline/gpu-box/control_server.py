#!/usr/bin/env python3
"""Lightweight control endpoint for managing embedding daemons remotely.

Why: the mac admin UI wants start/stop buttons for the embedding workers,
but the daemons live on the GPU box and run under systemd. Three options
were on the table:

1. SSH from the Mac's Next.js server (launchd) → systemctl. Awkward — ssh key
   discovery from launchd-spawned node is finicky, ssh failure modes
   are opaque, and we'd be granting the web server sudoless ssh.
2. Polkit + a dropped privilege wrapper. Heavy for two buttons.
3. Tiny HTTP endpoint on the GPU box that proxies systemctl. ← this.

Endpoint: http://$GPU_HOST:8200
  GET  /workers              → status of all managed daemons
  POST /worker/<name>/start  → systemctl start
  POST /worker/<name>/stop   → systemctl stop
  POST /worker/<name>/restart → systemctl restart

Auth: x-admin-key header matched against WTL_ADMIN_KEY env var. Same
key the report scripts use to push to mac — already shared between
the two hosts via the units' EnvironmentFile (/etc/wtl/wtl.env).

Only whitelisted units can be controlled (see UNITS below) so a
stolen admin key can at most flap two embedding daemons. The service
runs as root to skip the sudoers dance — easier to maintain and the
attack surface is bounded by the whitelist. Endpoint is only reachable
over Tailscale + admin-key auth.
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 8200
ADMIN_KEY = os.environ.get("WTL_ADMIN_KEY", "")
if not ADMIN_KEY:
    print("FATAL: WTL_ADMIN_KEY env var required", file=sys.stderr)
    sys.exit(1)

# Whitelist — only these units can be controlled. Catches typos and
# stops a stolen key from rebooting random services.
UNITS = {
    # Canonical name post-merge — DINO + SigLIP run in one process
    # (embed_daemon_combined.py) so one start/stop controls both.
    "embed-combined": "wtl-embed-combined.service",
    # Backwards-compatible aliases — the two existing admin cards
    # (EmbeddingWorkerSection + SiglipWorkerSection) each have a
    # WorkerControlButton; both now target the combined daemon.
    "dino-embed":     "wtl-embed-combined.service",
    "siglip-embed":   "wtl-embed-combined.service",
}
ACTIONS = {"start", "stop", "restart"}


def _systemctl(*args: str) -> tuple[int, str]:
    """Run `sudo systemctl <args>`. Returns (returncode, stdout+stderr).
    Times out at 30s so a stuck unit doesn't hang the http server."""
    try:
        # Server runs as root (see module docstring), so no `sudo` here.
        r = subprocess.run(
            ["systemctl", *args],
            capture_output=True, text=True, timeout=30,
        )
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return 124, "systemctl timed out (30s)"
    except Exception as e:
        return 1, str(e)


def worker_status(unit: str) -> dict:
    """Return {active, sub, since, pid, mem_mb, cpu_s}. Best-effort —
    missing fields default to None."""
    rc, out = _systemctl("show", unit,
                         "--property=ActiveState,SubState,ActiveEnterTimestamp,MainPID,MemoryCurrent,CPUUsageNSec")
    if rc != 0:
        return {"error": out}
    info: dict = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k] = v
    mem_bytes = info.get("MemoryCurrent", "")
    mem_mb = None
    try:
        if mem_bytes and mem_bytes != "[not set]":
            mem_mb = round(int(mem_bytes) / 1024 / 1024, 1)
    except ValueError:
        pass
    cpu_ns = info.get("CPUUsageNSec", "")
    cpu_s = None
    try:
        if cpu_ns:
            cpu_s = round(int(cpu_ns) / 1e9, 1)
    except ValueError:
        pass
    return {
        "active":  info.get("ActiveState"),
        "sub":     info.get("SubState"),
        "since":   info.get("ActiveEnterTimestamp"),
        "pid":     int(info.get("MainPID") or 0) or None,
        "mem_mb":  mem_mb,
        "cpu_s":   cpu_s,
    }


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, body: dict):
        b = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _auth_ok(self) -> bool:
        if self.headers.get("x-admin-key") != ADMIN_KEY:
            self._json(401, {"error": "unauthorized"})
            return False
        return True

    def do_GET(self):
        if not self._auth_ok():
            return
        if self.path == "/workers":
            workers = {name: {"unit": unit, **worker_status(unit)}
                       for name, unit in UNITS.items()}
            return self._json(200, {"workers": workers, "ts": time.time()})
        if self.path == "/health":
            return self._json(200, {"status": "ok"})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self._auth_ok():
            return
        parts = self.path.strip("/").split("/")
        if len(parts) != 3 or parts[0] != "worker":
            return self._json(404, {"error": "use POST /worker/<name>/<action>"})
        _, name, action = parts
        if name not in UNITS:
            return self._json(400, {"error": f"unknown worker '{name}'", "available": list(UNITS)})
        if action not in ACTIONS:
            return self._json(400, {"error": f"unknown action '{action}'", "available": list(ACTIONS)})
        unit = UNITS[name]
        rc, out = _systemctl(action, unit)
        ok = (rc == 0)
        status = worker_status(unit)
        return self._json(200 if ok else 500,
                          {"ok": ok, "name": name, "action": action,
                           "exit_code": rc, "output": out, "status": status})

    def log_message(self, format, *args):
        # Keep stderr terse; systemd journal captures these.
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {format % args}\n")


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[ready] wtl-control on :{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
