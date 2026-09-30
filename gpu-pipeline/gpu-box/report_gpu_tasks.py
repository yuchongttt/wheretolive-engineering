#!/usr/bin/env python3
"""Report GPU overall state + active compute processes to Mac admin.

Picks up optional progress JSON from /data/ml/gpu_status/<pid>.json
(scripts can opt in by writing this file every N seconds).
"""
import json, os, subprocess, sys, urllib.request
from datetime import datetime, timezone

API  = os.environ.get("MAC_ADMIN_URL", "http://mac-admin:3000")
KEY  = os.environ.get("WTL_ADMIN_KEY", "")
HOST = os.environ.get("WTL_HOST_ID",   "gpu-box")
URL  = f"{API}/api/admin/gpu-tasks"
STATUS_DIR = "/data/ml/gpu_status"

if not KEY:
    print("missing WTL_ADMIN_KEY", file=sys.stderr); sys.exit(1)

# --- Overall GPU ---
try:
    out = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,fan.speed",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5,
    ).stdout.strip()
    parts = [s.strip() for s in out.split(",")]
    name, util, mem_used, mem_total, temp, power, fan = parts
    gpu = {
        "name": name,
        "util_pct": int(util) if util.isdigit() else None,
        "mem_used_mb": int(mem_used),
        "mem_total_mb": int(mem_total),
        "temp_c": int(temp),
        "power_w": float(power),
        "fan_pct": int(fan) if fan.replace(".", "").isdigit() else None,
    }
except Exception as e:
    gpu = {"error": str(e)}

# --- Active compute processes ---
processes = []
try:
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5,
    ).stdout.strip()
    for line in apps.split("\n") if apps else []:
        bits = [p.strip() for p in line.split(",")]
        if len(bits) < 3: continue
        pid, pname, mem_mb = bits

        # cmdline
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
        except Exception:
            cmdline = pname

        # running_seconds
        running_seconds = None
        try:
            with open(f"/proc/{pid}/stat") as f:
                stat = f.read().split()
            clk_tck = os.sysconf("SC_CLK_TCK")
            with open("/proc/uptime") as f:
                uptime = float(f.read().split()[0])
            start_ticks = int(stat[21])
            running_seconds = int(uptime - start_ticks / clk_tck)
        except Exception: pass

        # Optional progress JSON written by the script itself
        progress = None
        status_path = f"{STATUS_DIR}/{pid}.json"
        if os.path.exists(status_path):
            try:
                with open(status_path) as f:
                    progress = json.load(f)
            except Exception: pass

        # Heuristic: detect well-known script names
        script_name = None
        for tok in cmdline.split():
            if tok.endswith(".py"):
                script_name = os.path.basename(tok)
                break

        processes.append({
            "pid": int(pid),
            "process_name": pname,
            "script_name": script_name,
            "cmdline": cmdline[:300],
            "gpu_mem_mb": int(mem_mb),
            "running_seconds": running_seconds,
            "progress": progress,
        })
except Exception as e:
    processes = [{"error": str(e)}]

payload = {
    "host_id": HOST,
    "reported_at": datetime.now(timezone.utc).isoformat(),
    "gpu": gpu,
    "processes": processes,
}
req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "x-admin-key": KEY})
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        body = r.read().decode()
        if '"ok":true' not in body:
            print(f"[report] unexpected: {body}", file=sys.stderr); sys.exit(1)
except Exception as e:
    print(f"[report] POST failed: {e}", file=sys.stderr); sys.exit(1)
