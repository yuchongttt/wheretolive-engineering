#!/usr/bin/env python3
"""GPU-box health check — the point is that the GPU is actually usable, not
merely that "the services are still running".

Origin (2026-07-24): unattended-upgrades upgraded the NVIDIA driver but the host
was not rebooted, so the kernel module (580.159.03) and the userspace library
(580.173.02) were out of sync. In processes that were **already running**, the
CUDA context had been created before the upgrade and kept working; but **any
newly started** process got cuda_available=False. So four ML services all looked
green while in fact surviving only because "they started before the upgrade";
one restart and they would all lose the GPU together. This state stayed hidden
for 3 days — the GPU box's gpu field in the admin page became null (because
nvidia-smi itself was broken too), but nothing alerted.

So the checks here deliberately do two things an ordinary "service alive" check
cannot:

  driver_match  compare kernel module vs userspace library version. **Detects it
                at upgrade time**, without waiting for some process restart to
                expose it. This is the signal that should have fired on 07-24.
  cuda_new_proc start a **brand-new** process and test torch.cuda.is_available().
                However healthy the existing processes are, they cannot prove a
                new process can get the GPU — which is exactly what fooled us.

Usage:
    python3 mac/gpu_box_healthcheck.py              # human-readable, post-reboot acceptance
    python3 mac/gpu_box_healthcheck.py --quiet      # print only FAIL lines + exit code

Exit code 0 = all passed; 1 = some check failed.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

SSH_HOST = os.environ.get("GPU_HOST", "gpu-box")
SSH_OPTS = ["-o", "ConnectTimeout=10", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no"]
VENV_PY = "/data/ml/venv/bin/python3"
EMBED_DB = "/data/ml/dataset/dataset.db"   # note: NOT /data/ml/dataset.db

# Units that should come back automatically after a reboot (the enabled ones).
# dino is deliberately off and not listed.
SYSTEM_UNITS = ["wtl-control", "wtl-embed-combined", "wtl-epc-server", "wtl-style-score"]
USER_UNITS = ["wtl-siglip-server"]

# Above this restart count, treat it as a crash loop. A healthy unit's NRestarts
# is single-digit (the occasional OOM restart); a crash loop counts in the tens of
# thousands (in the 2026-07-24 incident wtl-style-score reached 14536 in 3 days).
FLAP_RESTARTS = 20

# Embed output window (minutes). When uptime is shorter than this, the check
# steps aside; see check_embed_progress.
WINDOW_MIN = 30

# "Could not determine" != "broken". An ssh blip or a command timeout must not be
# treated by automatic alerting as a GPU fault. Failures carrying this prefix are
# skipped by gpu_watchdog instead of paging (human-readable mode shows them).
INCONCLUSIVE = "INCONCLUSIVE:"


def sh(cmd: str, timeout: int = 45, stdin: str | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run(["ssh", *SSH_OPTS, SSH_HOST, cmd],
                           input=stdin, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except Exception as e:  # noqa: BLE001
        return 1, str(e)


def sh_py(source: str, timeout: int = 90) -> tuple[int, str]:
    """Feed Python source to the remote interpreter over stdin. Avoids stuffing
    code into the ssh command line — that needs three nested layers of quoting
    (local shell / ssh / remote shell) and is very easy to get wrong."""
    return sh(f"{VENV_PY} -", timeout=timeout, stdin=source)


def check_reachable() -> tuple[bool, str]:
    code, out = sh("uptime", timeout=20)
    return code == 0, out.strip() if code == 0 else f"ssh unreachable: {out}"


def check_driver_match() -> tuple[bool, str]:
    """Kernel module version == userspace library version? If not, it was
    "upgraded but not rebooted"."""
    _, kern = sh("cat /proc/driver/nvidia/version 2>/dev/null")
    _, libs = sh("ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.* 2>/dev/null")
    k = re.search(r"Kernel Module\s+([\d.]+)", kern)
    u = re.findall(r"libnvidia-ml\.so\.([\d.]+)", libs)
    if not k or not u:
        return False, f"{INCONCLUSIVE} cannot read versions (kernel={kern[:50]!r} libs={libs[:50]!r})"
    kv, uv = k.group(1), sorted(u)[-1]
    if kv == uv:
        return True, f"kernel module and userspace library match ({kv})"
    return False, (f"driver version mismatch: kernel module {kv} != userspace library {uv} — "
                   f"upgraded without reboot, new processes cannot get the GPU; reboot the GPU box")


def check_cuda_new_proc() -> tuple[bool, str]:
    """The key check: can a brand-new process actually get CUDA."""
    _, out = sh_py(
        "import torch\n"
        "ok = torch.cuda.is_available()\n"
        "print('CUDA_OK' if ok else 'CUDA_FAIL')\n"
        "print(torch.cuda.get_device_name(0) if ok else '')\n",
        timeout=120,
    )
    if "CUDA_OK" in out:
        gpu = [l for l in out.splitlines() if l.strip() and "CUDA_OK" not in l]
        return True, f"new process got the GPU{(' — ' + gpu[-1]) if gpu else ''}"
    if "CUDA_FAIL" not in out:
        # torch never ran (ssh dropped / timeout / venv problem) — this is not
        # evidence that "the GPU is broken".
        return False, f"{INCONCLUSIVE} probe did not run: {out[-120:].strip()}"
    err = next((l for l in out.splitlines() if "Error" in l), out[-160:])
    return False, f"new process cannot get the GPU: {err.strip()}"


def check_units() -> tuple[bool, str]:
    """Units are running — and not idling in a crash-restart loop.

    Looking only at is-active gets fooled: a crash-looping unit bounces between
    active / activating / failed, and a sample has a good chance of landing on
    active. In the 2026-07-24 driver incident wtl-style-score "looked normal"
    this way while spinning for 3 days 9 hours, restarting 14536 times and burning
    CPU without anyone noticing. So this also reads NRestarts and compares it
    with the previous snapshot."""
    sys_cmd = "; ".join(
        f'echo "{u}=$(systemctl is-active {u}):$(systemctl show {u} -p NRestarts --value)"'
        for u in SYSTEM_UNITS)
    usr_cmd = "; ".join(
        f'echo "{u}(user)=$(systemctl --user is-active {u}):-"' for u in USER_UNITS)
    _, out = sh(f"{sys_cmd}; {usr_cmd}")

    down, flapping, starting = [], [], []
    for line in out.splitlines():
        if "=" not in line:
            continue
        unit, _, state = line.partition("=")
        active, _, restarts = state.partition(":")
        if active not in ("active", "activating"):
            down.append(f"{unit}={active}")
        elif restarts.isdigit() and int(restarts) > FLAP_RESTARTS:
            # The crash-loop criterion is the restart count, not the current
            # state — state sampling lies.
            flapping.append(f"{unit} restarted {restarts} times")
        elif active == "activating":
            # Right after a restart a unit is normally briefly 'activating'
            # (model load takes a dozen seconds or so), and the restart count is
            # low, so it is not a loop. Note it, don't fail it.
            starting.append(unit)

    if down or flapping:
        return False, "; ".join(down + flapping)
    note = f"({', '.join(starting)} starting)" if starting else ""
    return True, f"{len(SYSTEM_UNITS) + len(USER_UNITS)} units active, no restart loop {note}".strip()


def check_embed_progress() -> tuple[bool, str]:
    """Has the embedding pipeline actually produced output recently (confirms
    the GPU pipeline recovered after a reboot).

    Two traps: the GPU box has no sqlite3 CLI installed (so this goes through
    the venv python); and created_at is stored as ISO with a T and timezone
    (`2026-07-27T14:34:15+00:00`), so a bare string comparison against
    datetime('now')'s space format doesn't line up — 'T' > ' ', so every row from
    the same day always counts as "recent" and the window is meaningless. It must
    be wrapped in datetime()."""
    _, out = sh_py(
        "import sqlite3\n"
        "print(int(float(open('/proc/uptime').read().split()[0])))\n"
        f"c = sqlite3.connect('file:{EMBED_DB}?mode=ro', uri=True)\n"
        "print(c.execute(\n"
        "    \"SELECT COUNT(*) FROM image_embeddings \"\n"
        f"    \"WHERE datetime(created_at) > datetime('now','-{WINDOW_MIN} minutes')\"\n"
        ").fetchone()[0])\n"
        # Pending work: downloaded, not a floorplan (which SigLIP skips by
        # policy), and no vector yet. With nothing pending, "zero output" is
        # idle, not a fault — must be queried together, otherwise we'd be
        # reading "no output" as "broken".
        "print(c.execute(\n"
        "    \"SELECT COUNT(*) FROM images i WHERE i.status='done' \"\n"
        "    \"AND COALESCE(i.media_type,'') != 'floorplan' \"\n"
        "    \"AND NOT EXISTS (SELECT 1 FROM image_embeddings e \"\n"
        "    \"                 WHERE e.url=i.url AND e.model='siglip-lux128')\"\n"
        ").fetchone()[0])\n",
        timeout=90,
    )
    nums = [l.strip() for l in out.splitlines() if l.strip().isdigit()]
    if len(nums) < 3:
        return True, f"{INCONCLUSIVE} could not query ({out[:60]})"
    uptime_s, n, pending = int(nums[0]), int(nums[1]), int(nums[2])

    # No embeddable images = pipeline idle, not broken. (Floorplans excluded:
    # SigLIP skips them by policy, they will sit in status='done' forever, and
    # counting them as pending would make this always show work to do.)
    if pending == 0:
        return True, f"{n} new in the last {WINDOW_MIN} min; no images pending (pipeline idle, not a fault)"

    # Right after boot this window is necessarily 0 — the machine hasn't been up
    # WINDOW_MIN minutes yet; the pipeline isn't broken. Post-reboot acceptance is
    # exactly when this script is run most, so it must step aside here, or every
    # reboot would false-alarm.
    if uptime_s < WINDOW_MIN * 60:
        return True, (f"{INCONCLUSIVE} up only {uptime_s // 60} min, less than the {WINDOW_MIN}-min window"
                      f" ({n} produced so far); check again in a few minutes")
    # Work pending but zero output — that is a real stall.
    return (n > 0), f"{n} new embeddings in the last {WINDOW_MIN} min ({pending} images pending)"


CHECKS = [
    ("reachable",     check_reachable),
    ("driver_match",  check_driver_match),
    ("cuda_new_proc", check_cuda_new_proc),
    ("units",         check_units),
    ("embed_progress", check_embed_progress),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true", help="print failed checks only")
    args = ap.parse_args()

    failed = []
    for name, fn in CHECKS:
        ok, detail = fn()
        if not ok:
            failed.append(name)
        if not args.quiet or not ok:
            print(f"{'✅' if ok else '❌'} {name:<16} {detail}")
        if name == "reachable" and not ok:
            print("   Host unreachable, skipping the remaining checks.")
            return 1

    if failed:
        print(f"\n{len(failed)} check(s) failed: {', '.join(failed)}")
        return 1
    if not args.quiet:
        print("\nAll passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
