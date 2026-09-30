#!/usr/bin/env python3
"""GPU-box GPU watchdog: completely silent normally, sends one Telegram message
only on a **confirmed** fault.

Why it exists: the 2026-07-24 driver upgrade broke CUDA on the GPU box (kernel
module and userspace library versions out of sync), and the existing monitoring
— monitor_checks.py only had host-reachability, cloud-worker and hourly-job
checks — had no GPU dimension at all. The GPU box's gpu field in the admin page turning null
did not trigger any alert either. So the fault stayed silent for a full 3 days,
during which wtl-style-score crash-restarted 14536 times burning CPU, and four ML
services survived only because "they started before the upgrade".

Alerting follows this repo's convention (see backup_freshness_check.py):
**silent on success, report only real problems**. Concretely, only "confirmed
broken" pages on Telegram:

  confirmed -> report
               version mismatch / a new process measurably cannot get the GPU /
               a unit in a crash-restart loop. None of these self-heal, and none
               can be misjudged because of network jitter, so a single hit is
               enough to report.
  could not determine -> don't report
               ssh unreachable, command timeout, probe didn't run — the
               healthcheck tags these with the INCONCLUSIVE prefix and they are
               skipped here. The host may simply be off or offline; that is not
               the GPU alert's business (host_reachable covers it).

So there is no "only after N consecutive hits" throttling: the criteria
themselves exclude transient noise.

Usage:
    python3 mac/gpu_watchdog.py           # called on a launchd schedule
    python3 mac/gpu_watchdog.py --dry-run # print only, don't send
Exit code 0 = no alert (including "could not determine"); 1 = alerted.

Public-repo note: the private version shells out to a Telegram notifier script
that is not included here; the notifier command now comes from WTL_NOTIFY_CMD
(the message is appended as the last argument). If it is unset the alert is
only printed.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_box_healthcheck as hc  # noqa: E402

NOTIFY = os.environ.get("WTL_NOTIFY_CMD", "")

# Only failures of these checks are worth waking someone up. reachable is not
# on the list — the host going offline is not a GPU problem.
ALERT_ON = [
    ("driver_match", hc.check_driver_match),
    ("cuda_new_proc", hc.check_cuda_new_proc),
    ("units", hc.check_units),
]


def main() -> int:
    dry = "--dry-run" in sys.argv

    ok, detail = hc.check_reachable()
    if not ok:
        print(f"gpu watchdog: GPU box unreachable, skipping ({detail[:80]})")
        return 0

    problems = []
    for name, fn in ALERT_ON:
        passed, detail = fn()
        if passed:
            continue
        if detail.startswith(hc.INCONCLUSIVE):
            print(f"gpu watchdog: {name} could not be determined, not reporting — {detail}")
            continue
        problems.append(f"{name}: {detail}")

    if not problems:
        print("gpu watchdog: GPU OK")
        return 0

    msg = ("⚠️ GPU box GPU fault\n" + "\n".join(f"· {p}" for p in problems)
           + "\n\nVerify: python3 mac/gpu_box_healthcheck.py"
           + "\nThe fix for a version mismatch is to reboot the GPU box.")
    print(msg, file=sys.stderr)
    if dry or not NOTIFY:
        print("(--dry-run or WTL_NOTIFY_CMD unset: not sent)")
        return 1
    try:
        subprocess.run(shlex.split(NOTIFY) + [msg], timeout=30)
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"gpu watchdog: notify failed ({e!r})", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
