#!/bin/bash
# pyrun.sh — robust Python launcher for launchd jobs on this Mac.
#
# ⚠️ INSTALL LOCATION MATTERS. launchd CANNOT exec a shell script that lives
#    under ~/Documents (TCC-protected: bash gets "Operation not permitted"
#    opening the script). The production checkout is under ~/Documents, so the
#    plists do NOT point at scripts/pyrun.sh directly — they point at an
#    installed copy in an unprotected dir. Keep them in sync after editing this
#    file:
#
#        cp scripts/pyrun.sh /usr/local/bin/wtl-pyrun.sh && chmod +x /usr/local/bin/wtl-pyrun.sh
#
#    (The interpreter it execs — venv/bin/python3, a symlink into /opt/homebrew
#    — is fine for launchd to exec from Documents; only *scripts* are blocked.)
#
# WHY THIS EXISTS
#   Under memory-pressure bursts this machine intermittently kills a *freshly
#   spawned* Python interpreter during startup with:
#
#       Exception ignored while running getpath:
#       InterruptedError: [Errno 4] Interrupted system call   (<frozen getpath>)
#       Fatal Python error: error evaluating path
#
#   A signal interrupts one of getpath's startup syscalls (stat/readlink while
#   computing sys.path). getpath runs BEFORE main() and BEFORE PEP 475's
#   automatic EINTR-retry is active, so the EINTR is unhandled and fatal.
#   It is transient (clears in seconds) and happens before any user code runs,
#   so re-launching the interpreter is completely idempotent.
#
# WHAT IT DOES
#   Probe the interpreter with `-c ''` (startup only, no side effects) until it
#   starts cleanly, then exec the real target EXACTLY ONCE. Because we only ever
#   run the real script after a clean probe — and via exec, never in a loop —
#   the script's own side effects can never repeat. exec also preserves the PID
#   launchd monitors, so KeepAlive/StartInterval semantics are unchanged.
#
# USAGE (from a launchd plist ProgramArguments):
#   <string>/…/scripts/pyrun.sh</string>
#   <string>/…/venv/bin/python3</string>
#   <string>/…/scripts/your_script.py</string>
#   <string>--your</string><string>--args</string>
#
# Added 2026-07-01 after intermittent getpath crashes silently stalled a
# monitoring job and dropped an hourly job's run record (false alert).
set -u

PYBIN="${1:?pyrun.sh: missing python binary as first argument}"
shift

# Retry the startup probe with backoff (total ~30s worst case). The getpath
# window is milliseconds wide, so a successful probe means the next exec of the
# same interpreter will almost certainly clear startup too.
for delay in 0 1 2 4 8 15; do
    [ "$delay" -gt 0 ] && sleep "$delay"
    if "$PYBIN" -c '' 2>/dev/null; then
        exec "$PYBIN" "$@"
    fi
done

# Still can't start after all retries — exec anyway so the real error surfaces
# in the job's log rather than being silently swallowed by this wrapper.
exec "$PYBIN" "$@"
