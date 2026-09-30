#!/bin/bash
# Planned-maintenance window for the 3090 VLM (silences the pipeline monitor's
# "floorplan-vlm is down" Telegram alert during deliberate GPU handovers).
#
#   vlm-maintenance.sh start [hours]   # default 2h; extend by re-running
#   vlm-maintenance.sh stop            # clear the window (call after wtl-vlm restart)
#   vlm-maintenance.sh status
#
# Safety: silence lasts only as long as the VLM is genuinely down. Two ways it
# ends even if you forget `stop`: (1) the flag holds an absolute expiry epoch —
# on lapse the normal down alert fires; and (2) the monitor clears the flag
# itself the moment the VLM probes healthy again. A forgotten restore is NOT
# silenced forever.
#
# WTL_VLM_URL: base URL of the vLLM server on the VLM box (default http://vlm-box:8200).
set -euo pipefail
VLM_URL="${WTL_VLM_URL:-http://vlm-box:8200}"
FLAG="$(cd "$(dirname "$0")/.." && pwd)/data/vlm_maintenance_until"

case "${1:-status}" in
  start)
    HOURS="${2:-2}"
    UNTIL=$(( $(date +%s) + HOURS * 3600 ))
    echo "$UNTIL" > "$FLAG"
    echo "maintenance window ON until $(date -r "$UNTIL" '+%H:%M %Z') (${HOURS}h)"
    ;;
  stop)
    # After a restart vLLM takes ~2min to load the 27B model, during which
    # /health is down; clearing the flag immediately makes the monitor hit that
    # gap and false-alarm (observed 2026-07-02 15:49). By default wait until
    # healthy before clearing; --force skips the wait.
    if [ "${2:-}" != "--force" ]; then
      echo "waiting for vLLM /health (max 5min)..."
      n=0
      until curl -s -m 3 "$VLM_URL/health" >/dev/null 2>&1; do
        n=$((n+1))
        if [ $n -ge 30 ]; then
          echo "WARN: vLLM not healthy after 5min — flag kept (window will self-expire). Use 'stop --force' to clear now."
          exit 1
        fi
        sleep 10
      done
      echo "vLLM healthy."
    fi
    rm -f "$FLAG"
    echo "maintenance window OFF"
    ;;
  status)
    if [ -f "$FLAG" ]; then
      UNTIL=$(cat "$FLAG")
      NOW=$(date +%s)
      if [ "$NOW" -lt "$UNTIL" ]; then
        echo "ACTIVE until $(date -r "$UNTIL" '+%H:%M %Z') ($(( (UNTIL - NOW) / 60 )) min left)"
      else
        echo "EXPIRED at $(date -r "$UNTIL" '+%H:%M %Z') (flag stale, alerts live)"
      fi
    else
      echo "OFF (no window)"
    fi
    ;;
  *)
    echo "usage: $0 start [hours] | stop | status" >&2
    exit 1
    ;;
esac
