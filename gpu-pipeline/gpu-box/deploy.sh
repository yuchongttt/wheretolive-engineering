#!/usr/bin/env bash
# Deploy gpu-box/ + systemd/ to the GPU box. Source-of-truth is this repo,
# not whatever is currently sitting in /data/ml/dataset/ —
# always edit here, then run this script. The script rsyncs each
# subtree to its real on-host path and drops + reloads systemd units.
# It never restarts services: code can be hot-deployed under a running
# daemon, and long-loading model services should be recycled deliberately.
#
# Usage:
#   ./deploy.sh              # dry-run, prints diff
#   ./deploy.sh --apply      # actually push + daemon-reload
#   ./deploy.sh --apply ml   # only push the python services (no systemd reload)
#   ./deploy.sh --apply systemd  # only push systemd units + daemon-reload
#
# Pre-reqs:
#   - GPU_HOST=<user>@<host> of the GPU box, reachable over the private network
#   - The user on the remote side has sudo nopasswd for `cp /etc/systemd/system/`
#     and `systemctl daemon-reload`. If not, the script will prompt.

set -euo pipefail

HOST="${GPU_HOST:?set GPU_HOST=<user>@<host> of the GPU box}"
HERE="$(cd "$(dirname "$0")" && pwd)"
MODE="dryrun"
SCOPE="all"

for arg in "$@"; do
  case "$arg" in
    --apply) MODE="apply" ;;
    ml|systemd|home|all) SCOPE="$arg" ;;
    *) echo "unknown arg: $arg"; exit 2 ;;
  esac
done

rsync_opts=(-av)
[ "$MODE" = "dryrun" ] && rsync_opts+=(--dry-run)

push_ml() {
  echo "=== gpu-box/*.py → /data/ml/dataset/ ==="
  rsync "${rsync_opts[@]}" \
    --include='*.py' --exclude='*' \
    "$HERE/" "$HOST:/data/ml/dataset/"
}

push_home() {
  echo "=== host_report.sh → /opt/wheretolive/ ==="
  rsync "${rsync_opts[@]}" \
    "$HERE/host_report.sh" "$HOST:/opt/wheretolive/host_report.sh"
}

push_systemd() {
  echo "=== systemd/ → /etc/systemd/system/ ==="
  # Stage to /tmp first (no sudo over SSH), then sudo-cp into place +
  # reload. This avoids needing rsync-over-sudo and keeps the diff visible.
  rsync "${rsync_opts[@]}" \
    "$HERE/../systemd/" "$HOST:/tmp/wtl-systemd-staging/"
  if [ "$MODE" = "apply" ]; then
    ssh "$HOST" '
      sudo install -m 644 /tmp/wtl-systemd-staging/wtl-*.service /etc/systemd/system/
      sudo install -m 644 /tmp/wtl-systemd-staging/wtl-*.timer   /etc/systemd/system/
      # Drop-ins (<unit>.service.d/*.conf) are the override layer. They are NOT
      # optional decoration: without dino-enable.conf DINO embedding silently
      # stops, and without offline.conf the daemon tries to reach huggingface.co
      # on a box with no route to it and crash-loops forever (it never
      # self-heals). The base unit alone is actively misleading -- it implies
      # DINO_ENABLED unset and OOMScoreAdjust=200, both overridden here.
      for d in /tmp/wtl-systemd-staging/wtl-*.service.d; do
        [ -d "$d" ] || continue
        sudo install -d -m 755 "/etc/systemd/system/$(basename "$d")"
        sudo install -m 644 "$d"/*.conf "/etc/systemd/system/$(basename "$d")/"
      done
      sudo systemctl daemon-reload
      echo "systemd reloaded — restart units manually as needed"
    '
  fi
}

case "$SCOPE" in
  all)     push_ml; push_home; push_systemd ;;
  ml)      push_ml ;;
  home)    push_home ;;
  systemd) push_systemd ;;
esac

if [ "$MODE" = "dryrun" ]; then
  echo
  echo "*** DRY RUN *** — pass --apply to actually push."
fi
