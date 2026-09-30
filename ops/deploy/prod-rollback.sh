#!/usr/bin/env bash
# Roll back next-prod to a previously-tagged deploy. Lists the 10 most
# recent deploy-* tags if no arg given.
#
# Usage:
#   scripts/prod-rollback.sh                   # interactive — pick from list
#   scripts/prod-rollback.sh deploy-2026-05-24-1212
#
# What it does:
#   1. Stashes any uncommitted changes (you can pop them after)
#   2. git checkout <tag>           (detached HEAD)
#   3. npm run build
#   4. launchctl restart next-prod
#
# To get back to where you were:
#   git switch -                    # back to the previous branch
#   git stash pop                   # restore your uncommitted changes
#
# Run from anywhere — script cd's to the repo root.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEB_DIR="$REPO_ROOT/web"

cd "$REPO_ROOT"

# --- 1. Resolve target tag --------------------------------------------
TARGET="${1:-}"
if [ -z "$TARGET" ]; then
  echo "Recent deploy tags (newest first):"
  echo ""
  git for-each-ref --sort=-creatordate --format='%(refname:short)  %(creatordate:short)  %(contents:subject)' refs/tags/deploy-* | head -10
  echo ""
  read -rp "Tag to roll back to: " TARGET
fi

if ! git rev-parse --verify "refs/tags/$TARGET" >/dev/null 2>&1; then
  echo "ERROR: tag '$TARGET' not found" >&2
  exit 1
fi

# --- 2. Sanity: refuse if working tree is dirty unless --force --------
STASHED=0
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Working tree has uncommitted changes. Stashing them temporarily."
  git stash push -u -m "prod-rollback to $TARGET"
  STASHED=1
fi

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo ""
echo "─── rolling back ──────────────────────────────────"
echo "from:  $CURRENT_BRANCH ($(git rev-parse --short HEAD))"
echo "to:    $TARGET"
echo ""

# --- 3. Checkout the tag ----------------------------------------------
echo "[1/3] git checkout ${TARGET}…"
git checkout "$TARGET"
echo ""

# --- 4. Build ----------------------------------------------------------
echo "[2/3] building…"
cd "$WEB_DIR"
if ! npm run build 2>&1 | tail -25; then
  echo ""
  echo "ERROR: build failed at $TARGET — restoring previous state" >&2
  cd "$REPO_ROOT"
  git switch -
  [ "$STASHED" = "1" ] && git stash pop || true
  exit 1
fi
cd "$REPO_ROOT"

# --- 4b. Retain previous builds' static chunks (same as prod-deploy) ----
# Merge archived chunk files from previous builds back into .next/static
# so browsers holding older HTML don't hit ChunkLoadError after the
# rollback rebuild. See prod-deploy.sh step 2b for the full rationale.
ARCHIVE_DIR="$WEB_DIR/.next-static-archive"
mkdir -p "$ARCHIVE_DIR"
rsync -a "$WEB_DIR/.next/static/" "$ARCHIVE_DIR/"
find "$ARCHIVE_DIR" -type f -mtime +14 -delete
find "$ARCHIVE_DIR" -type d -empty -delete 2>/dev/null || true
rsync -a --ignore-existing "$ARCHIVE_DIR/" "$WEB_DIR/.next/static/"

# --- 5. Restart next-prod ----------------------------------------------
echo ""
echo "[3/4] restarting next-prod…"
launchctl kickstart -k "gui/$(id -u)/xyz.wheretolive.next-prod"
sleep 4
status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/ || echo "??")
echo "  / responded HTTP $status"

# --- 6. Warm the route cache (same as prod-deploy) -------------------
echo ""
echo "[4/4] warming route cache…"
WARM_ROUTES=("/" "/admin" "/properties/value" "/check" "/chat" "/explore" "/evaluate" "/image-search" "/compare-input" "/property-compare" "/map" "/radars")
WARM_PIDS=()
for route in "${WARM_ROUTES[@]}"; do
  (
    code=$(curl -s -o /dev/null -m 30 -w "%{http_code} %{time_total}s" "http://localhost:3000${route}" || echo "?? ??")
    printf "  %-22s %s\n" "$route" "$code"
  ) &
  WARM_PIDS+=($!)
done
for pid in "${WARM_PIDS[@]}"; do wait "$pid"; done

# Warm the admin skills API (admin-key header; first hit cold-opens the 700MB ops.db)
ADMIN_KEY_WARM=$(grep -h '^ADMIN_KEY=' "$REPO_ROOT/web/.env.local" "$REPO_ROOT/web/.env" 2>/dev/null | head -1 | cut -d= -f2-)
if [ -n "$ADMIN_KEY_WARM" ]; then
  code=$(curl -s -o /dev/null -m 30 -H "x-admin-key: $ADMIN_KEY_WARM" -w "%{http_code} %{time_total}s" "http://localhost:3000/api/admin/skills?list=1" || echo "?? ??")
  printf "  %-22s %s\n" "/api/admin/skills" "$code"
fi

echo ""
echo "─── rolled back to $TARGET ────────────────────────────"
echo ""
echo "You are on DETACHED HEAD. To return to your branch:"
echo "  git switch $CURRENT_BRANCH"
[ "$STASHED" = "1" ] && echo "  git stash pop                # restore your uncommitted changes"
echo ""
echo "To re-deploy current main after fixing forward:"
echo "  git switch $CURRENT_BRANCH"
echo "  scripts/prod-deploy.sh"
