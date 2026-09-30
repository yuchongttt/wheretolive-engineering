#!/usr/bin/env bash
# Build the Next.js prod bundle, tag the git revision, then restart
# next-prod. Tag format: deploy-YYYY-MM-DD-HHMM (UTC). Every successful
# deploy becomes a rollback point — see prod-rollback.sh.
#
# Usage:
#   scripts/prod-deploy.sh              # uses HEAD commit
#   scripts/prod-deploy.sh "fix nav"    # adds a short note to the tag
#
# Run from anywhere — script cd's to the repo root.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEB_DIR="$REPO_ROOT/web"
NOTE="${1:-}"
TAG="deploy-$(date -u +%Y-%m-%d-%H%M)"

cd "$REPO_ROOT"

# --- 0. Deploy lock: only ONE deploy at a time -------------------------
# wheretolive.xyz is ONE server with ONE build folder (.next). Two
# concurrent deploys write that folder at the same time → a scrambled
# half-A/half-B build → ChunkLoadError / 500s across the site
# until someone rebuilds cleanly. This atomic-mkdir lock makes concurrent
# deploys impossible: the second one aborts here with a clear message
# instead of corrupting prod. Every working session on the host runs as the
# same macOS user, and /tmp is shared, so this lock is seen across sessions.
LOCK_DIR="/tmp/wtl-prod-deploy.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  HOLDER="$(cat "$LOCK_DIR/info" 2>/dev/null || echo 'unknown')"
  HOLDER_PID="$(cat "$LOCK_DIR/pid" 2>/dev/null || echo '')"
  # Reclaim a stale lock whose owner process died mid-deploy (crash/Ctrl-C
  # before the trap could fire, e.g. SIGKILL).
  if [ -n "$HOLDER_PID" ] && ! kill -0 "$HOLDER_PID" 2>/dev/null; then
    echo "WARN: stale deploy lock from dead PID $HOLDER_PID — reclaiming." >&2
    rm -rf "$LOCK_DIR"
    mkdir "$LOCK_DIR" 2>/dev/null || { echo "ERROR: could not reclaim deploy lock — aborting." >&2; exit 1; }
  else
    echo "ERROR: another deploy is already running — refusing to start." >&2
    echo "  holder:  $HOLDER" >&2
    echo "  Two deploys at once corrupt prod. Wait for it to" >&2
    echo "  finish, then deploy. If you are CERTAIN no deploy is running:" >&2
    echo "    rm -rf '$LOCK_DIR'" >&2
    exit 1
  fi
fi
echo "$$" > "$LOCK_DIR/pid"
echo "PID $$ | $(whoami) | started $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCK_DIR/info"
# Release the lock however we exit: success, build failure, or Ctrl-C — and
# journal the attempt for the admin Deployments card (deploy-history.jsonl).
# Journaling is best-effort: it must never change the exit code or block
# lock cleanup. Installed only after lock acquisition, so lock-busy aborts
# are not journaled (they never started a deploy).
DEPLOY_START_TS=$(date +%s)
DEPLOY_START_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)
on_exit() {
  code=$?
  {
    result=ok
    if [ "$code" -ne 0 ]; then result=failed; fi
    j_dpl="$(cat "$WEB_DIR/.next/BUILD_ID" 2>/dev/null || echo unknown)"
    j_sha="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    mkdir -p "$REPO_ROOT/logs"
    printf '{"startedAt":"%s","finishedAt":"%s","durationSec":%s,"result":"%s","exitCode":%s,"tag":"%s","dpl":"%s","gitSha":"%s"}\n' \
      "$DEPLOY_START_ISO" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(( $(date +%s) - DEPLOY_START_TS ))" \
      "$result" "$code" "$TAG" "$j_dpl" "$j_sha" \
      >> "$REPO_ROOT/logs/deploy-history.jsonl"
  } 2>/dev/null || true
  rm -rf "$LOCK_DIR"
  exit "$code"
}
trap on_exit EXIT
# Async-signal correctness: if SIGINT/SIGTERM lands between statements, the
# EXIT trap's $? would read the PREVIOUS command's status (usually 0) and
# `exit "$code"` would launder an aborted deploy into a journaled success.
# Explicit signal traps make the exit status deterministic (130/143).
trap 'exit 130' INT
trap 'exit 143' TERM

# --- 1. Sanity: HEAD commit + working tree state -----------------------
HEAD_SHA="$(git rev-parse --short HEAD)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
DIRTY=""
if ! git diff --quiet || ! git diff --cached --quiet; then
  DIRTY=" (DIRTY — uncommitted changes will NOT be in the tagged commit)"
fi

echo "─── prod deploy ────────────────────────────────────"
echo "branch:  $BRANCH"
echo "commit:  $HEAD_SHA$DIRTY"
echo "tag:     $TAG"
[ -n "$NOTE" ] && echo "note:    $NOTE"
echo ""

# --- Policy: prod deploys ONLY from main -------------------------------
# Forward deploys must build 'main'. Develop on branches, merge to main,
# deploy main. (2026-06-18: accidental feature-branch deploys put two
# features' WIP on prod and reverted each other — this guard prevents that.)
# prod-rollback.sh is the sanctioned exception; it checks
# out a known-good deploy-* tag, not a branch.
if [ "$BRANCH" != "main" ] && [ "${WTL_DEPLOY_ALLOW_NON_MAIN:-}" != "1" ]; then
  echo "ERROR: prod deploys only from 'main' — you are on '$BRANCH'." >&2
  echo "  Develop on a branch, then:  git checkout main && git merge $BRANCH" >&2
  echo "  Rare override:              WTL_DEPLOY_ALLOW_NON_MAIN=1 scripts/prod-deploy.sh" >&2
  exit 1
fi

# --- 2. Build ----------------------------------------------------------
echo "[1/3] building…"
cd "$WEB_DIR"
if ! npm run build 2>&1 | tail -25; then
  echo ""
  echo "ERROR: build failed — aborting (next-prod not restarted, no tag)" >&2
  exit 1
fi
cd "$REPO_ROOT"

# --- 2b. Retain previous builds' static chunks -------------------------
# Root cause of the recurring post-deploy ChunkLoadError: `npm run build`
# rewrites .next/ in place, DELETING the previous build's content-hashed
# chunk files. Any browser still holding the old HTML (stale cache, or
# simply a tab that was open during the deploy) then requests a chunk
# hash that no longer exists → 404 → the client component never loads and
# its buttons/navigation silently die. no-cache HTML headers + the
# ChunkErrorReloader only mitigate AFTER the error; this step removes the
# error: archive every build's static assets and merge previous builds'
# files back into .next/static, so old chunk URLs keep resolving for
# ~30 days after a deploy (raised from 14 on 2026-07-16: Googlebot caches
# resource URLs for up to ~30 days, so shorter windows still surfaced
# 404s in Search Console). Safe because filenames are content-hashed
# (immutable — same name ⇒ same bytes) and per-build files live under a
# unique buildId directory, so merging can never overwrite new files
# with stale content (`--ignore-existing` guarantees it besides).
echo ""
echo "[1b] retaining previous builds' chunks…"
ARCHIVE_DIR="$WEB_DIR/.next-static-archive"
mkdir -p "$ARCHIVE_DIR"
rsync -a "$WEB_DIR/.next/static/" "$ARCHIVE_DIR/"          # new build → archive
find "$ARCHIVE_DIR" -type f -mtime +30 -delete             # drop chunks >30 days old (Googlebot cache horizon)
find "$ARCHIVE_DIR" -type d -empty -delete 2>/dev/null || true
rsync -a --ignore-existing "$ARCHIVE_DIR/" "$WEB_DIR/.next/static/"  # old chunks → live dir
echo "  archive: $(find "$ARCHIVE_DIR" -type f | wc -l | tr -d ' ') files, $(du -sh "$ARCHIVE_DIR" | cut -f1)"

# --- 3. Tag the commit -------------------------------------------------
echo ""
echo "[2/3] tagging ${TAG}…"
TAG_MSG="prod deploy $(date -u +%Y-%m-%dT%H:%M:%SZ) | branch=$BRANCH | commit=$HEAD_SHA"
[ -n "$NOTE" ] && TAG_MSG="$TAG_MSG | $NOTE"
git tag -a "$TAG" -m "$TAG_MSG"
echo "  tag created: $TAG"

# --- 4. Restart next-prod ----------------------------------------------
echo ""
echo "[3/4] restarting next-prod…"
# Real-restart assertion (2026-07-13): a 200 alone can't prove the restart
# took — an old process serves new lazy-loaded chunks convincingly (today's
# incident: dpl hash lagged a build behind while / returned 200). Require
# the :3000 listener PID to actually change, else fail the deploy loudly.
old_pid=$(lsof -tiTCP:3000 -sTCP:LISTEN 2>/dev/null | head -1 || true)
launchctl kickstart -k "gui/$(id -u)/xyz.wheretolive.next-prod"
sleep 4
status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/ || echo "??")
echo "  / responded HTTP $status"
new_pid=$(lsof -tiTCP:3000 -sTCP:LISTEN 2>/dev/null | head -1 || true)
if [ -n "$old_pid" ] && [ "$new_pid" = "$old_pid" ]; then
  echo "ERROR: :3000 listener PID unchanged ($old_pid) — kickstart did NOT replace next-prod;" >&2
  echo "       the old process may still serve stale code. Investigate before trusting this deploy." >&2
  exit 1
fi
echo "  listener replaced: ${old_pid:-none} → ${new_pid:-pending}"

# --- 4b. Deploy breadcrumb for the admin Deployments card ----------------
# Read by /api/admin/deploy-status (Site Health tab). Best-effort: a
# breadcrumb failure must never fail a verified deploy.
{
  BC_DPL="$(cat "$WEB_DIR/.next/BUILD_ID" 2>/dev/null || echo unknown)"
  BC_TAGS="$(git -C "$REPO_ROOT" tag -l 'deploy-*' | sort | tail -5 | sed 's/.*/"&"/' | paste -sd, -)"
  mkdir -p "$REPO_ROOT/logs"
  cat > "$REPO_ROOT/logs/deploy-status.json" <<BREADCRUMB
{
  "service": "wheretolive",
  "tag": "$TAG",
  "dpl": "$BC_DPL",
  "gitSha": "$(git -C "$REPO_ROOT" rev-parse --short HEAD)",
  "deployedAt": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "recentTags": [$BC_TAGS]
}
BREADCRUMB
  echo "  breadcrumb: logs/deploy-status.json (dpl=$BC_DPL)"
} || true

# --- 5. Warm the route cache ------------------------------------------
# After restart, Next.js prod does a one-time per-route compile +
# chunk-load-into-memory on the FIRST request. Cold /admin was 15s
# (DOM 7s + chunks 8s) vs ~2s warm. Pre-hit the heavy routes here so
# the first real visitor always lands on a warm cache. Parallelized —
# adds ~3-5s to the deploy step instead of 30s+ first-user pain.
echo ""
echo "[4/4] warming route cache…"
WARM_ROUTES=(
  "/"
  "/admin"
  "/properties/value"
  "/check"
  "/chat"
  "/explore"
  "/evaluate"
  "/image-search"
  "/compare-input"
  "/property-compare"
  "/map"
  "/radars"
)
WARM_PIDS=()
for route in "${WARM_ROUTES[@]}"; do
  (
    code=$(curl -s -o /dev/null -m 30 -w "%{http_code} %{time_total}s" "http://localhost:3000${route}" || echo "?? ??")
    printf "  %-22s %s\n" "$route" "$code"
  ) &
  WARM_PIDS+=($!)
done
for pid in "${WARM_PIDS[@]}"; do wait "$pid"; done

# Warm the admin skills API too — needs the admin key header, and its first hit
# cold-opens the 700MB ops.db (was a 6-10s stall for whoever opened admin first).
# `|| true`: grep exits 2 when one of the env files is missing (web/.env
# usually is) and 1 on no match — under set -e -o pipefail either would
# silently kill the whole deploy here. No key just means skip this warm.
ADMIN_KEY_WARM=$(grep -h '^ADMIN_KEY=' "$REPO_ROOT/web/.env.local" "$REPO_ROOT/web/.env" 2>/dev/null | head -1 | cut -d= -f2- || true)
if [ -n "$ADMIN_KEY_WARM" ]; then
  code=$(curl -s -o /dev/null -m 30 -H "x-admin-key: $ADMIN_KEY_WARM" -w "%{http_code} %{time_total}s" "http://localhost:3000/api/admin/skills?list=1" || echo "?? ??")
  printf "  %-22s %s\n" "/api/admin/skills" "$code"
fi

# --- 5. Push to origin: offsite code backup -----------------------------
# The prod host and the GPU box sit in the same building, so GitHub is the only
# offsite copy of the code (we once sat 196 commits / a week unpushed).
# Non-fatal by design: the deploy itself already succeeded, and a dead
# network or GitHub outage must not mark it failed. --follow-tags brings
# the annotated deploy-* tags along so rollback points exist offsite too.
echo ""
echo "[5/5] pushing to origin…"
if git push origin main --follow-tags 2>&1 | tail -2; then
  echo "  pushed main + tags to origin"
else
  echo "  ⚠ push failed (network/auth?) — deploy is fine; push manually: git push origin main --follow-tags"
fi

echo ""
echo "─── done — tagged $TAG ────────────────────────────────"
echo "rollback with: scripts/prod-rollback.sh $TAG"
