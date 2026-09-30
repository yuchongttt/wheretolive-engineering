# ops/deploy: deploying and scheduling the production host

Production is one Mac mini, with no container orchestrator. **launchd** supervises the processes
and runs the schedules. A deploy builds the site, tags the release, restarts the server and
checks the restart. A rollback rebuilds an earlier tag. This directory holds `prod-deploy.sh`,
`prod-rollback.sh`, `pyrun.sh` (the startup-retry wrapper, installed as `wtl-pyrun.sh`), 10
representative plists in `launchd/`, and `tests/`.

## Scheduling model

A plist uses one of three launchd shapes:

| Shape | Keys | Behaviour | Examples here |
|---|---|---|---|
| **Daemon** | `KeepAlive=true`, `RunAtLoad=true`, `ThrottleInterval` | always running; launchd restarts it on exit, at most once per throttle interval | `next-prod` (30 s), `cloudflared` (10 s) |
| **Interval job** | `StartInterval=N` | re-run every N seconds | `host-report` 60 s, `watchdog` 120 s, `wal-checkpoint` 300 s, `synthetic-smoke` 1800 s |
| **Calendar job** | `StartCalendarInterval` | wall-clock schedule, like cron | `db-backup` 05:20 daily, `db-retention` Sun 03:30, `logs-retention` Sun 04:50 |

The watchdog combines two of them: `StartInterval=120` plus `KeepAlive{SuccessfulExit=false}`
and `ThrottleInterval=15`. A tick that crashes is retried after 15 s. A clean exit waits for the
next interval. `tests/test_launchd_plists.py` checks that every plist is either a daemon or a
periodic job, never both.

The plists share these conventions (each is documented in a plist comment or script header):

- **The macOS privacy sandbox (TCC) is the main constraint.** The production checkout sits in a
  TCC-protected folder, where under launchd neither `/bin/bash` nor Apple's `/usr/bin/python3`
  can open a script (the latter once failed `logs-retention` with exit 2). So Python jobs run
  under the Homebrew or venv interpreter, which has Full Disk Access, and shell entry points
  (`wtl-pyrun.sh`, `host_report.sh`) are installed outside the checkout.
- **`wtl-pyrun.sh`.** Under memory pressure, freshly spawned interpreters sometimes died in
  `getpath` with `EINTR` before any user code ran. The wrapper probes the interpreter with
  `-c ''` and backs off for up to about 30 s. Then it `exec`s the real script **exactly once**.
  The job's side effects never repeat, and launchd keeps watching the same PID.
  `tests/test_shell_scripts.py` pins this behaviour.
- **launchd stdio logs live outside the checkout.** A log path inside the checkout lost its
  `com.apple.macl` attribute when the file was recreated. After that, launchd could not open it,
  and `synthetic-smoke` failed with exit 78 and no log at all for 10 days (2026-06-27 to 07-07).
- **Secrets never go in the repo copies.** Placeholders stay in git. Only the live copy in
  `~/Library/LaunchAgents` holds real values.

## Deploy: `prod-deploy.sh`

1. **Lock.** The script creates `/tmp/wtl-prod-deploy.lock` with an atomic `mkdir` and records
   its PID in it. It takes over a stale lock if the recorded PID is dead. There is one `.next`
   build folder, so two deploys running at once would leave a half-A/half-B build. An `EXIT`
   trap releases the lock and appends one JSON line to `logs/deploy-history.jsonl`. Explicit
   `INT`/`TERM` traps make the exit code deterministic (130/143), so an aborted deploy is never
   journaled as a success.
2. **Policy.** Deploys run only from `main`. The escape hatch is `WTL_DEPLOY_ALLOW_NON_MAIN=1`.
   This was added after feature-branch deploys overwrote each other on 2026-06-18.
3. **Build.** `npm run build`. If it fails, the script aborts before any restart or tag.
4. **Keep old chunks.** It archives each build's content-hashed `.next/static` files and merges
   the last 30 days back in. Browsers and crawlers holding old HTML can still load their chunks,
   so they do not hit `ChunkLoadError`. Merging cannot overwrite anything, because the filenames
   are content-hashed and the merge uses `--ignore-existing`.
5. **Tag.** It creates an annotated tag `deploy-YYYY-MM-DD-HHMM` (UTC). Every deploy becomes a
   rollback point.
6. **Restart and verify.** It runs `launchctl kickstart -k …/next-prod`, then **requires the
   :3000 listener PID to change**. An HTTP 200 alone once hid a restart that had not happened
   (2026-07-13).
7. **Breadcrumb and warm-up.** It writes `logs/deploy-status.json` for the admin Deployments
   card. Then it requests the heavy routes in parallel, so the first real visitor does not pay
   the cold-compile cost.
8. **Offsite copy.** `git push origin main --follow-tags`. This step is non-fatal: the deploy has
   already succeeded, and the push also takes the tags offsite.

## Rollback: `prod-rollback.sh`

With no argument, the script lists the 10 newest `deploy-*` tags. It stashes a dirty tree, checks
out the tag (detached HEAD) and rebuilds. If the build fails, it switches back and restores the
stash. Then it merges the chunk archive, kickstarts `next-prod` and warms the routes. Rollback
rebuilds from source instead of swapping artefacts. That keeps it simple (there is no artefact
store) at the cost of one build. The 2026-05-18 system-design doc puts a build at about 30 s.

**Known gaps (as of the snapshot):** the rollback path does not take the deploy lock, does not
write to `deploy-history.jsonl`, and does not check that the listener PID changed. It also prunes
the chunk archive at 14 days, where deploy uses 30.

## How the watchdog, monitoring, backups and retention fit

- **Serving.** launchd restarts `next-prod` and `cloudflared` if they crash. The
  [watchdog](../watchdog) never restarts `next-prod`. It restarts `cloudflared` only after two
  ticks confirm the tunnel is down.
- **Liveness.** `host-report` (every 60 s) and the watchdog heartbeat (every 120 s) post to the
  same admin host-metrics API. A checker on another host alerts when either goes stale. That
  check is what makes "no alerts" a trustworthy sign.
- **Content checks.** Every 30 min, `synthetic-smoke` loads public pages in a headless browser
  and asserts that real content renders, not just an HTTP 200. It alerts only when the set of
  failing pages changes.
- **SQLite health.** Every 5 min, `wal-checkpoint` runs a `PASSIVE` checkpoint. It keeps the WAL
  moving when long-running readers stop SQLite's own auto-checkpoint, without competing for
  locks.
- **Backups.** `db-backup` runs daily at 05:20. It takes a `VACUUM INTO` snapshot, runs
  `PRAGMA integrity_check`, rsyncs to the GPU box, verifies with a checksum dry-run, keeps 3
  dailies and writes a success marker. `backup-freshness` checks at 09:10 and alerts if that
  marker is more than 48 h old. So a backup that fails silently is still caught.
- **Retention** runs in the quiet Sunday window. `db-retention` (03:30) prunes audit and cache
  rows, and refuses any single delete that would remove more than 50 % of a table.
  `ops-retention` (04:40) keeps 45 days of high-frequency run rows. `logs-retention` (04:50)
  deletes launchd logs idle for more than 90 days.

The plists point at the production layout (`/opt/wheretolive/scripts/…`). The scripts they run
are in [`../monitoring`](../monitoring) and [`../watchdog`](../watchdog).

## Inventory of launchd jobs (snapshot 2026-09-29)

The private repo tracks 79 launchd plists. The 71 under `scripts/` and `docs/` are counted below,
by reading each file: 65 in `scripts/launchd/`, 2 in `scripts/` and 4 in `docs/launchd/`. The
other 8 belong to tooling outside this extract. These are *tracked* definitions, not necessarily
loaded ones. For example, `changelog-poll` was retired on 2026-07-27 but is still tracked.

| Category | Plists | Daemon / interval / calendar | Jobs |
|---|---:|---|---|
| Serving | 3 | 3 / 0 / 0 | next-prod, cloudflared, monitors-server |
| Monitoring & alerting | 9 | 0 / 5 / 4 | watchdog, host-report, synthetic-smoke, gpu-watchdog, prewarm-freshness, queue-probe, queue-snapshot, queue-status, claude-cost-alert |
| Backups | 2 | 0 / 0 / 2 | db-backup, backup-freshness |
| Maintenance & retention | 6 | 0 / 1 / 5 | wal-checkpoint, db-retention, ops-retention, logs-retention, chat-img-cleanup, claude-mem-retention |
| Data pipeline (derived tables, public datasets, cache warming) | 16 | 2 / 2 / 12 | build-dp-identity, build-dp-quality, cluster-full-weekly, council-layer-weekly, crime-monthly, geo-cagr-rebuild, leases-sync, market-baselines, pipr-monthly, planning-drainer, planning-enqueue, postcode-eval-daemon, prewarm-commute-hourly, home-stats, homepage-snapshot, board-reaggregate |
| ML / GPU dispatch | 8 | 3 / 3 / 2 | floorplan-vlm-daemon, floorplan-vlm-dispatcher, floorplan-north, garden-facing-derived, planning-summarize, aesthetic-board, style-tags-weekly, dispatched-reconciler |
| LLM evals & quality | 5 | 0 / 0 / 5 | bhv-canary-daily, bhv-routing, chat-weekly-audit, self-review-daily, user-sim-weekly |
| Product jobs | 4 | 1 / 0 / 3 | radar-matcher, daily-picks, changelog-check, changelog-poll |
| Developer-tooling telemetry | 2 | 0 / 0 / 2 | cc-usage-snapshot, mem-usage-snapshot |
| Data-ingestion jobs (not included) | 16 | 1 / 12 / 3 | — |
| **Total** | **71** | **10 / 23 / 38** | |

Across the 71 plists, 20 launch through `wtl-pyrun.sh` and 66 send launchd stdio to the log
directory outside the checkout.

## Paths, placeholders and environment

| In these files | Production equivalent |
|---|---|
| `/opt/wheretolive` | the app checkout (in a TCC-protected folder) |
| `/usr/local/bin/wtl-pyrun.sh` | the installed copy of `pyrun.sh`, in a user bin dir outside the checkout |
| `/var/log/wheretolive/` | the per-user launchd log directory |
| `/usr/local/lib/wheretolive/` | where the host-report deploy step copies `host_report.sh` |

Placeholders are `<set-me>` (written `&lt;set-me&gt;` in the XML). They stand in for `HOME`,
`WTL_API` (the admin API base URL), `WTL_ADMIN_KEY` (**secret**), the cloudflared tunnel name
and `db-backup`'s working directory (the service user's home). `prod-deploy.sh` reads
`WTL_DEPLOY_ALLOW_NON_MAIN`. It also reads `ADMIN_KEY` from the web app's untracked
`.env.local`, but only to warm one admin route, and it skips that step if the key is missing.

## Tests

`cd ops/deploy && python -m pytest -q` gave 56 passed at extraction time. The tests need
`pytest` and `bash`. `plutil -lint` runs only where it exists (macOS). The scripts themselves use
`bash`, `git`, `npm`, `rsync`, `lsof`, `curl` and `launchctl`.

**Changes from the private repo:**

- Paths and hosts are replaced as in the table above.
- Plist comments were translated or trimmed, and reworded so they contain no `--`, which is
  invalid inside XML comments and rejected by `plistlib`.
- Internal names were removed from three comments and one error message in `prod-deploy.sh`, and
  from two comments in `pyrun.sh`.
- Both test files are new.
- The watchdog plist comment now also describes the network rule.
