# ops/monitoring — layered observability for a one-person production system

wheretolive.xyz runs on a Mac mini in a home network (Next.js + SQLite, launchd), with
a Linux GPU box (`gpu-box`, systemd) beside it and Cloudflare Tunnel in front. One
person is on call and every alert goes to a single Telegram chat. The design aim:
**every failure mode has a detector that does not share its fate**, and the phone
only buzzes when something needs a human.

## Layers

```
                                                                ┌──────────┐
 L1  in-host (Mac mini, launchd)                                │          │
     WAL checkpoint + SQLite health · backups + freshness       │          │
     LLM spend · synthetic smoke · queue-drain watchdog ───────►│          │
     host_report.sh on every host ─► host-metrics API (Mac)     │          │
                                            ▼                   │          │
 L2  cross-host dead-man's switch (gpu-box, every 5 min)        │          │
     heartbeat_check: stale heartbeat / API unreachable ───────►│          │
                                                                │ Telegram │
 L3  external black box (VM outside the home network, hourly)   │          │
     site-monitor: Playwright + Lighthouse via public DNS       │          │
     + CF Tunnel ─► site-health ingest ─► admin dashboard       │          │
     (records history, does not page)                           │          │
                                                                │          │
 L4  edge probe (Cloudflare Worker cron, every 5 min)           │          │
     GET /api/health · 2-strike confirm · state in KV ─────────►│          │
                                                                └──────────┘
```

| Layer | Catches | Blind to |
|---|---|---|
| **L1 in-host** | Internal state invisible from outside: WAL checkpoint stalls and pinned readers, backups that stopped landing, daily LLM spend, a queue that silently stopped draining, a clobbered launchd plist that would only fail at next reboot, pages that return 200 but never render data | Anything that kills the host — it dies with it |
| **L2 heartbeat** | A host (or the self-healing watchdog) going dark; "cannot reach the Mac's API at all" is itself the Mac-down signal | The public path (DNS, tunnel, rendering) |
| **L3 external** | What a visitor gets through public DNS + Cloudflare Tunnel: tunnel/DNS failures, SSR errors, console errors, missing SEO meta, Lighthouse regressions | Records to a dashboard only; results queue locally while the Mac is unreachable |
| **L4 edge** | Site down as seen from Cloudflare — keeps working when every home host and the home link are down | Anything `/api/health` doesn't check; "degraded" is deliberately not paged |

Strictly, L2 is the dead-man's switch (alerts on the *absence* of a signal); L4 is an
active poller, and the last line of defence because it shares nothing with the house.

## Files

| File | Where / when (launchd plists, systemd units, docstrings; 2026-09-29) | What |
|---|---|---|
| `host_report.sh` | Mac + Linux hosts, every minute | CPU/mem/disk/IO/temp/GPU snapshot → host-metrics API; doubles as the heartbeat |
| `heartbeat_check.py` | gpu-box, every 5 min | L2 — see de-duplication below |
| `monitor_checks.py` | imported by the 5-min in-host monitor and the self-healing watchdog (neither is in this module) | `tg()` with `[ALERT]` prefix, ping, SQLite WAL health helpers |
| `wal_checkpoint.py` | Mac, every 5 min | `PASSIVE` checkpoint; `TRUNCATE` once a WAL passes 64 MB; adopts any `data/*.db` whose WAL passes 8 MB |
| `synthetic_smoke.mjs` | Mac, every 30 min | headless Chromium waits for real content text on public pages; `plutil -lint` on launchd plists |
| `claude_cost_alert.py` | Mac, hourly | global daily LLM (Claude) spend ceiling; alert-only |
| `prewarm_freshness_check.py` | Mac, daily | "work pending **and** no successful run for 48 h" watchdog for a cache-warming job |
| `backup_dbs.py` / `backup_freshness_check.py` | Mac, daily 05:20 / 09:10 | verified backup to gpu-box / independent staleness check |
| `nightly_db_retention.py`, `ops_retention.py`, `launchd_logs_retention.py` | Mac, weekly (Sun) | guarded retention sweeps |
| `wtl_tg.py`, `notify_telegram.py` | — | the single Telegram sender + a CLI wrapper for non-Python callers |
| `site-monitor/` | external VM, hourly at :05 | L3 (own README) |
| `cf-worker/` | Cloudflare edge, `*/5` cron | L4 (own README) |

## Alert de-duplication: page on transitions, not on states

Each checker keeps a small state record between runs and alerts only when it *changes*:

- **heartbeat_check** keeps the set of hosts already flagged down. New alerts =
  `down − prev_down`, recoveries = `prev_down − down`. Mac down suppresses the
  Mac-watchdog page (one actionable alert) and can't yield a false "recovered". If the
  host-metrics API is unreachable, the Mac is marked down and other hosts carry over as
  *unknown*. That branch exists because on 2026-08-27 the Mac's DNS/Tailscale was down
  for 33.5 h and the old checker exited with an error every 5 min without a word
  (docstring + regression tests).
- **cf-worker** keeps `last_status`, `down_streak` and `last_alert_ts` in Workers KV.
  "Down" needs 2 consecutive failed probes (rides out ~1 s restarts); it re-alerts every
  30 min while down and sends one `[RECOVERED]`. KV is written only on change, and the
  streak is capped at threshold + 1: the free tier allows 1,000 writes/day and
  unconditional writes had already hit a 50 % usage warning (code comment, 2026-05-20).
- **synthetic_smoke** alerts only when the sorted set of failing checks changes (a new
  failure or a recovery). An always-red check was removed rather than tolerated: a
  monitor that is permanently red trains you to ignore it.
- **claude_cost_alert** alerts once per integer multiple of the threshold per UTC day
  (`bucket = floor(spend / threshold)`), so a runaway day pages at $T, $2T, $3T…
- **SQLite health** uses tick counters: a stall needs 6 five-minute ticks in which the
  WAL grew ≥ 50k frames with zero backfill progress; a WAL ≥ 4 GB must persist 3
  consecutive ticks, because the weekly VACUUM pushes the whole DB through the WAL
  for a few minutes. `pinned_reader_pid()` names the reader via `F_GETLK` on the `-shm`
  file (macOS `lsof` can't show byte-range locks). Written after a daemon pinned a
  snapshot for 34 h and the WAL reached 11.5 GB (2026-09-05, code comment).

Telegram prefixes are a triage contract: `[ALERT]` = broken, look now; `[TODO]` = needs
a decision; `[DAILY]/[WEEKLY]` = digests; silence means healthy. `send_telegram` falls
back to plain text on a Markdown 400 so an unbalanced `_` can't eat an alert, and it
raises (never silently no-ops) when credentials are missing.

## Backups and freshness verification

`backup_dbs.py` (allow-listed SQLite DBs, Mac → gpu-box):

1. `VACUUM INTO` a staging copy — a consistent, WAL-safe snapshot of a live DB.
2. `PRAGMA integrity_check` on the snapshot; any failure aborts the run.
3. `rsync` to `daily/<date>.partial`, then `rsync --checksum --dry-run --itemize-changes`;
   any difference aborts.
4. Atomic `mv` to `daily/<date>`, optional weekly hard-link copy, prune to N dailies.
5. Only then write the success marker; a status JSON feeds the admin panel.

Guards: an `mkdir` lock with stale-lock breaking; an allow/deny list with a warning
for any DB in neither, so a new database can't silently escape the backup set.
It is Python rather than bash because launchd's `/bin/bash` and `sqlite3` lack macOS
TCC access to `~/Documents`, while the venv's interpreter has it.

A backup job that doesn't run can't report that it didn't run, so freshness is a
**separate** job reading the marker. A single failure is logged but not paged; the
watchdog threshold is 48 h so one missed night (~28 h) is tolerated and two (~52 h)
alerts. Trade-off: verification is integrity + transfer checksum; there is no
automated restore drill in this module.

## Retention

All sweeps share one pattern: a predicate that structurally can't match what must be
kept, a refusal threshold for typo'd parameters, and VACUUM only when it pays.

- `nightly_db_retention.py` — dim cache rows not on the current `CACHE_VERSION`
  (refuses if > 50 % of a table would go), audit rows > 30 d, chats purged 7 d after
  the user soft-deleted them, orphaned rows. VACUUM (exclusive lock on a multi-GB
  file) only if rows were deleted or the freelist > 5 %.
- `ops_retention.py` — the run ledger keeps 45 d for high-frequency jobs only (> 5,000
  runs all-time); everything else is kept forever. Refuses `retain_days < 45`, asserts
  after the delete that no row inside the window disappeared, VACUUMs above 20 % free.
- `launchd_logs_retention.py` — `*.log`/`*.err` directly under the launchd log dirs,
  idle > 90 d; a garbage collector for retired units, never a rotator of live files.
- Backups keep 3 dailies by default; site-monitor screenshots/reports keep 24 h.

## Tests

`python -m pytest -q` in this directory — 57 offline tests. `conftest.py` puts this
directory on `sys.path` and pins dummy Telegram credentials, so no test can read a
real token. Covered: heartbeat state machine, Telegram fallback, backup snapshot /
transfer / rotate end-to-end against local dirs, freshness thresholds, WAL adoption,
WAL-index parsing and stall logic (the `pinned_reader_pid` test assumes the Darwin
`struct flock` layout, i.e. macOS). Not covered: the JS pieces and the retention
scripts; tests for pipeline code outside this repo were left out.

## Dependencies

In-host Python is stdlib-only (tests: `pytest`) and shells out to `rsync`, `ssh`,
`/sbin/ping`, `plutil`, `curl`. `synthetic_smoke.mjs`: Node + `playwright`;
`site-monitor/`: `playwright`, `httpx`, Lighthouse CLI; `cf-worker/`: `wrangler`. The two
DB retention scripts and the gather step of `prewarm_freshness_check.py` import
main-application modules (run ledger, cache config, commute prewarm) not in this repo.

## Environment variables

| Var | Used by |
|---|---|
| `WTL_ROOT` | app checkout root with `data/`, `logs/`, `venv/` (default `/opt/wheretolive`) |
| `WTL_TG_TOKEN`, `WTL_TG_CHAT`, `WTL_TG_ENV`, `WTL_TG_DISABLE` | Telegram token / chat id / credential-file path / global mute |
| `WTL_API`, `WTL_ADMIN_KEY`, `WTL_HOST_ID`, `WTL_HEARTBEAT_STATE`, `WTL_MONITOR_HOST_IP` | host_report, heartbeat_check; host pinged by `check_host_reachable` (default `gpu-box`) |
| `WTL_DATA_DIR`, `WTL_STAGING`, `WTL_BACKUP_SSH`, `WTL_BACKUP_DEST`, `WTL_RSYNC_TIMEOUT`, `WTL_KEEP_DAILY`, `WTL_KEEP_WEEKLY`, `WTL_FORCE_WEEKLY`, `WTL_MARKER`, `WTL_STATUS_JSON`, `WTL_LOCKDIR`, `WTL_LOCK_STALE_MIN`, `WTL_ALLOW_OVERRIDE`, `WTL_BACKUP_DATE` | backup_dbs |
| `WTL_NOTIFY_CMD` | notifier command for backup_dbs and the freshness checks |
| `WTL_BACKUP_MAX_AGE_H` · `CLAUDE_DAILY_COST_ALERT_USD` · `WTL_SMOKE_BASE` | freshness threshold (48 h) · daily spend threshold (USD 50) · smoke base URL |
| `INGEST_URL`, `SITE_HEALTH_INGEST_TOKEN` | site-monitor |
| `TG_TOKEN` (secret), `TG_CHAT_ID`, `HEALTH_URL`, KV binding `STATE` | cf-worker |

The site is bilingual (EN/ZH); the one Chinese string left in this module is a
login-wall keyword the external monitor matches on the page.
