# ops/watchdog: self-healing watchdog

A small watchdog for a single-host production setup: one Mac mini serving the site, dozens of
launchd jobs, and one worker on an external VPS. It fixes a short whitelist of known failures on
its own. It checks each fix on the next tick, and it only messages a human when a fix did not
work. Design rationale, failure history and non-goals are in **[DESIGN.md](DESIGN.md)**.

## What it handles

| Rule | Detects | Does | Budget |
|---|---|---|---|
| `launchd-job` | an allowlisted job has a non-zero last exit **and** no fresh healthy signal | `launchctl kickstart -k` | 3 / 30 min |
| `hetzner-worker` | `systemctl is-active wtl-worker` on the VPS is not `active` | `ssh … systemctl restart wtl-worker` | 1 / 30 min |
| `net-wedged` | cloudflared has 0 edge connections on 2 ticks in a row | restart cloudflared, or bounce Wi-Fi if the host has no route at all | 2 / 30 min |

Anything not on this list is left to the normal alerting. Once a rule's budget is used up, the
watchdog stops and escalates.

## How it works

```
launchd (every 120 s) ─► watchdog.py: run_once(conn, mode, now, deps)
   for each rule:  healthy? ── yes ─► close any open episode ("recover")
                        └──── no ──► observe: log "would-fix"      (no action)
                                     enforce: within budget → fix, log "fix"
                                              over budget   → escalate once (Telegram)
   always: POST heartbeat ─► admin API ◄─ checker on another host alerts if it goes stale
```

- **Verify after acting.** A `fix` row never means "fixed". The next tick has to see the item
  healthy before it writes `recover`.
- **Ledger.** Every decision is a row in `ops.db:watchdog_actions`. The newest row per
  `(rule, target)` is that item's state. Budgets, recovery detection and the two-tick check for
  the network rule are all computed from it. There is no in-memory state, so a crash or restart
  never loses track.
- **Observe first.** `WATCHDOG_MODE=observe` (the default) detects and logs what it *would* do,
  and executes nothing. Production ran in observe mode before being switched to `enforce`.
- **Silence means healthy.** Auto-fixes that succeed are recorded but not sent. The watchdog
  speaks only on escalation and on recovery after an escalation. That is safe only because of
  the external heartbeat check (`../monitoring/heartbeat_check.py`), which catches the watchdog,
  or the whole host, going dark.

## Key design decisions and trade-offs

- **Whitelist, not an agent.** Only failures with a proven, idempotent, bounded fix are handled
  automatically. The website process is deliberately *not* restarted. The watchdog also does not
  kill processes or reboot the host. Those stay with a human, and the escalation message suggests
  the next step.
- **Two signals before acting.** A launchd job is "down" only if its last exit is non-zero *and*
  its freshness signal is stale. A one-shot job that exited 0 is healthy. When the signal cannot
  be read, the job counts as fresh: the watchdog does not act when it is unsure.
- **No DNS in the network rule.** The tunnel gauge is read from `127.0.0.1`, and egress is probed
  by literal IP. In the incident this rule was built for, DNS failed along with routing, so a
  probe that used a hostname would have misled the check.
- **Repairs fail safe.** The Wi-Fi bounce always ends with the radio on, even if the previous
  tick died half-way.
- **Testable by construction.** All side effects (launchctl, ssh, HTTP, Telegram, sockets) are
  passed into `run_once` as callables. The tests run the real state machine against fakes and a
  temporary SQLite file.

## Files

| File | Role |
|---|---|
| `watchdog.py` | the tick loop, plus the real-world wiring (launchctl, ssh, cloudflared metrics, `networksetup`) |
| `watchdog_rules.py` | whitelist, budgets, launchctl parsing, network fault classification |
| `watchdog_jobs.json` | the launchd allowlist and per-job freshness windows |
| `watchdog_fresh.py` | freshness signals (log mtime, or the newest run-ledger row, allowing for catch-up runs) and log tails |
| `watchdog_ledger.py` | `watchdog_actions` table and budget/state queries |
| `watchdog_notify.py` | message formatting and the tier-1 coalescing helper |
| `watchdog_heartbeat.py` | liveness POST for the external checker |

The launchd plist is `../deploy/launchd/xyz.wheretolive.watchdog.plist`. The Telegram sender
(`monitor_checks.tg` → `wtl_tg`) lives in `../monitoring`, and `main()` imports it from there.

## Running the tests

```bash
cd ops/watchdog && python -m pytest -q
```

The tests are standard-library only, apart from `pytest`. They are offline: no network, no
launchctl, no real database (they use `tmp_path` SQLite files and injected fakes). The result at
extraction time was `34 passed`.

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `WATCHDOG_MODE` | `observe` or `enforce` | `observe` |
| `WATCHDOG_JOBS_FILE` | path to the allowlist JSON | `./watchdog_jobs.json` |
| `WTL_ROOT` | app checkout: `ops.db` is at `$WTL_ROOT/data/ops.db` | `/opt/wheretolive` |
| `WATCHDOG_LOG_DIR` | where `<label>.log` freshness files are read | `$WTL_ROOT/logs` |
| `WATCHDOG_REMOTE_HOST` | ssh target of the VPS running `wtl-worker` | `root@hetzner-box` (placeholder) |
| `WATCHDOG_WIFI_IF` | interface bounced by the network repair | `en1` |
| `WTL_API` | admin API base URL for the heartbeat | `http://127.0.0.1:3000` |
| `WTL_ADMIN_KEY` | **secret**: admin API key for the heartbeat POST | none, must be set |
| `WTL_TG_TOKEN`, `WTL_TG_CHAT` | **secret**: Telegram bot credentials (read by `../monitoring/wtl_tg.py`) | none, must be set |

## Differences from the private repo

- The allowlist moved from a Python constant to `watchdog_jobs.json`. The production list also
  names data-ingestion jobs that are not part of this repo, so the public file keeps only the
  others. The special case for the one job whose freshness comes from `ops.db` became a general
  `ledger_skill` option, and the catch-up constant was renamed to `CATCHUP_FRESH_MINUTES`.
- The hard-coded VPS host, heartbeat API address and repo-relative paths are now environment
  variables.
- `check_hetzner_worker()` moved from the shared `monitor_checks` module into `watchdog.py` (as
  `_hetzner_ok`), because the public `monitor_checks` extract does not carry it.
- Alert text and comments were translated to English. Test fixtures use neutral job labels, and
  `tests/test_watchdog_fresh.py` keeps only the watchdog cases of a larger monitor test.
  `test_load_jobs_reads_budgets_and_ledger_skills` is new, for the JSON loader.
