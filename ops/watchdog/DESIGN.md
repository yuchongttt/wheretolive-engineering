# Self-healing watchdog: design

Condensed from the private repo's design spec and implementation plan (both dated 2026-07-01).
It is updated to match the code as of 2026-09-29. Where the code moved on from the spec, this
document follows the code.

## 1. Problem

One Mac mini is the production host. It serves the site (Next.js behind a Cloudflare Tunnel) and
runs dozens of launchd jobs. Before the watchdog, a person found out about a failure hours later
from a Telegram alert and then fixed it by hand. Three kinds of failure kept coming back:

| Failure | What happened | Shape of the fix |
|---|---|---|
| Interpreter dies at startup | Under memory pressure, Python processes that launchd had just spawned sometimes died inside `getpath` with `EINTR`, before any user code ran (first seen 2026-07-01). | Kickstart the job. The retry is idempotent. |
| Remote worker down | The `wtl-worker` systemd unit on the external Hetzner VPS stopped being `active`. | Restart the unit over SSH. |
| Host loses its route out | From 2026-08-27 12:51Z to 2026-08-28 22:24Z (33h32m) the Mac stayed up but could not reach the internet. cloudflared logged `network is unreachable` 11,679 times, and DNS (Tailscale MagicDNS) failed with it. Nothing noticed until a manual reboot. *(Figures from code comments in `watchdog_rules.py`.)* | Restart cloudflared if only the tunnel is down. Rebuild the Wi-Fi interface if the host has no route at all. |

**Goal.** When a failure is known and has a proven, idempotent, bounded fix: detect it, apply the
fix, check that the fix worked, and tell the human the *outcome*. Anything else goes to the
existing alerting.

## 2. Autonomy model: fix what is on the whitelist, only alert on the rest

| Rule | Detection | Remediation | Budget |
|---|---|---|---|
| `launchd-job` | `launchctl list` last exit ≠ 0 **and** no fresh healthy signal | `launchctl kickstart -k gui/$UID/<label>` | 3 per 30 min |
| `hetzner-worker` | `ssh … systemctl is-active wtl-worker` ≠ `active` | `ssh … systemctl restart wtl-worker` | 1 per 30 min |
| `net-wedged` | cloudflared reports 0 edge connections, seen on 2 ticks within 15 min | tunnel fault → kickstart cloudflared · host fault → bounce Wi-Fi | 2 per 30 min (one per rung) |

**`launchd-job` needs both signals.** A one-shot job that ran fine shows exit 0 and has no
process, and that is healthy. A job with a bad exit code but a fresh signal gets one more cycle.
The fresh signal comes from one of two places:

- the mtime of the job's log file, checked against a per-job window (`fresh_minutes`), or
- for jobs tracked in the run ledger, the newest `skill_runs` row in `ops.db`. A *catch-up* run
  (after an outage, marked `mode=catchup`) counts as fresh for up to 180 min. Without that
  exception the watchdog would kill every catch-up at the 70-minute mark. In production this is
  used for the hourly data-ingestion job, which is not part of this repo.

If the signal cannot be read (log missing, DB error), the job counts as fresh. The watchdog does
not act when it is unsure.

**`net-wedged` uses two independent pieces of evidence:**

1. cloudflared's own Prometheus gauge `cloudflared_tunnel_ha_connections`, read from
   `127.0.0.1`. That needs no DNS, and DNS is one of the things that fails in this outage.
2. A raw TCP connect to `1.1.1.1:443` or `9.9.9.9:443`. These are literal IPs from two operators,
   so a dead resolver cannot make the probe agree with itself.

If the tunnel has connections, everything is healthy, whatever the probe says. If the tunnel is
down and egress works, the fault is cloudflared's own, so the watchdog restarts it. If the tunnel
is down and egress fails too, the kernel has no route. Restarting cloudflared cannot fix that, so
the watchdog turns the Wi-Fi radio off and on again with `networksetup`, which needs no sudo.
That repair always ends by switching the radio **on**. If an earlier tick died between off and on,
it only switches the radio on. Nothing is repaired on a single sighting, because cloudflared shows
zero connections for a moment during its own restarts. A blip that clears before any action is
closed in the ledger, so the next fault has to pass the two-tick check again. Every ledger row for
this rule carries a one-line routing snapshot (default gateway, interface, gateway ping, Wi-Fi
power), because the 2026-08-27 outage could not be reconstructed afterwards.

## 3. Architecture

```
                 launchd: every 120 s, via wtl-pyrun.sh
                              │
                     ┌────────▼─────────┐
 launchctl list ────►│                  │──► launchctl kickstart -k <job>
 log mtime / ops.db ►│    run_once()    │──► ssh <vps> systemctl restart wtl-worker
 ssh is-active ─────►│ observe|enforce  │──► kickstart cloudflared | bounce Wi-Fi
 cloudflared gauge ─►│                  │
 raw-IP TCP probe ──►└──┬───────────┬───┘
                        │           │
          watchdog_actions ledger   Telegram: escalation / recovery-after-escalation
          (ops.db, WAL)             │
                                    └─► heartbeat POST ─► admin API ◄─ gpu-box checker (every 5 min)
                                                                        └─► Telegram "watchdog / Mac down"
```

- **The healer does not depend on what it heals.** The watchdog is its own launchd job and is not
  part of the monitor it protects. It is started through `wtl-pyrun.sh` (see `../deploy`), so the
  startup crash it exists to fix cannot kill it. Its schedule is `StartInterval=120`,
  `KeepAlive{SuccessfulExit=false}` and `ThrottleInterval=15`. A crashed tick is retried after
  15 s. A clean exit waits for the next interval.
- **One tick per process.** There is no long-lived loop and no state in memory. All state lives
  in the ledger, so a restart is always safe.
- **Side effects are injected.** `run_once(conn, mode, now, deps)` receives every side effect
  (launchctl, ssh, HTTP, Telegram, network probes) as a dict of callables. The tests drive the
  whole state machine with fakes and a temporary SQLite file.
- **Dead-man switch.** Every tick posts a heartbeat (`host_id=mac-watchdog`) to the admin
  host-metrics API. A checker on the GPU box runs from a systemd timer every 5 min
  (`../monitoring/heartbeat_check.py`) and alerts when the heartbeat is more than 600 s old.
  - If the whole Mac is stale, it sends only the Mac alert.
  - If it cannot reach the API at all, it treats that as "Mac down". During the 2026-08-27
    outage this checker exited with an error every 5 minutes and never alerted, so that case was
    added afterwards.

## 4. Observe mode and enforce mode

With `WATCHDOG_MODE=observe` (the default), the watchdog runs full detection. It writes `detect`
rows with `would-fix: <reason>` to the ledger, but it executes nothing and sends nothing. It still
posts the heartbeat. The rollout was to run in observe mode for a few days, check the `detect`
rows for false positives (for example, "would-fix" on healthy idle jobs), and only then set
`enforce` in the plist. In short: eyes before hands. The production plist in
`../deploy/launchd` runs in `enforce` mode.

## 5. Verify after acting

The watchdog never reports a fix as a success just because it ran. It records a `fix` row with the
exit code. On the **next** tick, if the item is healthy, it records `recover` with
`verified_recovered=1`. If the item is still unhealthy, it tries again, and each attempt counts
against the budget. Once the fixes in the rolling window reach the budget, it sends one
escalation per episode and stops acting. It acts again only after the item recovers or the window
has moved past the old fixes.

## 6. The ledger

Table `watchdog_actions` in `ops.db`, opened with WAL and `busy_timeout=10000`:

```
id · ts · rule_id · target · action (detect|fix|escalate|recover) · mode (observe|enforce)
   · exit_code · verified_recovered (1/0/NULL) · note
```

It has one row per event. The newest row for a `(rule_id, target)` pair is that item's current
state:

```
detect ─────────────► recover   ("cleared before action", net rule only)
detect ─► fix ──────► recover   ("verified", next healthy tick)
          fix ×n ───► escalate ─► recover ("post-escalation", sends ✅)
```

The ledger drives the budget (fix rows in the window), recovery detection, the two-tick check and
the admin "Runs" view.

## 7. Alerting contract

The rules: check before reporting, speak only when state changes, and treat silence as healthy.
Silence can only mean healthy because the heartbeat turns "the reporter went dark" into an alert
of its own. All alerts go through the Telegram bot the rest of the fleet already uses, so there is
no new channel and no new secret.

| Tier | When | Sent? |
|---|---|---|
| 🔧 1 auto-fixed | a fix was verified | **Not sent since 2026-08-05.** Recorded in the ledger only, because a self-healed item needs no action. `fmt_fixed` and the 1-per-hour coalescing helper are still in `watchdog_notify.py`. |
| 🔴 2 needs you | budget used up | Once per episode. Includes the symptom, the last 3 log lines and a concrete hint (for `net-wedged`: "most likely needs a reboot"). |
| ✅ 3 recovered | an escalated item is healthy again | Yes. It closes an alert the human actually received. |
| 🔴🔴 4 watchdog / Mac down | heartbeat stale | Sent by the external checker, not by this module. |

## 8. What it deliberately does not do

- **Restart the website (`next-prod`).** The stakes are high, and failures there are usually
  about builds or permissions, so they only raise alerts. A test checks that `next-prod` is not on
  the allowlist.
- **Kill processes.** Picking a PID automatically is not safe.
- **Act on anything off the whitelist.** An unknown failure is left to the existing monitors to
  alert on. The watchdog never improvises.
- **Reboot the host.** The escalation tells the human to do it.
- **Use an LLM to diagnose or fix.** The rules are code and config that someone can review.
- **Take commands.** It only sends notifications, so there is no inbound channel to spoof.
- **Retry forever.** Every rule has a budget.
- **Use broad remote access.** SSH to the VPS runs only `systemctl is-active` and
  `systemctl restart wtl-worker`. The spec lists an `authorized_keys` forced command as the next
  hardening step.

## 9. Testing

The unit tests (in `tests/`) cover:

- launchctl parsing and the two-signal health rule
- ledger windows, notification formatting and coalescing
- observe mode versus enforce mode
- the network detector: gauge parsing, fault classification, the two-tick check and closing blips
- Wi-Fi bounce safety: the radio always ends up on, including when it was already off
- the routing snapshot in fault notes
- catch-up freshness

Observe mode in production was the live acceptance test.

## 10. Known gaps (as of the 2026-09-29 snapshot)

- The recovery-after-escalation message uses a fixed 3600 s downtime
  (`fmt_recovered(target, 3600)`) instead of the measured one.
- Log-based freshness reads `$WATCHDOG_LOG_DIR/<label>.log` (default `$WTL_ROOT/logs`). The
  launchd plists in `../deploy` write stdio to a different directory, so both must point at the
  same place. If the file is missing, the job counts as fresh, and the `launchd-job` rule never
  fires for it.
- The spec asked for each rule to be isolated with its own `try/except`. The loop does not do
  that: an exception in any probe aborts the whole tick. launchd then re-runs the tick after 15 s
  (`KeepAlive{SuccessfulExit=false}`). If the crash persists, the heartbeat stops and the external
  checker raises the alarm.
