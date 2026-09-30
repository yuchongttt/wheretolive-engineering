# wheretolive Health Monitor · Cloudflare Worker

Runs every 5 minutes from CF edge. Fetches `https://wheretolive.xyz/api/health`,
sends Telegram alert on **down ↔ ok** state transitions (deduped via KV).

## Architecture

```
CF cron (*/5 * * * *)
  → fetch /api/health
  → read last_status + down_streak from KV
  → "down" is confirmed only after 2 consecutive failed probes
  → if confirmed state changed (ok→down / down→ok): send TG alert
  → if still down after 30 min: re-alert
  → write current state to KV
```

## One-time deploy (5 min)

```bash
# 1. Install Wrangler (CLI) if missing
npm install -g wrangler
wrangler login   # opens browser, authorise with CF account

cd ops/monitoring/cf-worker

# 2. Create a KV namespace and grab the id
wrangler kv namespace create wtl-monitor-state
# Output example:
#   🌀 Creating namespace with title "wtl-health-monitor-wtl-monitor-state"
#   ✨ Success!
#   { binding = "STATE", id = "abc123def456..." }

# 3. Paste the id into wrangler.toml (replace <KV_NAMESPACE_ID>)
#    and set TG_CHAT_ID (replace <TG_CHAT_ID>, or make it a secret)
# 4. Set the Telegram bot token as a secret (won't be in repo)
wrangler secret put TG_TOKEN
# Paste: <the bot token — same value as WTL_TG_TOKEN; never commit it>
# Optional: a separate secret for the manual trigger (see Verify)
wrangler secret put TRIGGER_TOKEN

# 5. Deploy
wrangler deploy
```

## Verify

The committed config is cron-only (`workers_dev = false`), so the manual
trigger below is only reachable if you enable `workers_dev` (or a route) or run
`wrangler dev`.

```bash
# Trigger a manual check (the fetch handler requires TRIGGER_TOKEN in a header)
curl -H "x-trigger-token: <TRIGGER_TOKEN>" "https://wtl-health-monitor.<your-cf-subdomain>.workers.dev/"
# Expect: {"current":"ok","last":"ok","alertSent":false,...}

# Force-test alert path by setting last_status to 'down' via wrangler:
wrangler kv key put --binding STATE last_status down
# Wait <5 min for next cron tick OR re-trigger with the curl above
# You should get a "[RECOVERED]" message in Telegram.
```

## Roll back

```bash
wrangler delete wtl-health-monitor
```

## Alerting behaviour

| Event | TG message |
|---|---|
| Transition ok → down (2 consecutive failed probes) | `[DOWN]` + list of failed checks |
| Still down after 30 min | `[STILL DOWN]` re-alert (then every 30 min) |
| Transition down → ok | `[RECOVERED]` |
| Health = degraded (DINO/SigLIP down) | **no alert** (by design: page on down only) |

## Files

- `wrangler.toml` — CF Worker config + KV binding + cron schedule
- `src/index.js` — Worker code (cron handler + manual fetch trigger)

## Security notes

- TG bot token must be set as a CF **secret** (`wrangler secret put TG_TOKEN`),
  not in `wrangler.toml`. Token in repo = anyone can send TG to your chat.
- The manual trigger is gated by its own `TRIGGER_TOKEN`, sent as a header, so the
  bot token never appears in a URL or an access log. With no `TRIGGER_TOKEN`
  set, the trigger always returns 403.
- If the bot token is ever exposed, revoke it via @BotFather `/revoke`.
