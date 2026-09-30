/**
 * wheretolive health monitor — Cloudflare Worker.
 *
 * Runs every 5 minutes (cron), fetches /api/health, sends Telegram alert on
 * down → up / up → down transitions. Avoids spam by tracking last state in KV.
 *
 * Bindings:
 *   STATE        Workers KV namespace (stores last_status + last_alert_ts)
 *   TG_TOKEN     secret: Telegram bot token
 *   TRIGGER_TOKEN secret (optional): enables the manual fetch trigger
 *   TG_CHAT_ID   var/secret: Telegram chat id to DM
 *   HEALTH_URL   var: full health endpoint URL
 */

const ALERT_COOLDOWN_S = 1800;   // re-alert if down for >30 min uninterrupted
const DOWN_THRESHOLD = 2;        // alert only after 2 consecutive DOWN probes
                                  // (avoids false alarms during ~1s prod restarts)

export default {
  async scheduled(event, env, ctx) {
    ctx.waitUntil(runCheck(env));
  },
  // Manual trigger for debugging (only reachable if workers_dev / a route is enabled):
  //   curl -H "x-trigger-token: <TRIGGER_TOKEN>" https://<worker>.workers.dev/
  // Uses its own secret in a header, so the bot token never appears in a URL or access log.
  async fetch(req, env, ctx) {
    const given = req.headers.get('x-trigger-token');
    if (!env.TRIGGER_TOKEN || given !== env.TRIGGER_TOKEN) {
      return new Response('forbidden', { status: 403 });
    }
    const r = await runCheck(env);
    return Response.json(r);
  },
};

async function runCheck(env) {
  let healthBody, healthStatus;
  let fetchError = null;
  try {
    const resp = await fetch(env.HEALTH_URL, {
      headers: { 'User-Agent': 'wtl-health-monitor/1' },
      cf: { cacheTtl: 0, cacheEverything: false },
    });
    healthStatus = resp.status;
    healthBody = await resp.json();
  } catch (e) {
    fetchError = e.message || String(e);
    healthStatus = 0;
    healthBody = null;
  }

  // Derive current state
  let current;
  if (fetchError) {
    current = 'down';
  } else if (healthStatus >= 500) {
    current = 'down';
  } else if (healthBody?.status === 'down') {
    current = 'down';
  } else {
    current = 'ok';   // ok or degraded both count as "up" per user preference
  }

  // Read last state + down-streak + last alert time from KV.
  // last_status: confirmed alert state, only flips after DOWN_THRESHOLD probes
  // down_streak: count of consecutive DOWN probes since last OK
  const last = (await env.STATE.get('last_status')) || 'ok';
  const lastAlertTs = parseInt((await env.STATE.get('last_alert_ts')) || '0', 10);
  const downStreak = parseInt((await env.STATE.get('down_streak')) || '0', 10);
  const now = Math.floor(Date.now() / 1000);

  // Update down streak based on this probe.
  // Conditional write — Workers KV free tier is 1000 writes/day, and at
  // 5-min cadence × 2 unconditional puts we burn 576/day on no-ops
  // (hit 50% warning email on 2026-05-20). In steady state (always up)
  // newStreak === downStreak === 0 → skip the write entirely.
  //
  // Cap at DOWN_THRESHOLD + 1 — once we're well past the alert
  // threshold the exact streak length affects no decision, so we don't
  // need to keep incrementing on every 5-min tick of a sustained
  // outage (which would be 288 writes/day = 30% of free tier).
  const STREAK_CAP = DOWN_THRESHOLD + 1;
  const newStreak = current === 'down'
    ? Math.min(downStreak + 1, STREAK_CAP)
    : 0;
  if (newStreak !== downStreak) {
    await env.STATE.put('down_streak', String(newStreak));
  }

  // Confirmed state requires DOWN_THRESHOLD consecutive failures
  const confirmedCurrent = newStreak >= DOWN_THRESHOLD ? 'down' : 'ok';

  let alertSent = false;
  let alertText = null;

  if (confirmedCurrent === 'down' && last !== 'down') {
    // Transitioned down → alert immediately
    alertText = buildDownAlert(env.HEALTH_URL, healthStatus, healthBody, fetchError);
    await sendTelegram(env, alertText);
    alertSent = true;
    await env.STATE.put('last_alert_ts', String(now));
  } else if (confirmedCurrent === 'down' && last === 'down') {
    if (now - lastAlertTs > ALERT_COOLDOWN_S) {
      alertText = buildDownAlert(env.HEALTH_URL, healthStatus, healthBody, fetchError, true);
      await sendTelegram(env, alertText);
      alertSent = true;
      await env.STATE.put('last_alert_ts', String(now));
    }
  } else if (confirmedCurrent === 'ok' && last === 'down') {
    alertText = `*[RECOVERED]* wheretolive health is back to OK\n\nhttp ${healthStatus} · status \`${healthBody?.status || 'ok'}\``;
    await sendTelegram(env, alertText);
    alertSent = true;
    await env.STATE.put('last_alert_ts', '0');
  }

  // Same conditional-write logic as down_streak above. Once a status is
  // confirmed and persisted, repeating the put on every cron is free-tier
  // waste — only write on actual transition.
  if (confirmedCurrent !== last) {
    await env.STATE.put('last_status', confirmedCurrent);
  }
  return { current, confirmedCurrent, last, downStreak: newStreak, alertSent, healthStatus, healthBody, fetchError };
}

function buildDownAlert(url, httpStatus, body, fetchError, isRepeat = false) {
  const head = isRepeat
    ? '*[STILL DOWN]* wheretolive uninterrupted for >30 min'
    : '*[DOWN]* wheretolive health failed';
  if (fetchError) {
    return `${head}\n\nFetch error: \`${fetchError}\`\nEndpoint: ${url}`;
  }
  const failed = body && body.checks
    ? Object.entries(body.checks)
        .filter(([_, v]) => !v.ok)
        .map(([k, v]) => `• \`${k}\`: ${escape(v.error || 'fail')}`)
        .join('\n')
    : '(no checks structure in body)';
  return `${head}\n\nhttp ${httpStatus} · status \`${body?.status || 'unknown'}\`\n\n${failed}`;
}

function escape(s) {
  // Telegram MarkdownV1 — escape characters that confuse formatting
  return String(s).replace(/[*_`\[]/g, (c) => '\\' + c);
}

async function sendTelegram(env, text) {
  const resp = await fetch(`https://api.telegram.org/bot${env.TG_TOKEN}/sendMessage`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      chat_id: env.TG_CHAT_ID,
      text,
      parse_mode: 'Markdown',
      disable_web_page_preview: true,
    }),
  });
  if (!resp.ok) {
    // Best-effort log; can't do much from a worker if TG itself fails
    console.error('telegram send failed', resp.status, await resp.text());
  }
}
