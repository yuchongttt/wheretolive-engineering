// Synthetic content smoke — catches "page returns 200 but the data never renders"
// failures that plain HTTP health checks (status-code only) miss. This is exactly
// the class of bug that left /report + /area silently stuck on "Fetching data…"
// for days while site-health stayed green (a Next.js force-dynamic / hydration regression).
//
// Loads the real PUBLIC pages in a headless browser and asserts the meaningful
// content actually appears (not just HTTP 200). On a CHANGE in the failing set
// (new failure or recovery) it sends a Telegram alert via notify_telegram.py.
// Run from launchd every ~30 min. Always exits 0 (alerting is the signal).

import { chromium } from 'playwright';
import { spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync, appendFileSync, mkdirSync, readdirSync } from 'node:fs';
import { homedir } from 'node:os';
import { fileURLToPath } from 'node:url';

// Application checkout root (holds data/, logs/, venv/).
const ROOT = process.env.WTL_ROOT || '/opt/wheretolive';
const NOTIFY = fileURLToPath(new URL('./notify_telegram.py', import.meta.url));
const BASE = process.env.WTL_SMOKE_BASE || 'https://wheretolive.xyz';
const STATE = `${ROOT}/data/synthetic_smoke_state.json`;
const LOG = `${ROOT}/logs/synthetic-smoke.log`;
try { mkdirSync(`${ROOT}/logs`, { recursive: true }); } catch { /* ignore */ }

// Each check: load the page, wait for a selector proving the real content rendered.
// `waitText` is a regex string matched against page text; absence within `timeout` = FAIL.
const CHECKS = [
  { name: 'report', url: `${BASE}/report?postcode=SW1A%201AA&destination=EC2R%208AH`, waitText: 'better than|/100', timeout: 55000 },
  { name: 'area', url: `${BASE}/area/sw1a-1aa`, waitText: 'Scores by measure|area guide', timeout: 25000 },
  // Two checks for a sibling app were removed (2026-07-13 / 2026-08-13) when that
  // app was split out onto its own server and stopped proxying any route here.
  // One of them had been red 277 times in a row since 08-07, guarding a contract
  // that no longer had a consumer — an always-red monitor only trains you to
  // ignore it, so it was taken out.
  { name: 'home', url: `${BASE}/`, waitText: 'wheretolive|postcode|London|Explore', timeout: 25000 },
];

function log(line) {
  const ts = new Date().toISOString();
  try { appendFileSync(LOG, `${ts} ${line}\n`); } catch { /* ignore */ }
  console.log(`${ts} ${line}`);
}

// Keep our own synthetic traffic out of Cloudflare RUM. The beacon is injected
// unconditionally in web/src/app/layout.tsx, so this 30-min smoke was landing
// in Web Analytics as real "visits" alongside the hourly site-monitor.
// Only the payload endpoint is intercepted: beacon.min.js carries an SRI
// integrity attribute (so it cannot be stubbed) and POSTs first-party to
// /cdn-cgi/rum, so answering that POST is enough. Full write-up in
// site-monitor/monitor.py.
const RUM_PAYLOAD = /\/cdn-cgi\/rum/;

async function runCheck(browser, c) {
  const ctx = await browser.newContext();
  await ctx.route(RUM_PAYLOAD, (r) => r.fulfill({ status: 204, body: '' }));
  if (c.init) await ctx.addInitScript(c.init);
  const page = await ctx.newPage();
  try {
    await page.goto(c.url, { waitUntil: 'load', timeout: 60000 });
    // >> visible=true: without it waitForSelector latches onto the FIRST DOM match —
    // if that's a hidden element (e.g. mobile-nav button at desktop viewport) the
    // check times out even though the same text is visible elsewhere on the page.
    await page.waitForSelector(`text=/${c.waitText}/i >> visible=true`, { timeout: c.timeout });
    await ctx.close();
    return { name: c.name, ok: true };
  } catch (e) {
    await ctx.close().catch(() => {});
    return { name: c.name, ok: false, detail: (e && e.message ? e.message : String(e)).slice(0, 120) };
  }
}

// launchd plist lint — catches the 2026-06-12 rot class: a live plist gets clobbered
// (e.g. plutil -extract output redirected onto itself) while the in-memory registration
// keeps the service running, so nothing notices until a reboot fails to load it and the
// site stays down. plutil -lint is deterministic — zero false-positive risk.
function checkLaunchdPlists() {
  const dir = `${homedir()}/Library/LaunchAgents`;
  const mine = /^(xyz\.wheretolive\.|com\.wheretolive\.).*\.plist$/;
  const broken = [];
  for (const f of readdirSync(dir).filter((f) => mine.test(f))) {
    const r = spawnSync('/usr/bin/plutil', ['-lint', `${dir}/${f}`], { timeout: 10000 });
    if (r.status !== 0) broken.push(f);
  }
  return broken.length
    ? { name: 'launchd-plists', ok: false, detail: `won't survive reboot: ${broken.join(', ')}`.slice(0, 120) }
    : { name: 'launchd-plists', ok: true };
}

const browser = await chromium.launch();
const results = [];
for (const c of CHECKS) results.push(await runCheck(browser, c));
await browser.close();
results.push(checkLaunchdPlists());

const failing = results.filter((r) => !r.ok).map((r) => r.name).sort();
const summary = results.map((r) => `${r.ok ? '✅' : '❌'}${r.name}`).join(' ');
log(`smoke ${BASE} → ${summary}`);

// Alert only when the failing set CHANGES (new failure or recovery) — no spam.
let prev = [];
try { prev = JSON.parse(readFileSync(STATE, 'utf8')).failing || []; } catch { /* first run */ }
const changed = JSON.stringify(failing) !== JSON.stringify(prev);
try { writeFileSync(STATE, JSON.stringify({ failing, at: new Date().toISOString() })); } catch { /* ignore */ }

if (changed) {
  let msg;
  if (failing.length) {
    const lines = results.filter((r) => !r.ok).map((r) => `• ${r.name}: ${r.detail}`);
    msg = `🔴 wheretolive synthetic smoke FAILING (${failing.join(', ')})\n${lines.join('\n')}\n${BASE}`;
  } else {
    msg = `🟢 wheretolive synthetic smoke RECOVERED — all pages render again.`;
  }
  log(`ALERT (set changed ${JSON.stringify(prev)}→${JSON.stringify(failing)})`);
  spawnSync(`${ROOT}/venv/bin/python3`, [NOTIFY, msg], { timeout: 20000 });
}
