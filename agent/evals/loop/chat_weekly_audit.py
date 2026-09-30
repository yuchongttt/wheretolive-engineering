#!/usr/bin/env python3
"""Weekly chat-agent audit — the observation-period watchtower (2026-07-12).

One run produces everything "content correctness beats speed" needs watched:
  1. behavior suite (../behaviour/chat_behavior_suite.py) — prompt-contract
     drift, incl. silent model updates behind the 'sonnet' alias;
  2. 7-day telemetry: volume, error rate, TTFB/latency percentiles,
     image_intent funnel (the FAISS go/no-go data), vlm_down count;
  3. tool usage incl. style_tags / visual_rerank adoption;
  4. sampled CONTENT audit: an LLM judge (Claude Sonnet) re-reads up to N real
     turns and checks reply numbers/names are traceable to tool outputs,
     identity claims are evidence-gated, inferences carry disclosures;
  5. HTML report dropped into web/content/research/ (admin Docs reads disk —
     no rebuild) + Telegram alert only when something needs a human.

Scheduled: launchd xyz.wheretolive.chat-weekly-audit (Mon 06:30).
Env knobs: SKIP_SUITE=1 (suite ran recently), SAMPLE=n (judge sample size),
AUDIT_JUDGE_CMD (judge command line; the prompt is appended as the last
argument; unset = judge step skipped), WTL_INTERNAL_USER_IDS, WTL_ROOT.

Public copy: the private version also renders a product growth-funnel section
(visitors → first chat / radar) from a separate analytics module; that section
is removed here — it is product analytics, not agent evaluation.
"""
import html
import json
import os
import re
import shlex
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Root of the production checkout (data/, web/content/research/, scripts/).
REPO = Path(os.environ.get("WTL_ROOT", HERE.parents[2]))
DB = REPO / "data" / "evaluations.db"
OUT = REPO / "web" / "content" / "research" / "chat-weekly-audit.html"
SAMPLE = int(os.environ.get("SAMPLE", "10"))
SKIP_SUITE = os.environ.get("SKIP_SUITE") == "1"

from chat_telemetry_counts import count_errors, count_sessions, split_denials  # noqa: E402

TEST_SESSION = re.compile(r"^(bhv|e2e|dev|prod-ttfb|verify|sim)-")

# Internal accounts — owner and test accounts. Their logged-in
# browser testing must not count as "real user" traffic (user_id='admin'
# only covers x-admin-key calls, not these). The account ids are deployment
# data and come from the environment (comma-separated).
INTERNAL_USER_IDS = {
    u.strip() for u in os.environ.get("WTL_INTERNAL_USER_IDS", "").split(",") if u.strip()
} | {"admin"}                                   # x-admin-key synthetic id


def tg(text: str) -> None:
    try:
        # The Telegram helper is the only home of the bot credentials; it lives
        # in the production checkout's scripts/ and is imported lazily.
        sys.path.insert(0, str(REPO / "scripts"))
        from wtl_tg import send_telegram
        send_telegram(text)
    except Exception as e:
        print(f"[audit] telegram failed: {e}")


def pct(vals, p):
    if not vals:
        return None
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(len(vals) * p))]


# ── 1. behavior suite ────────────────────────────────────────────────────────
suite = {"ran": False, "fails": 0, "warns": 0, "summary": "skipped (SKIP_SUITE=1)"}
if not SKIP_SUITE:
    print("[audit] running behavior suite …", flush=True)
    r = subprocess.run([sys.executable, str(HERE.parent / "behaviour" / "chat_behavior_suite.py")],
                       capture_output=True, text=True, timeout=1800)
    m = re.search(r"(\d+) checks: (\d+) pass, (\d+) warn, (\d+) fail", r.stdout)
    suite = {
        "ran": True,
        "fails": int(m.group(4)) if m else (0 if r.returncode == 0 else 99),
        "warns": int(m.group(3)) if m else 0,
        "summary": (m.group(0) if m else f"exit={r.returncode}"),
        "detail": "\n".join(l for l in r.stdout.splitlines()
                            if l.startswith(("FAIL", "WARN"))),
    }
    print(f"[audit] suite: {suite['summary']}", flush=True)

# ── 2+3. telemetry & tool usage (7 days, real users only) ───────────────────
c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
c.row_factory = sqlite3.Row
rows = c.execute("""
    SELECT session_id, user_id, query_type, has_image, image_intent,
           ttfb_ms, response_latency_ms, status, error_message
    FROM chat_telemetry
    WHERE created_at > datetime('now', '-7 days')
""").fetchall()
rows = [r for r in rows if not TEST_SESSION.match(r["session_id"] or "")
        and r["user_id"] not in INTERNAL_USER_IDS]
# A daily-cap refusal (2026-09-24+) is a signal, not an error, and carries no
# session — keep it out of the error rate and the session count, show it on
# its own line (chat_telemetry_counts is the shared caliber).
rows, denied = split_denials(rows)
n_denied = len(denied)

n_turns = len(rows)
n_sessions = count_sessions(rows)
n_users = len({r["user_id"] for r in rows})
n_err = count_errors(rows)
ttfbs = [r["ttfb_ms"] for r in rows if r["ttfb_ms"]]
lats = [r["response_latency_ms"] for r in rows if r["response_latency_ms"]]
img_turns = [r for r in rows if r["has_image"]]
intent_dist: dict = {}
for r in img_turns:
    k = r["image_intent"] or "(router failed)"
    intent_dist[k] = intent_dist.get(k, 0) + 1
vlm_down = intent_dist.get("vlm_down", 0)
qtype_dist: dict = {}
for r in rows:
    qtype_dist[r["query_type"] or "?"] = qtype_dist.get(r["query_type"] or "?", 0) + 1

msg_rows = c.execute("""
    SELECT session_id, user_id, user_text, assistant_text, tool_calls_json
    FROM chat_messages
    WHERE ts > datetime('now', '-7 days')
""").fetchall()
msg_rows = [r for r in msg_rows if not TEST_SESSION.match(r["session_id"] or "")
            and r["user_id"] not in INTERNAL_USER_IDS]
tool_counts: dict = {}
style_uses = 0
for r in msg_rows:
    try:
        calls = json.loads(r["tool_calls_json"] or "[]")
    except Exception:
        calls = []
    for tc in calls:
        tool_counts[tc.get("name", "?")] = tool_counts.get(tc.get("name", "?"), 0) + 1
        if tc.get("name") == "search_properties" and (tc.get("args") or {}).get("style_tags"):
            style_uses += 1
c.close()


# ── 4. sampled content audit (LLM judge, Claude Sonnet, no tools) ────────────
JUDGE_CMD = shlex.split(os.environ.get("AUDIT_JUDGE_CMD", ""))
JUDGE_PROMPT = """You are auditing ONE turn of a property-search assistant for CONTENT problems. Data:

USER: {user}
ASSISTANT REPLY: {reply}
TOOL CALLS MADE: {tools}

Check (be strict but fair):
1. traceable: are the specific numbers, prices, addresses and building names in the reply plausibly grounded in the tool calls made (or clearly marked as estimates/inference)? If the reply cites specifics but NO tool was called that could have produced them, that's false.
2. evidence_gated: if the reply asserts a definite identification ("this IS the listing"), was there tool evidence; single-clue matches must be hedged ("possibly").
3. disclosed: are inferences (style/similarity/era) framed as inference rather than listed fact, and data limits mentioned where relevant?

Output ONLY compact JSON: {{"traceable": true/false, "evidence_gated": true/false, "disclosed": true/false, "issue": "one line, empty if all fine"}}"""

judge_results = []
sample = [r for r in msg_rows if (r["assistant_text"] or "").strip()]
sample.sort(key=lambda r: (0 if '![' in (r["assistant_text"] or '') else 1))  # image-rich first
sample = sample[:SAMPLE]
if sample and JUDGE_CMD:
    print(f"[audit] judging {len(sample)} sampled turns …", flush=True)
    for r in sample:
        prompt = JUDGE_PROMPT.format(
            user=(r["user_text"] or "")[:500],
            reply=(r["assistant_text"] or "")[:2500],
            tools=(r["tool_calls_json"] or "[]")[:1500])
        try:
            out = subprocess.run(
                JUDGE_CMD + [prompt],
                capture_output=True, text=True, timeout=180).stdout
            m = re.search(r"\{.*\}", out, re.S)
            v = json.loads(m.group(0)) if m else {}
        except Exception as e:
            v = {"issue": f"judge error: {e}"}
        ok = v.get("traceable", True) and v.get("evidence_gated", True) and v.get("disclosed", True)
        judge_results.append({
            "session": r["session_id"], "ok": ok,
            "issue": v.get("issue", ""), "user": (r["user_text"] or "")[:80],
        })
flagged = [j for j in judge_results if not j["ok"]]
print(f"[audit] judge: {len(judge_results) - len(flagged)}/{len(judge_results)} clean", flush=True)

# ── 5. report + alerts ───────────────────────────────────────────────────────
today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
e = html.escape


def dist_rows(d: dict) -> str:
    return "".join(f"<tr><td>{e(str(k))}</td><td>{v}</td></tr>"
                   for k, v in sorted(d.items(), key=lambda kv: -kv[1]))


report = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Chat weekly audit {today}</title>
<style>body{{font-family:-apple-system,'PingFang SC',sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;line-height:1.6}}
table{{border-collapse:collapse;margin:.6rem 0}}td,th{{border:1px solid #cfdcd4;padding:.3rem .6rem;font-size:.88rem}}
h2{{border-bottom:1px solid #d8e3dc;padding-bottom:.2rem}}.bad{{color:#b3261e;font-weight:600}}.ok{{color:#2E7D5B}}</style></head><body>
<h1>Chat weekly audit · {today}</h1>
<p>Last 7 days (admin and test sessions excluded). Generated by chat_weekly_audit.py (launchd, Mondays 06:30).</p>

<h2>1. Behaviour contracts (suite)</h2>
<p class="{'bad' if suite['fails'] else 'ok'}">{e(suite['summary'])}</p>
<pre>{e(suite.get('detail', '') or '(no FAIL/WARN)')}</pre>

<h2>2. Traffic and experience</h2>
<table>
<tr><td>turns / sessions / users</td><td>{n_turns} / {n_sessions} / {n_users}</td></tr>
<tr><td>error rate</td><td class="{'bad' if n_turns and n_err / n_turns > 0.1 else ''}">{n_err}/{n_turns}</td></tr>
<tr><td>requests refused by the daily cap</td><td>{n_denied}</td></tr>
<tr><td>TTFB p50 / p90 (ms)</td><td>{pct(ttfbs, .5)} / {pct(ttfbs, .9)}</td></tr>
<tr><td>full turn p50 / p90 (ms)</td><td>{pct(lats, .5)} / {pct(lats, .9)}</td></tr>
</table>

<h2>3. Image funnel (evidence for the FAISS decision)</h2>
<table><tr><th>image_intent</th><th>count</th></tr>{dist_rows(intent_dist) or '<tr><td colspan=2>no image turns</td></tr>'}</table>
<p class="{'bad' if vlm_down > 20 else ''}">vlm_down (lost while the vision service was offline): {vlm_down}</p>

<h2>4. Tool usage</h2>
<table><tr><th>tool</th><th>calls</th></tr>{dist_rows(tool_counts) or '<tr><td colspan=2>none</td></tr>'}</table>
<p>style_tags parameter used: {style_uses} times · query_type distribution: {e(json.dumps(qtype_dist, ensure_ascii=False))}</p>

<h2>5. Sampled content audit ({len(judge_results)} turns, LLM review)</h2>
<table><tr><th>status</th><th>user message</th><th>issue</th></tr>
{"".join(f"<tr><td class='{'ok' if j['ok'] else 'bad'}'>{'✓' if j['ok'] else '✗'}</td><td>{e(j['user'])}</td><td>{e(j['issue'])}</td></tr>" for j in judge_results) or '<tr><td colspan=3>no sample</td></tr>'}
</table>
<p>{len(flagged)} flagged — when anything is flagged, read the original conversation before concluding (the judge has false positives too).</p>
</body></html>"""
OUT.write_text(report)
print(f"[audit] report -> {OUT}", flush=True)

alerts = []
if suite["fails"]:
    alerts.append(f"behaviour suite FAIL x{suite['fails']} (possibly a silent model update)")
if n_turns and n_err / n_turns > 0.1:
    alerts.append(f"chat error rate {n_err}/{n_turns}")
if vlm_down > 20:
    alerts.append(f"vlm_down x{vlm_down} (vision service frequently offline)")
if len(flagged) >= 2:
    alerts.append(f"content audit flagged {len(flagged)}/{len(judge_results)}")

# Always announce the report (owner request 2026-07-12), alerts on top if any.
digest = (
    f"[weekly] Chat weekly audit {today}\n"
    f"turns {n_turns} · sessions {n_sessions} · users {n_users} · errors {n_err}"
    + (f" · cap refusals {n_denied}" if n_denied else "") + "\n"
    + f"TTFB p50/p90: {pct(ttfbs, .5)}/{pct(ttfbs, .9)}ms\n"
    f"image turns {len(img_turns)} (intent: "
    + (", ".join(f"{k}x{v}" for k, v in sorted(intent_dist.items(), key=lambda kv: -kv[1])) or "none") + ")\n"
    f"suite: {suite['summary']} · content audit: "
    f"{len(judge_results) - len(flagged)}/{len(judge_results)} clean\n"
    f"full report: admin → Docs → chat-weekly-audit"
)
# 2026-07-27 (replacing the fixed weekly broadcast of 07-12): stay silent when
# there is no signal — only real traffic or an alert sends a Telegram message.
# With no external users and no alerts, the HTML report is still archived to
# admin → Docs, but the phone is not disturbed (same philosophy as db-backup).
send = bool(alerts) or n_turns > 0
if alerts:
    digest = "[alert] needs a look:\n- " + "\n- ".join(alerts) + "\n\n" + digest
    print(f"[audit] ALERTED: {alerts}", flush=True)
elif send:
    print(f"[audit] {n_turns} real turns — sending weekly digest", flush=True)
else:
    print("[audit] 0 real users this week and no alerts — Telegram silent (report archived to admin Docs)", flush=True)
if send:
    tg(digest)

# admin Runs visibility (fire-and-forget)
try:
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "skill_report.py"), "dispatch",
         "--skill", "chat-weekly-audit", "--invoked-by", "launchd",
         "--summary", json.dumps({"turns": n_turns, "suite_fails": suite["fails"],
                                  "flagged": len(flagged), "alerts": len(alerts)})],
        capture_output=True, timeout=30)
except Exception:
    pass

sys.exit(0)
