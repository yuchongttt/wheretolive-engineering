# agent/evals — how the property assistant is evaluated and improved

The chat assistant ("Arden") answers London property questions in English and Chinese. In
production it is Claude (Sonnet) calling 27 MCP tools over a SQLite database (the tool
layer is in [`../tools/`](../tools/)). Three things make it hard to test:

- **Output is stochastic.** The same prompt does not give the same answer twice, and a
  silent model update behind the `sonnet` alias can change behaviour overnight.
- **The data moves.** About 8% of listings turn over per week (measured 2026-08-13: 5,514 of
  70,190 active listings delisted, 4,887 added in seven days), so a test pinned to one flat
  goes red because the flat sold.
- **The costly failures are quiet.** A fabricated number, a confident identification from one
  clue, an invented cause for a data anomaly, a tool error swallowed — all read fluently.

So the evaluation is a stack of layers, cheapest and most deterministic first; each catches
something the others cannot. Layer 1 lives with the tools in `../tools/`.

| # | layer | where | catches | cost / when |
|---|---|---|---|---|
| 1 | tool unit tests | `../tools/` | wrong SQL, filters, formatting | offline, every change |
| 2 | structural end-to-end assertions | `harness/agent_eval.py`, `harness/selftest_chat.py` | wrong / missing tool calls and args, errors, truncated streams | live server, minutes per case |
| 3 | needle and capability probes | `harness/needle_test*.py`, `loop/capability_gaps.py` | retrieval that cannot find a known listing; gaps the agent admits to | needles: live server; gaps: zero tokens |
| 4 | LLM-as-critic grading | `benchmark/` | tool hallucination, wrong facts, missing hedges, format | one critic call per answer |
| 5 | behaviour contract suite | `behaviour/chat_behavior_suite.py` | honesty, routing and ordering contracts | ~14 turns, 15-25 min |
| 6 | weekly audit | `loop/` | drift in real traffic; quality of real answers | weekly, scheduled |

## Layer 2 — structural assertions on the tool-call trace

`agent_eval.py` posts to the **real** `/api/chat` route (admin-key bypass), parses the SSE
stream and records every `tool_call` (name + args) and the answer. It never re-implements the
agent, so it cannot drift from what users hit. Assertions are deliberately structural —
`tools_any` / `tools_all` / `tools_none`, per-argument checks (`equals`, `in`, `gte`, …),
minimum answer length, no error event, stream reached `done` — and the full trace is printed
for a human. Exit code is non-zero on any failure. `selftest_chat.py` runs one ad-hoc turn and
reads it back from `chat_messages`. *Limits:* three built-in cases; blind to whether the content
is true.

## Layer 3 — needle and capability probes

`needle_test.py` writes strict but realistic queries for known listings **without a
postcode** ("within 40 minutes of Stratford"), leaning on image-derived attributes (floorplan
rooms / garden / garage, decor score). Two signals per needle: *search recall* — replay each
search call's args against the DB with the tool's WHERE semantics ("could the agent's query
find it?") — and *exact surfacing* — the target's listing id is in the answer (the street name
is only a soft signal: it false-positives on a sibling unit). `needle_test_cn.py` runs the
same needles in Chinese. Target ids are redacted in this copy.

`capability_gaps.py` mines `chat_messages` for turns where the agent **admits** a gap ("my
tools have no field for …") and for "explicit search request with enough constraints, zero
tool calls". It exists because one such admission sat unread for five weeks. It drops failed
turns (errors are not behaviour), guards against negated sentences, dedups by
normalised question, and ranks hard gaps (no data touched) before partial ones.

## Layer 4 — LLM-as-critic grading (the benchmark)

- **Queries:** 50 hand-written (Chinese; a first set mined from public forum posts was
  off-target and dropped), expanded to 210 with anchor listings/postcodes for follow-ups and a
  train / test_query / test_anchor split (held-out anchors test generalisation).
- **Gold:** `build_gold_answers.py` — Claude Opus drafts with two MCP tools, Claude Sonnet
  critiques without tools, Opus revises (max 2 rounds). Author and critic are different
  models on purpose: cross-model disagreement catches what self-review misses. The gold set is used only to score answers.
- **Rubric** (`CRITIC_SYSTEM`): six categories — `tool_hallucination`,
  `general_hallucination`, `missing_hedge`, `off_domain_answered`, `format`, `incomplete` —
  each `high` / `medium` / `low`; **pass = zero high and at most one medium**; strict JSON out.
- **Scoring any model:** `critic_score.py` applies the same critic to any answer file, with a
  SQLite cache keyed on sha256(query, answer, tool calls, `CRITIC_VERSION`), and re-executes
  tools whose stored result looks truncated. Unparseable critic output is recorded as a *low*
  issue, not a high one (a calibration audit found 5 of 8 "high" gold flags were parse errors).
- **Baselines:** local Qwen3.5-9B via Ollama with and without the same tools (5 prompt
  iterations), plus a thinking-mode A/B. `summarize_results.py` turns the run files into
  tables. **Numbers and ten caveats are in [RESULTS.md](RESULTS.md)** — including two critic
  blind spots found while writing it (it never sees the follow-up context, and it only sees
  the first 5,000 chars of each tool result).

## Layer 5 — behaviour contract suite

About 25 contract cases (T1-T22, WEB1-5) for rules that live in the prompt and re-roll on every
model change: speak before the first tool call; batch independent lookups in parallel; no
re-search crawl on zero results; route images by intent; hedge single-clue identifications;
report both figures when two sources disagree (without calling the listing false); treat a
listing description as data (prompt injection must not trigger `create_radar`); flag that
"offers in excess of" makes asking a floor; check a user's false premise against live stock;
name a recorded cause of a data anomaly but never invent one; search first when constraints
suffice (plus the reverse case); re-measure commute minutes when the destination changes.

Design decisions, each learned from a wrong verdict:
- **Two tiers.** Hard = structural facts (event order, tool names/counts, telemetry rows).
  Soft/WARN = wording regexes, read by a human. Several mechanical wording checks gave the
  wrong verdict in one day, all too harsh.
- **Nothing pinned.** Subjects are chosen by SQL at run time; expected numbers are computed
  from the same DB in the same run (or from what the tool returned in that turn).
- **Fixtures we control** for contracts real stock cannot provide (a contradicted service
  charge, an injection): inserted into the live DB under a reserved id prefix, already
  delisted, removed in `finally`, and swept at startup if older than an hour.
- **Retry before crying wolf:** a failed hard check re-runs once; a pass on try 2 says so.
- **No false greens:** a selector that matches nothing exits 2; skips are recorded; every run
  is written to a run-history table so flakiness can be measured before scheduling.

## Layer 6 — weekly audit

`chat_weekly_audit.py` (Mondays): runs the suite, summarises 7 days of real traffic (error
rate, TTFB p50/p90, image-intent funnel, tool usage) and has an LLM judge re-read a sample of
real turns for *traceable* numbers, *evidence-gated* identifications and *disclosed*
inferences. HTML report to the admin docs; a Telegram message only when there is a signal.

## What is still manual

The behaviour suite is triggered by hand (owner decision: schedule it once its recorded run
history shows it is stable); kill-switch and hosted-page injection checks are a runbook. Every
WARN, judge flag and `needs_human` finding is read by a person. Benchmark runs and their reading
(RESULTS.md) were manual; there is no CI gate on critic scores.

**Known limitations:** non-determinism (single runs, no confidence intervals); critic parse
failures (8-20% of answers); critic blind spots and errors (RESULTS.md findings 5, 6, 10);
cost (one LLM call per critic verdict, up to three per gold item plus author and revisions;
~14 production turns per suite run); the judge samples only 10 real turns a week.

## Running the tests

```bash
cd agent/evals && python -m pytest -q     # 49 tests, offline: temp SQLite DBs + a synthetic JSONL fixture
```

Python >= 3.10, standard library only; `pytest` for the tests. The scripts need a running
deployment: the web app's `/api/chat` and SQLite DB, `curl` (agent_eval), Ollama (Qwen
baselines), a headless agent CLI (gold, critic).

## Environment variables

| var | used by | meaning |
|---|---|---|
| `WTL_ROOT` | all scripts | root of the production checkout (`data/evaluations.db`, `web/…`, `scripts/`) |
| `ADMIN_KEY` | harness, behaviour | admin bypass key for `/api/chat` (falls back to `web/.env.local`) |
| `CHAT_EVAL_URL`, `BASE` | agent_eval, behaviour | chat server, default `http://localhost:3000` |
| `LISTING_URL_TEMPLATE` | behaviour | listing-page URL with `{id}` (WEB5, T15, T17) |
| `BHV_ASSETS_DIR`, `BHV_T21_OWNER_ADDRESS`, `BHV_INVOKED_BY` | behaviour | image fixtures; T21 subject (redacted here, case skipped if unset); run-history label |
| `LLM_CLI_CMD` | benchmark | base command of the headless agent CLI (stream-json output) |
| `AUDIT_JUDGE_CMD`, `SAMPLE`, `SKIP_SUITE` | weekly audit | judge command (prompt appended), sample size, skip the suite |
| `WTL_INTERNAL_USER_IDS` | audit, driver | comma-separated internal account ids to exclude |

**Not included:** raw benchmark files and chat data (tables are in RESULTS.md); the forum-post
miner and its filter; tool regression scripts that need the production DB (see `../tools/`);
DB-dependent guard tests; the audit's product-analytics section; the Telegram and run-history
helpers (imported lazily from the production checkout).
