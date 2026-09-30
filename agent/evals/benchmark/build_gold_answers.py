#!/usr/bin/env python3
"""Construct gold reference answers for the benchmark queries (50, later 210).

Pipeline per query:
  1. AUTHOR (Claude Opus + MCP tools): plan tool calls, execute, draft answer
  2. CRITIC (Claude Sonnet 4.6, no tools): verify facts, format, refusal correctness
  3. REVISE (Claude Opus, no tools): rewrite if critic flagged issues — max 2 rounds

The author and critic use different models on purpose: cross-model disagreement
catches hallucinations that single-model self-review misses.

Output: data/benchmark/gold_answers.jsonl (one record per query)

Model access: every step shells out to a headless agent CLI that streams
stream-json events (assistant text / tool_use, tool_result, result) and accepts
an MCP server config. Its base command line comes from LLM_CLI_CMD.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Root of the production checkout (holds data/benchmark/ and chat-skills/mcp.json).
REPO_ROOT = Path(os.environ.get("WTL_ROOT", Path(__file__).resolve().parents[3]))
DATA_DIR = REPO_ROOT / "data" / "benchmark"
# v2 files contain the expanded 210-query set with anchor_id + split.
# Fall back to v1 if v2 doesn't exist (legacy / clean checkout).
_V2_REVIEW = DATA_DIR / "queries_review.v2.json"
_V2_SOURCE = DATA_DIR / "queries_filtered.v2.jsonl"
KEPT_PATH = _V2_REVIEW if _V2_REVIEW.exists() else DATA_DIR / "queries_review.json"
SOURCE_PATH = _V2_SOURCE if _V2_SOURCE.exists() else DATA_DIR / "queries_filtered.jsonl"
OUTPUT_PATH = DATA_DIR / "gold_answers.jsonl"
MCP_CONFIG = REPO_ROOT / "chat-skills" / "mcp.json"

# Base command of the headless agent CLI (binary + non-interactive/print flag),
# shell-split. Protocol flags below are appended by run_claude().
LLM_CLI_CMD = shlex.split(os.environ.get("LLM_CLI_CMD", ""))
ALLOWED_TOOLS = (
    "mcp__search_properties__search_properties "
    "mcp__get_postcode_scores__get_postcode_scores"
)

AUTHOR_MODEL = "opus"
CRITIC_MODEL = "sonnet"

# Per-step subprocess timeout. Author + tools is the slowest (multiple tool
# round-trips); critic and revise are pure text.
AUTHOR_TIMEOUT_S = 300
CRITIC_TIMEOUT_S = 300
REVISE_TIMEOUT_S = 300

MAX_REVISE_ROUNDS = 2


# ──────────────────────────────────────────────────────────────────────
# Subprocess wrapper
# ──────────────────────────────────────────────────────────────────────


def run_claude(
    prompt: str,
    *,
    model: str,
    system_prompt: str,
    use_tools: bool,
    timeout_s: int,
) -> dict[str, Any]:
    """Spawn the headless agent CLI, return {tool_calls, final_text, success, error}.

    Tool calls capture name + input + result for each MCP invocation.
    """
    if not LLM_CLI_CMD:
        return {"success": False, "error": "LLM_CLI_CMD is not set",
                "tool_calls": [], "final_text": ""}
    args = [
        *LLM_CLI_CMD,
        "--output-format", "stream-json",
        "--verbose",  # required for stream-json
        "--include-partial-messages",
        "--model", model,
        "--system-prompt", system_prompt,
    ]
    if use_tools:
        args += [
            "--mcp-config", str(MCP_CONFIG),
            "--allowed-tools", ALLOWED_TOOLS,
            "--strict-mcp-config",
        ]
    # When use_tools=False we don't pass --allowed-tools at all. The critic
    # and reviser prompts don't ask for tools, so the model won't call any.
    # Passing `--allowed-tools ""` tripped the CLI's prompt-arg parser.

    args.append(prompt)

    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(REPO_ROOT),
        )
    except subprocess.TimeoutExpired as e:
        return {
            "success": False,
            "error": f"timeout after {timeout_s}s",
            "tool_calls": [],
            "final_text": "",
        }

    tool_calls: list[dict[str, Any]] = []
    final_text_parts: list[str] = []
    pending_tool_use: dict[str, dict] = {}  # id -> {name, input}

    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev_type = ev.get("type")

        if ev_type == "assistant":
            msg = ev.get("message", {})
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for blk in content:
                btype = blk.get("type")
                if btype == "text":
                    text = blk.get("text", "")
                    if text:
                        final_text_parts.append(text)
                elif btype == "tool_use":
                    tid = blk.get("id")
                    name = blk.get("name", "")
                    # Only record OUR MCP tools — the CLI's built-in tools
                    # (tool search / shell / file read) leak through even with
                    # --strict-mcp-config and shouldn't pollute the gold record.
                    if tid and tid not in pending_tool_use and name.startswith("mcp__"):
                        pending_tool_use[tid] = {
                            "name": name,
                            "input": blk.get("input", {}),
                        }

        elif ev_type == "user":
            msg = ev.get("message", {})
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for blk in content:
                if blk.get("type") == "tool_result":
                    tid = blk.get("tool_use_id")
                    result_content = blk.get("content", "")
                    if isinstance(result_content, list):
                        result_content = "".join(
                            c.get("text", "") for c in result_content if isinstance(c, dict)
                        )
                    if tid and tid in pending_tool_use:
                        tu = pending_tool_use.pop(tid)
                        tool_calls.append({
                            "name": tu["name"],
                            "input": tu["input"],
                            "result": str(result_content)[:4000],
                            "is_error": blk.get("is_error", False),
                        })

        elif ev_type == "result":
            # Last event — overall summary
            if ev.get("is_error"):
                return {
                    "success": False,
                    "error": ev.get("result", "unknown")[:200],
                    "tool_calls": tool_calls,
                    "final_text": "".join(final_text_parts),
                }

    final_text = "".join(final_text_parts).strip()
    if not final_text and proc.returncode != 0:
        return {
            "success": False,
            "error": (proc.stderr or proc.stdout)[-300:],
            "tool_calls": tool_calls,
            "final_text": "",
        }
    return {
        "success": True,
        "tool_calls": tool_calls,
        "final_text": final_text,
        "error": None,
    }


# ──────────────────────────────────────────────────────────────────────
# Phase prompts
# ──────────────────────────────────────────────────────────────────────


AUTHOR_SYSTEM = """You are constructing a GOLD REFERENCE ANSWER for a benchmark query about UK property. The audience is a general UK home-buyer / renter / investor; your answer will be the standard against which other models are evaluated.

ROLE: You act as a GENERAL UK PROPERTY ASSISTANT — not limited to any single app's tool set. Anything property-relevant is in scope: area search, postcode evaluation, leasehold legal notes, conveyancing, BTL / yield, mortgages, planning, schools, lifestyle fit, etc. Only refuse if the query is genuinely off-domain (cooking, politics, programming, dating) or unethical (tax evasion, tenant discrimination).

LANGUAGE: Match the user's query language. Chinese (中文) → Chinese. English → English. Mixed → match the dominant language.

TOOLS AVAILABLE (use them whenever they can ground a factual claim):
- search_properties: query rm_sales_overview by beds, max_price, postcode_prefix, etc.
- get_postcode_scores: 5-dim livability scores (Transport / Community / Environment / Price / Schools) for a postcode.

TOOL BUDGET: at most 4 tool calls total. Stop as soon as you have enough.

TWO MODES OF GROUNDING — be explicit which is which:
1. TOOL-GROUNDED facts: anything specific (a listing, a postcode score, an asking price, a station match) MUST come from tool results and be cited like [tool=search_properties,n=N] or [postcode=E14 0GQ,Transport=72]. Never invent specific numbers.
2. GENERAL KNOWLEDGE: when tools can't answer (legal, financial, lifestyle, conveyancing, BTL, leasehold, planning), use widely-known UK property knowledge. Frame it explicitly — "Generally in the UK ...", "Industry rule of thumb ...", "Most lenders ...". Then point to a real authority for verification (a solicitor for legal, gov.uk Environment Agency for flood, ONS / Met Police for crime, etc.).

HEDGING RULES (these are hard constraints):
- NEVER cite specific UK statute names or section numbers (no "Section 38 of the 1995 Land Act"). For legal questions, give general principles + recommend a solicitor. The only acceptable specific names are widely-known current laws stated in plain form (Leasehold Reform (Ground Rent) Act 2022, Renters' Rights Act 2024, etc.) — and even then frame as "broadly speaking, the 2022 leasehold reform requires...".
- For yields / growth rates / mortgage rates / fees: give a typical range ("typically 4-6% gross" / "varies £300-£800 per year") instead of a single fabricated number, and disclaim "varies by area / lender / time".
- For tube/rail station names: only mention specific stations you're confident about (e.g. Canary Wharf is Jubilee + Elizabeth + DLR — well-known facts). When unsure, say "the nearest stations include X / Y, please verify line and zone via TfL".
- For boroughs / borough boundaries: stick to well-known mappings (Hackney is in Hackney; Walthamstow is in Waltham Forest, East London). When unsure, say "this area falls in [outer London / east / etc.] — confirm exact borough via the postcode".

ANSWER STRUCTURE:
- 80-300 words. Markdown bullets for lists.
- Lead with tool-grounded paragraphs if you have any.
- General-knowledge / hedged paragraphs after, clearly framed with "Generally..." / "Broadly..." etc.
- End with one "next step" or "verify with X" pointer where useful.
- For follow-up queries (turn 2+), reference the prior listing/postcode and use tools to gather more.

Refusal is reserved for off-domain only. If you do refuse, output exactly:
  REFUSE: <one-sentence reason — match query language>

Output: just the answer markdown (or REFUSE: line for off-domain). Nothing else."""

# Synthetic prior-turn context for follow_up class queries. The author can't
# resolve "this property..." or "this area..." (as the Chinese follow-up
# queries phrase it) without knowing which property/area was being
# discussed. We inject a real listing + a real evaluated postcode as the
# implicit prior turn so the gold answer can ground in actual data.
#
# Public copy: the property anchors were real rows from the listings table.
# Their listing ids, street/building names, full postcodes and the agent's
# description text are redacted to <placeholders> below; the outcode, price,
# size and tenure are kept so the prompts stay readable.

# Real anchor: a 2-bed riverside flat in Poplar, E14 (listings-table row)
PROPERTY_ANCHOR_CONTEXT = (
    "PRIOR TURN CONTEXT (this is turn 2 of a multi-turn conversation).\n"
    "At turn 1 the user shared this listing for evaluation:\n"
    "  • Listing id: <listing-id>\n"
    "  • Address: <development>, Poplar, London E14 <inward-code>\n"
    "  • Asking price: £585,000\n"
    "  • 2 bed, 1 bath, Flat, 788 sqft\n"
    "  • Tenure: LEASEHOLD, 995 years remaining\n"
    "  • Description: '<the agent's listing description: 2-bed apartment, high-spec finish, "
    "private balcony, riverside development near Canning Town station>'\n"
    "  • EPC rating: not in record\n"
    "  • Service charge / ground rent: not in record\n"
    "\nThe agent has search_properties (sales listings) and get_postcode_scores (5-dim livability) "
    "tools available. Use them to answer the user's follow-up question grounded in real data for "
    "this specific listing or its postcode (E14 <inward-code> / E14 prefix).\n\n"
    "User now asks: "
)

# Real anchor: postcode E14 9NA, evaluated and cached (verified from earlier
# get_postcode_scores test). Includes its 5-dim scores so follow-up area
# questions can reference real data.
AREA_ANCHOR_CONTEXT = (
    "PRIOR TURN CONTEXT (this is turn 2 of a multi-turn conversation).\n"
    "At turn 1 the user asked about postcode E14 9NA (Isle of Dogs / Canary Wharf area). "
    "The agent ran get_postcode_scores and found:\n"
    "  • Price: 52.5 / 100\n"
    "  • Schools: 40.8 / 100\n"
    "  • Transport: strong (DLR + Jubilee + Elizabeth line; South Quay 4.6 min walk)\n"
    "  • Other dimensions cached 2026-04-25\n"
    "  • Repeat-sales: ~1.96% CAGR over 10 years (slow growth)\n"
    "\nThe agent has search_properties and get_postcode_scores tools available. Use them to "
    "answer the user's follow-up about this postcode/neighbourhood (E14 9NA, the Isle of "
    "Dogs / Canary Wharf area) grounded in real data when possible.\n\n"
    "User now asks: "
)


# ─── Multi-anchor registry (must match expand_curated_queries.py) ──────
# When a query record carries an `anchor_id` field, render the matching
# block below. Without `anchor_id`, fall back to the legacy single-anchor
# behaviour (A1 for property_eval, E14 9NA for area_eval).

PROPERTY_ANCHOR_BLOCKS: dict[str, str] = {
    "A1": PROPERTY_ANCHOR_CONTEXT,  # legacy default — Poplar E14 £585k 2-bed Flat leasehold
    "A2": (
        "PRIOR TURN CONTEXT (this is turn 2 of a multi-turn conversation).\n"
        "At turn 1 the user shared this listing for evaluation:\n"
        "  • Listing id: <listing-id>\n"
        "  • Address: <street>, Ladywell, London SE13 <inward-code>\n"
        "  • Asking price: £300,000\n"
        "  • 1 bed, 1 bath, Maisonette\n"
        "  • Tenure: typical Maisonette (could be leasehold or share-of-freehold; not on the listing summary)\n"
        "  • Profile: first-time buyer territory in Lewisham; older/conversion stock\n"
        "\nThe agent has search_properties + get_postcode_scores. Use them to ground "
        "answers about this listing or its postcode (SE13 <inward-code> / SE13).\n\n"
        "User now asks: "
    ),
    "A3": (
        "PRIOR TURN CONTEXT (this is turn 2 of a multi-turn conversation).\n"
        "At turn 1 the user shared this listing for evaluation:\n"
        "  • Listing id: <listing-id>\n"
        "  • Address: <street>, London E14 <inward-code>\n"
        "  • Asking price: £635,000\n"
        "  • 2 bed Flat\n"
        "  • Tenure: LEASEHOLD\n"
        "  • Profile: modern Docklands flat, similar segment to A1 but a different building\n"
        "\nUse tools to ground answers about this listing or postcode (E14 <inward-code> / E14 prefix).\n\n"
        "User now asks: "
    ),
    "A4": (
        "PRIOR TURN CONTEXT (this is turn 2 of a multi-turn conversation).\n"
        "At turn 1 the user shared this listing for evaluation:\n"
        "  • Listing id: <listing-id>\n"
        "  • Address: <street>, W6 <inward-code> (Hammersmith)\n"
        "  • Asking price: £1,000,000\n"
        "  • 3 bed, Terraced (Period property — Victorian/Edwardian era)\n"
        "  • 1,365 sq ft\n"
        "  • Tenure: FREEHOLD\n"
        "  • Profile: classic Period family house with extension potential, conservation-area sensitive\n"
        "\nUse tools to ground answers about this property or postcode (W6 <inward-code> / W6 prefix).\n\n"
        "User now asks: "
    ),
}

AREA_ANCHOR_BLOCKS: dict[str, str] = {
    "Z1": AREA_ANCHOR_CONTEXT,  # legacy — E14 9NA Isle of Dogs
    "Z2": (
        "PRIOR TURN CONTEXT (this is turn 2 of a multi-turn conversation).\n"
        "At turn 1 the user asked about postcode N1 9AA (Islington / King's Cross side). "
        "The agent ran get_postcode_scores and found:\n"
        "  • Safety: 42 / 100  (notably low — central urban)\n"
        "  • Commute: 86 / 100 (excellent transport — King's Cross hub)\n"
        "  • Price: 77 / 100\n"
        "  • Profile: gentrified central-north London, dense, mixed demographics\n"
        "\nUse tools to ground answers about this postcode/area (N1 9AA / N1 prefix).\n\n"
        "User now asks: "
    ),
    "Z3": (
        "PRIOR TURN CONTEXT (this is turn 2 of a multi-turn conversation).\n"
        "At turn 1 the user asked about postcode N5 1FL (Highbury). "
        "The agent ran get_postcode_scores and found:\n"
        "  • Safety: 80 / 100  (very good — quieter residential)\n"
        "  • Commute: 70 / 100 (good — Victoria + Piccadilly nearby)\n"
        "  • Price: 75 / 100\n"
        "  • Profile: family-friendly residential pocket, good schools, lower density than N1\n"
        "\nUse tools to ground answers about this postcode/area (N5 1FL / N5 prefix).\n\n"
        "User now asks: "
    ),
    "Z4": (
        "PRIOR TURN CONTEXT (this is turn 2 of a multi-turn conversation).\n"
        "At turn 1 the user asked about postcode EC4M 8AD (St Paul's). "
        "The agent ran get_postcode_scores and found:\n"
        "  • Safety: 55 / 100  (average — busy central commercial area)\n"
        "  • Commute: 90 / 100 (Central line, Thameslink and Bank nearby)\n"
        "  • Price: 60 / 100\n"
        "  • Profile: City of London fringe; mostly offices, some new-build flats\n"
        "\nUse tools to ground answers about this postcode/area (EC4M 8AD / EC4M prefix).\n\n"
        "User now asks: "
    ),
}


def get_prior_context(candidate_id: str, intent_subtype: str | None,
                      anchor_id: str | None = None) -> str:
    """Return the appropriate prior-turn context for a follow-up query.

    If `anchor_id` is provided (for queries authored after the v2 expand),
    render the matching anchor block from the registry. Otherwise fall back
    to the legacy single-anchor behaviour.
    """
    # Original test follow-ups (kept for backward compat with manual entries)
    if candidate_id == "manual_followup_1":
        return (
            "PRIOR TURN CONTEXT (this is turn 2):\n"
            "User asked at turn 1: 'Find 2-bed flats in N1 under £700k'\n"
            "search_properties returned 4 listings (real-looking IDs).\n\n"
            "User now asks: "
        )
    if candidate_id == "manual_followup_2":
        return (
            "PRIOR TURN CONTEXT (this is turn 2):\n"
            "User asked at turn 1: 'Compare W11 and W2 for first-time buyers'\n"
            "Both postcodes returned 5-dim scores from get_postcode_scores.\n\n"
            "User now asks: "
        )
    # v2 queries with an explicit anchor_id
    if anchor_id and anchor_id in PROPERTY_ANCHOR_BLOCKS:
        return PROPERTY_ANCHOR_BLOCKS[anchor_id]
    if anchor_id and anchor_id in AREA_ANCHOR_BLOCKS:
        return AREA_ANCHOR_BLOCKS[anchor_id]
    # Legacy fallback: subtype determines which default anchor
    if intent_subtype == "property_eval":
        return PROPERTY_ANCHOR_CONTEXT
    if intent_subtype == "area_eval":
        return AREA_ANCHOR_CONTEXT
    return ""


CRITIC_SYSTEM = """You are an independent QA reviewer for a UK property-assistant gold benchmark. You receive: the user query, the tool calls + results that were executed, and the candidate gold answer.

The gold answer should behave as a GENERAL UK property assistant — answer anything property-relevant, refuse only fully off-domain queries (cooking, politics, programming).

Check 6 categories of issue. For each issue, output one entry:
- category:
  - "tool_hallucination" — specific listing / postcode / score / price claim NOT supported by tool results
  - "general_hallucination" — non-tool claim that's factually wrong by external standards (fake statute name, non-existent tube station, wrong borough mapping, fake institution address)
  - "missing_hedge" — gave a specific number (yield / price / growth rate / fee) without tool grounding AND without a "typical range / varies by X" disclaimer
  - "off_domain_answered" — answered a fully off-domain query instead of refusing (rare; only for cooking/politics/programming/dating)
  - "format" — markdown / word count issues (target 80-300 words; <40 too thin, >400 too long; missing tool citations on tool-grounded paragraphs)
  - "incomplete" — missed obvious tool data the agent had access to
- severity: "high" (factually misleading or fully wrong) | "medium" (suboptimal) | "low" (cosmetic)
- detail: 1-sentence description.

Output format — strict JSON, no markdown fences:
{"issues": [{"category":"...","severity":"...","detail":"..."}], "verdict": "pass" | "needs_revision"}

Rules:
- "pass" verdict requires: zero high-severity issues AND ≤1 medium issue.
- For "general_hallucination", verify against widely-known UK property facts (TfL line / station list, real UK statute names, gov.uk Environment Agency categories, ONS / Met Police data structure). DO NOT mark a knowledge claim as hallucination just because it isn't in the tool results — only if it contradicts external reality.
- A specific UK statute name + section number is "general_hallucination" UNLESS clearly real and current (Leasehold Reform (Ground Rent) Act 2022, Housing Act 1988, Renters' Rights Act 2024, etc.). When in doubt, flag it.
- A property-related query is answerable by default. Only flag "off_domain_answered" for genuinely off-topic queries.
- Refusing a property-related query is itself a high-severity issue — flag as "off_domain_answered" with detail noting "should have answered with general knowledge".
"""


REVISER_SYSTEM = """You are revising a UK property gold reference answer based on critic feedback.

Rules:
- Address every issue listed by the critic.
- For "tool_hallucination": remove the unsupported claim or replace with a tool-grounded fact.
- For "general_hallucination": fix the factual error (correct statute name / station / borough) or remove the wrong claim entirely.
- For "missing_hedge": add explicit hedging — convert single specific number into "typical range" or "varies by X", and add "verify with [authority]" pointer.
- For "off_domain_answered": change to "REFUSE: <one-sentence reason — match query language>".
- For "format" / "incomplete": fix per detail.
- DO NOT introduce new tool-grounded facts not present in the tool results.
- You MAY introduce general UK property knowledge with appropriate hedging ("Generally...", "Industry rule of thumb...") — that's expected.

Output: just the revised markdown (or REFUSE: line for off-domain only)."""


def make_author_prompt(query: str, candidate_id: str = "", intent_subtype: str | None = None,
                       anchor_id: str | None = None) -> str:
    prior = get_prior_context(candidate_id, intent_subtype, anchor_id=anchor_id)
    if prior:
        return f"{prior}{query}\n\nUse tools to ground your answer in real data for the specific property/postcode from the prior turn. Reference the prior context when answering deictic questions ('this property', 'this area')."
    return f"User query:\n\n{query}\n\nUse tools to ground your answer in real data, then write the gold reference answer per the system instructions."


def make_critic_prompt(query: str, tool_calls: list[dict], draft: str) -> str:
    # Result cap was 1200 chars; bumped to 5000 after critic-calibration audit
    # found that a 25-listing search_properties result (~3700 chars) was being
    # truncated to first 8 listings. Critic then flagged answers that cited
    # listings 9+ as hallucinations. 5000 fits a full 25-result search.
    tools_summary = "\n\n".join(
        f"### Tool {i+1}: {tc['name']}\nInput: {json.dumps(tc['input'])[:500]}\nResult: {tc['result'][:5000]}"
        for i, tc in enumerate(tool_calls)
    ) if tool_calls else "(no tools were called)"
    return (
        f"## User Query\n{query}\n\n"
        f"## Tool calls executed\n{tools_summary}\n\n"
        f"## Candidate gold answer\n{draft}\n\n"
        "Output the JSON verdict per the system instructions."
    )


def make_reviser_prompt(query: str, tool_calls: list[dict], draft: str, issues: list[dict]) -> str:
    tools_summary = "\n".join(
        f"- {tc['name']}({json.dumps(tc['input'])[:200]}) → {tc['result'][:300]}"
        for tc in tool_calls
    ) if tool_calls else "(no tools)"
    issues_summary = "\n".join(f"- [{iss['severity']}/{iss['category']}] {iss['detail']}" for iss in issues)
    return (
        f"## User Query\n{query}\n\n"
        f"## Tool results\n{tools_summary}\n\n"
        f"## Prior draft\n{draft}\n\n"
        f"## Issues to fix\n{issues_summary}\n\n"
        "Output the revised gold answer."
    )


# ──────────────────────────────────────────────────────────────────────
# Critic JSON parsing — must be defensive
# ──────────────────────────────────────────────────────────────────────


def parse_critic(text: str) -> dict[str, Any]:
    text = text.strip()
    # Strip possible ```json fences just in case
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        obj = json.loads(text)
        issues = obj.get("issues", [])
        verdict = obj.get("verdict", "needs_revision")
        if not isinstance(issues, list):
            issues = []
        return {"issues": issues, "verdict": verdict, "parse_ok": True}
    except json.JSONDecodeError as e:
        return {
            # Severity is "low" not "high" — a critic output JSON parse failure
            # is critic self-noise, not a real issue with the model's answer.
            # Don't pollute high-severity counts with these. (Auto-fixed
            # 2026-05-09 after critic-calibration audit found 5/8 of gold v2's
            # high-severity flags were actually JSON parse failures.)
            "issues": [{"category": "incomplete", "severity": "low", "detail": f"critic output was unparseable: {e}"}],
            "verdict": "needs_revision",
            "parse_ok": False,
            "raw": text[:500],
        }


# ──────────────────────────────────────────────────────────────────────
# Per-query orchestration
# ──────────────────────────────────────────────────────────────────────


def process_query(query_text: str, candidate_id: str, intent_subtype: str | None = None,
                  anchor_id: str | None = None) -> dict[str, Any]:
    """Run the full pipeline for one query. Returns gold record dict."""
    record: dict[str, Any] = {
        "candidate_id": candidate_id,
        "query": query_text,
        "intent_subtype": intent_subtype,
        "anchor_id": anchor_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "auto_flags": [],
    }

    # Phase 1: Author + tools
    author = run_claude(
        prompt=make_author_prompt(query_text, candidate_id, intent_subtype, anchor_id=anchor_id),
        model=AUTHOR_MODEL,
        system_prompt=AUTHOR_SYSTEM,
        use_tools=True,
        timeout_s=AUTHOR_TIMEOUT_S,
    )
    record["tool_calls"] = author["tool_calls"]
    if not author["success"]:
        record["auto_flags"].append(f"author_failed:{author.get('error', '')[:100]}")
        record["draft_answer"] = ""
        record["final_answer"] = ""
        record["status"] = "error"
        return record

    draft = author["final_text"]
    record["draft_answer"] = draft
    # follow_up queries legitimately may answer from prior turn context
    # without calling tools; not a flag for that class.
    is_follow_up = (
        candidate_id.startswith("manual_followup_")
        or intent_subtype in ("property_eval", "area_eval")
    )
    if not author["tool_calls"] and not is_follow_up:
        record["auto_flags"].append("no_tools_called")
    if draft.startswith("REFUSE:"):
        record["should_refuse"] = True
        # Skip critic — refusals don't need fact-checking. A refusal is the
        # gold answer for queries that legitimately fall outside the tool
        # surface (mortgage strategy, conveyancing, etc).
        record["final_answer"] = draft
        # No tools called is expected for refusal — drop that flag
        record["auto_flags"] = [f for f in record["auto_flags"] if f != "no_tools_called"]
        record["status"] = "ok_refusal"
        return record

    record["should_refuse"] = False

    # Phase 2-3: Critic loop
    critic_history: list[dict] = []
    current = draft
    for round_n in range(MAX_REVISE_ROUNDS + 1):  # initial + 2 revises = 3 critic checks
        critic_raw = run_claude(
            prompt=make_critic_prompt(query_text, author["tool_calls"], current),
            model=CRITIC_MODEL,
            system_prompt=CRITIC_SYSTEM,
            use_tools=False,
            timeout_s=CRITIC_TIMEOUT_S,
        )
        if not critic_raw["success"]:
            record["auto_flags"].append(f"critic_failed_round_{round_n}:{critic_raw.get('error','')[:100]}")
            critic_history.append({
                "round": round_n,
                "verdict": None,
                "issues": [],
                "error": critic_raw.get("error"),
                "raw_text": critic_raw.get("final_text", "")[:500],
            })
            break
        verdict = parse_critic(critic_raw["final_text"])
        critic_history.append({
            "round": round_n,
            "verdict": verdict.get("verdict"),
            "issues": verdict.get("issues"),
            "parse_ok": verdict.get("parse_ok"),
        })

        if verdict.get("verdict") == "pass":
            break
        if round_n >= MAX_REVISE_ROUNDS:
            record["auto_flags"].append("critic_exhausted_revisions")
            break

        # Revise
        revised = run_claude(
            prompt=make_reviser_prompt(query_text, author["tool_calls"], current, verdict["issues"]),
            model=AUTHOR_MODEL,
            system_prompt=REVISER_SYSTEM,
            use_tools=False,
            timeout_s=REVISE_TIMEOUT_S,
        )
        if not revised["success"]:
            record["auto_flags"].append(f"reviser_failed_round_{round_n}")
            break
        current = revised["final_text"]

    record["final_answer"] = current
    record["critic_history"] = critic_history
    record["status"] = "ok" if not record["auto_flags"] else "needs_human_review"
    return record


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def load_kept_queries() -> list[dict]:
    """Return list of {candidate_id, query, final_class, intent_subtype} for kept queries."""
    if not KEPT_PATH.exists() or not SOURCE_PATH.exists():
        sys.exit(f"Missing {KEPT_PATH} or {SOURCE_PATH}")
    decisions = json.loads(KEPT_PATH.read_text())
    by_id = {}
    for line in SOURCE_PATH.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        by_id[rec["candidate_id"]] = rec
    out = []
    for cid, dec in decisions.items():
        if dec.get("status") != "kept":
            continue
        if cid not in by_id:
            continue
        out.append({
            "candidate_id": cid,
            "query": by_id[cid]["primary_query"],
            "final_class": dec.get("final_class") or by_id[cid].get("filter_class"),
            "intent_subtype": by_id[cid].get("intent_subtype"),
            # v2 fields (None for legacy entries)
            "anchor_id": by_id[cid].get("anchor_id"),
            "split": by_id[cid].get("split", "train"),
        })
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--single", help="Test on one candidate_id only")
    p.add_argument("--limit", type=int, help="Process at most N queries")
    p.add_argument("--resume", action="store_true", help="Skip queries already in output JSONL")
    args = p.parse_args()

    queries = load_kept_queries()
    if args.single:
        queries = [q for q in queries if q["candidate_id"] == args.single]
        if not queries:
            print(f"No kept query with id {args.single}", file=sys.stderr)
            return 1

    if args.resume and OUTPUT_PATH.exists():
        existing = {json.loads(line)["candidate_id"]
                    for line in OUTPUT_PATH.read_text().splitlines() if line.strip()}
        queries = [q for q in queries if q["candidate_id"] not in existing]
        print(f"Resuming: {len(queries)} remaining (skipped {len(existing)} already done)")
    elif not args.single and not args.resume:
        OUTPUT_PATH.unlink(missing_ok=True)

    if args.limit:
        queries = queries[: args.limit]

    print(f"Processing {len(queries)} queries → {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("a") as f:
        for i, q in enumerate(queries, start=1):
            anchor_str = f" anchor={q.get('anchor_id')}" if q.get("anchor_id") else ""
            split_str = f" split={q.get('split', 'train')}"
            print(f"[{i}/{len(queries)}] {q['candidate_id']} ({q['final_class']}/{q.get('intent_subtype', '-')}{anchor_str}{split_str}): {q['query'][:60]}...", flush=True)
            rec = process_query(q["query"], q["candidate_id"], q.get("intent_subtype"),
                                anchor_id=q.get("anchor_id"))
            rec["query_class"] = q["final_class"]
            rec["split"] = q.get("split", "train")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"   → status={rec['status']} flags={rec['auto_flags']} tools={len(rec['tool_calls'])}", flush=True)

    print(f"\nDone. Output: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
