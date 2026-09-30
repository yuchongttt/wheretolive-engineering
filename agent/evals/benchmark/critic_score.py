#!/usr/bin/env python3
"""Run the v2 gold critic on any model's answer JSONL.

Lets us auto-score new Qwen baselines (with/without tools, prompt
variants, etc.) against the same critic that v2 gold uses,
without re-running gold.

Input JSONL schema (per line):
  {
    "candidate_id": "curated_07",
    "query": "...",
    "response": "<model's answer>",
    "tool_calls": [{"name": "...", "args": {...}, ...}],   # optional, default []
    "model": "qwen3.5:9b",                                  # optional
    ...rest ignored
  }

Output JSONL schema (per line):
  {
    "candidate_id": "curated_07",
    "verdict": "pass" | "needs_revision",
    "issues": [{"category": "...", "severity": "...", "detail": "..."}],
    "high_count": 0, "medium_count": 1, "low_count": 0,
    "category_counts": {"tool_hallucination": 1, ...},
    "scored_at": "2026-...",
    "wall_s": 12.3
  }

Aggregate stats printed to stderr at the end.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Reuse gold pipeline's critic prompt + LLM CLI wrapper + parser
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_gold_answers import (  # noqa: E402
    CRITIC_SYSTEM,
    CRITIC_TIMEOUT_S,
    CRITIC_MODEL,
    make_critic_prompt,
    parse_critic,
    run_claude,
)
# Reuse Qwen+MCP runner's tool implementations to backfill full results
# when an answer JSONL stored only a truncated result_preview (from earlier
# runs before the critic-calibration audit). Re-executes the tool with the
# same args to give critic the complete view, eliminating ~10-15 of the
# truncation-driven false-positive tool_hallucination flags.
from run_qwen_9b_off_mcp import execute_tool as _execute_tool  # noqa: E402


# A "result" is considered truncated if it's clearly cut short (<800 chars)
# AND wasn't a tool error / "not yet evaluated" message. Those are legitimate
# short outputs and don't need re-execution.
TRUNCATION_THRESHOLD = 800


def _normalise_tool_name(name: str) -> str:
    """Strip MCP prefixes — gold v2 stores names as
    'mcp__search_properties__search_properties' (the JSON-RPC server name),
    but our local impls register as plain 'search_properties'. Without this
    normalisation, hydration would re-execute as 'unknown tool' and mark
    every Sonnet cite as a tool_hallucination.
    """
    if name.startswith("mcp__"):
        # Format: mcp__<server_name>__<tool_name>
        parts = name.split("__")
        if len(parts) >= 3:
            return parts[-1]
    return name


def hydrate_tool_result(name: str, args: dict, stored: str) -> str:
    """Return full tool result, re-executing if `stored` looks truncated."""
    if stored and len(stored) >= TRUNCATION_THRESHOLD:
        return stored
    # Skip re-execute if the stored text is clearly a complete short message
    if stored and any(marker in stored for marker in (
        "No matches found.",
        "not yet evaluated",
        "Missing postcode parameter",
        "Error executing",
    )):
        return stored
    try:
        return _execute_tool(_normalise_tool_name(name), args or {})
    except Exception as e:
        # Fall back to whatever was stored
        return stored or f"(re-execute failed: {e})"

# Root of the production checkout (critic_cache lives in data/evaluations.db).
REPO_ROOT = Path(os.environ.get("WTL_ROOT", Path(__file__).resolve().parents[3]))
DATA_DIR = REPO_ROOT / "data" / "benchmark"
CRITIC_DB = REPO_ROOT / "data" / "evaluations.db"

# Bumped manually whenever CRITIC_SYSTEM, parse_critic, or
# make_critic_prompt change in a way that would alter verdicts. Cache
# entries with a different version are ignored. Keep in sync with
# build_gold_answers.py edits.
CRITIC_VERSION = "v2-2026-05-10"


def _cache_key(query: str, response: str, tool_calls: list[dict]) -> str:
    """Stable hash of the inputs critic actually sees."""
    h = hashlib.sha256()
    h.update(query.encode("utf-8"))
    h.update(b"\x00")
    h.update(response.encode("utf-8"))
    h.update(b"\x00")
    # Include tool name+args+result to be safe — if any of these change
    # the critic's verdict could legitimately differ.
    for tc in tool_calls:
        h.update(json.dumps({
            "name": tc.get("name", ""),
            "input": tc.get("input", {}),
            "result": tc.get("result", ""),
        }, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        h.update(b"\x00")
    h.update(CRITIC_VERSION.encode("utf-8"))
    return h.hexdigest()


def _cache_lookup(key: str) -> dict | None:
    try:
        conn = sqlite3.connect(str(CRITIC_DB))
        try:
            row = conn.execute(
                "SELECT verdict, issues_json, high_count, medium_count, low_count, "
                "category_counts_json, scored_at, wall_s FROM critic_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        return {
            "verdict": row[0],
            "issues": json.loads(row[1]),
            "high_count": row[2],
            "medium_count": row[3],
            "low_count": row[4],
            "category_counts": json.loads(row[5]),
            "scored_at": row[6],
            "wall_s": row[7] or 0.0,
            "_cached": True,
        }
    except Exception:
        return None


def _cache_store(key: str, scored: dict) -> None:
    try:
        conn = sqlite3.connect(str(CRITIC_DB))
        try:
            conn.execute(
                "INSERT OR REPLACE INTO critic_cache (cache_key, verdict, issues_json, "
                "high_count, medium_count, low_count, category_counts_json, scored_at, wall_s) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    scored["verdict"],
                    json.dumps(scored["issues"], ensure_ascii=False),
                    scored["high_count"],
                    scored["medium_count"],
                    scored["low_count"],
                    json.dumps(scored["category_counts"], ensure_ascii=False),
                    scored["scored_at"],
                    scored["wall_s"],
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        print(f"  ! cache store failed: {e}", file=sys.stderr)


def normalise_tool_calls(raw: list) -> list[dict]:
    """Coerce any model's tool-call shape to {name, input, result} for critic prompt.

    If `result` is missing/truncated (older Qwen+MCP runs stored only first
    300 chars as `result_preview`), re-execute the tool to give critic the
    complete result. This is the fix for the truncation false-positive
    documented in the critic-calibration report.
    """
    out = []
    for tc in raw or []:
        name = tc.get("name") or tc.get("function", {}).get("name", "?")
        args = tc.get("args") or tc.get("input") or tc.get("function", {}).get("arguments", {}) or {}
        stored = tc.get("result") or tc.get("result_preview") or ""
        result = hydrate_tool_result(name, args, stored)
        out.append({"name": name, "input": args, "result": result})
    return out


def score_one(rec: dict, use_cache: bool = True) -> dict:
    query = rec.get("query", "")
    response = rec.get("response") or rec.get("final_answer") or ""
    tool_calls = normalise_tool_calls(rec.get("tool_calls", []))
    if not response:
        return {
            "candidate_id": rec.get("candidate_id"),
            "verdict": "needs_revision",
            "issues": [{"category": "format", "severity": "high", "detail": "empty response"}],
            "high_count": 1, "medium_count": 0, "low_count": 0,
            "category_counts": {"format": 1},
            "scored_at": datetime.now(timezone.utc).isoformat(),
            "wall_s": 0.0,
            "error": "empty response — skipped critic call",
        }

    if use_cache:
        ck = _cache_key(query, response, tool_calls)
        cached = _cache_lookup(ck)
        if cached is not None:
            cached["candidate_id"] = rec.get("candidate_id")
            return cached

    prompt = make_critic_prompt(query, tool_calls, response)
    start = time.time()
    res = run_claude(
        prompt=prompt,
        system_prompt=CRITIC_SYSTEM,
        model=CRITIC_MODEL,
        use_tools=False,
        timeout_s=CRITIC_TIMEOUT_S,
    )
    wall = round(time.time() - start, 2)
    if not res.get("success"):
        return {
            "candidate_id": rec.get("candidate_id"),
            "verdict": "needs_revision",
            "issues": [],
            "high_count": 0, "medium_count": 0, "low_count": 0,
            "category_counts": {},
            "scored_at": datetime.now(timezone.utc).isoformat(),
            "wall_s": wall,
            "error": f"critic call failed: {res.get('error', '')[:200]}",
        }
    parsed = parse_critic(res["final_text"])
    issues = parsed.get("issues", [])
    sev = Counter(i.get("severity", "?") for i in issues)
    cats = Counter(i.get("category", "?") for i in issues)
    scored = {
        "candidate_id": rec.get("candidate_id"),
        "verdict": parsed.get("verdict", "needs_revision"),
        "issues": issues,
        "high_count": sev.get("high", 0),
        "medium_count": sev.get("medium", 0),
        "low_count": sev.get("low", 0),
        "category_counts": dict(cats),
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "wall_s": wall,
    }
    if use_cache:
        # ck is bound in the cache-lookup branch above; recompute defensively
        # in case the lookup was skipped.
        try:
            _cache_store(_cache_key(query, response, tool_calls), scored)
        except Exception:
            pass
    return scored


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("answers", help="JSONL with model answers (one record per line)")
    p.add_argument("--out", help="Output path (default: <answers>.critic.jsonl)")
    p.add_argument("--limit", type=int, help="Score at most N records")
    p.add_argument("--single", help="Score only this candidate_id")
    p.add_argument("--resume", action="store_true", help="Skip already-scored candidate_ids in output")
    p.add_argument("--no-cache", action="store_true", help="Bypass critic_cache lookups (force re-score)")
    args = p.parse_args()

    in_path = Path(args.answers)
    if not in_path.exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        return 1
    out_path = Path(args.out) if args.out else in_path.with_suffix(".critic.jsonl")

    records = [json.loads(l) for l in in_path.read_text().splitlines() if l.strip()]
    if args.single:
        records = [r for r in records if r.get("candidate_id") == args.single]
    if args.limit:
        records = records[: args.limit]

    if args.resume and out_path.exists():
        already = {json.loads(l)["candidate_id"]
                   for l in out_path.read_text().splitlines() if l.strip()}
        records = [r for r in records if r.get("candidate_id") not in already]
        print(f"Resume mode: {len(records)} remaining (skipped {len(already)})", file=sys.stderr)
    elif not args.resume:
        out_path.unlink(missing_ok=True)

    print(f"Critic-scoring {len(records)} records → {out_path.name}", file=sys.stderr)
    t0 = time.time()
    pass_n = 0
    high_total = 0
    cat_total: Counter = Counter()
    cache_hits = 0
    with out_path.open("a") as f:
        for i, rec in enumerate(records, start=1):
            print(f"[{i}/{len(records)}] {rec.get('candidate_id', '?')}", flush=True, file=sys.stderr)
            scored = score_one(rec, use_cache=not args.no_cache)
            f.write(json.dumps(scored, ensure_ascii=False) + "\n")
            f.flush()
            if scored["verdict"] == "pass":
                pass_n += 1
            high_total += scored["high_count"]
            for c, n in scored["category_counts"].items():
                cat_total[c] += n
            cache_marker = " [cached]" if scored.get("_cached") else ""
            if scored.get("_cached"):
                cache_hits += 1
            print(f"   verdict={scored['verdict']:16s} h={scored['high_count']} m={scored['medium_count']} l={scored['low_count']} wall={scored['wall_s']}s{cache_marker}", file=sys.stderr)

    print(f"\nDone in {time.time()-t0:.0f}s ({cache_hits}/{len(records)} cache hits)", file=sys.stderr)
    print(f"pass: {pass_n}/{len(records)}  high-severity total: {high_total}", file=sys.stderr)
    print("by category:", file=sys.stderr)
    for c, n in cat_total.most_common():
        print(f"  {c}: {n}", file=sys.stderr)
    print(f"output: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
