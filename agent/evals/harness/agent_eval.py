#!/usr/bin/env python3
"""Arden chat-agent end-to-end eval harness.

Drives the REAL production /api/chat endpoint (localhost:3000) with the
x-admin-key bypass header, so every run exercises the exact production path:
the live route, buildSystemPrompt(), the MCP tool config (chat-skills/mcp.json)
and the production model, Claude (Sonnet). Nothing is re-implemented here, so
the harness can never drift from what real users hit.

For each golden case it parses the SSE stream, captures every tool_call (name +
args) and the final answer text, then runs lightweight structural assertions
(no_error, min answer length, required/forbidden tools, per-arg checks). LLM
output is non-deterministic, so assertions are deliberately structural — the
full tool-call trace is always printed for human review.

Usage:
    python3 agent_eval.py                 # run the built-in suite
    python3 agent_eval.py --cases x.json  # run a custom suite
    python3 agent_eval.py --grep farr     # run only cases whose name matches
    python3 agent_eval.py --json out.json # also write machine-readable results

Exit code is non-zero if any case fails a hard assertion (CI-friendly).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

BASE_URL = os.environ.get("CHAT_EVAL_URL", "http://localhost:3000")
# Root of the production checkout (holds web/.env.local). Public-repo default
# is the repo root; set WTL_ROOT when running against a real deployment.
REPO = Path(os.environ.get("WTL_ROOT", Path(__file__).resolve().parents[3]))


def _admin_key() -> str:
    if os.environ.get("ADMIN_KEY"):
        return os.environ["ADMIN_KEY"]
    for fn in (".env.local", ".env", ".env.production"):
        p = REPO / "web" / fn
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if line.startswith("ADMIN_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("ADMIN_KEY not found (set env or web/.env.local)")


# --- Built-in golden suite ------------------------------------------------
# Structural assertions only. `tools_any` = at least one must fire;
# `tools_all` = every one must fire; `tools_none` = none may fire;
# `arg_checks` = [{tool, arg, equals|not_equals|in|gte|lte}].
DEFAULT_CASES = [
    {
        "name": "farringdon-houses",
        "query": ("Find me a 3-bed freehold house under 900k within 30 minutes "
                  "commute of Farringdon, with decent schools."),
        "lang": "en",
        "assert": {
            "no_error": True,
            "min_answer_chars": 200,
            "tools_any": ["search_properties", "search_floorplans",
                          "screen_by_commute", "screen_areas"],
            # The fix: when the agent filters for houses it should pass the
            # umbrella word, NOT a specific stored type it guesses.
            "arg_checks": [
                {"tool": "search_properties", "arg": "property_type",
                 "in": ["house", "houses", None]},
            ],
        },
    },
    {
        "name": "postcode-scores",
        "query": "How good is SW11 to live in? Give me the scores.",
        "lang": "en",
        "assert": {
            "no_error": True,
            "min_answer_chars": 150,
            "tools_any": ["get_postcode_scores", "get_area_overview"],
        },
    },
    {
        "name": "commute-screen",
        "query": ("Which London areas are within 25 minutes of Canary Wharf and "
                  "have a median price under 600k?"),
        "lang": "en",
        "assert": {
            "no_error": True,
            "min_answer_chars": 150,
            "tools_any": ["screen_by_commute", "screen_areas", "get_commute"],
        },
    },
]


def run_case(case: dict, admin_key: str, timeout: int = 240) -> dict:
    """POST one query, parse the SSE stream, return captured trace."""
    body = {
        "messages": [{"role": "user", "content": case["query"]}],
        "language": case.get("lang", "en"),
        "stream": True,
    }
    # Use curl for robust SSE streaming (no extra Python deps).
    cmd = [
        "curl", "-sS", "-N", "-X", "POST", f"{BASE_URL}/api/chat",
        "-H", "Content-Type: application/json",
        "-H", f"x-admin-key: {admin_key}",
        "--max-time", str(timeout),
        "-d", json.dumps(body),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    tool_calls, chunks, errors, done = [], [], [], False
    for line in proc.stdout.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload:
            continue
        try:
            evt = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if "tool_call" in evt:
            tc = evt["tool_call"]
            tool_calls.append({"name": tc.get("name"), "args": tc.get("args", {})})
        elif "chunk" in evt:
            chunks.append(evt["chunk"])
        elif "error" in evt:
            errors.append(evt["error"])
        elif evt.get("done"):
            done = True

    return {
        "answer": "".join(chunks),
        "tool_calls": tool_calls,
        "errors": errors,
        "done": done,
        "elapsed_s": round(elapsed, 1),
        "curl_rc": proc.returncode,
        "stderr": proc.stderr.strip()[:300],
    }


def check(case: dict, res: dict) -> list:
    """Return a list of failure strings ([] == pass)."""
    a = case.get("assert", {})
    fails = []
    used = [tc["name"] for tc in res["tool_calls"]]

    if a.get("no_error") and res["errors"]:
        fails.append(f"error event(s): {res['errors']}")
    if a.get("no_error") and not res["done"]:
        fails.append("stream did not reach done=true")
    if res["curl_rc"] != 0:
        fails.append(f"curl rc={res['curl_rc']} ({res['stderr']})")

    n = len(res["answer"])
    if n < a.get("min_answer_chars", 0):
        fails.append(f"answer too short ({n} < {a['min_answer_chars']})")

    for t in a.get("tools_all", []):
        if t not in used:
            fails.append(f"required tool not used: {t}")
    if a.get("tools_any") and not any(t in used for t in a["tools_any"]):
        fails.append(f"none of expected tools used: {a['tools_any']} (got {used or '∅'})")
    for t in a.get("tools_none", []):
        if t in used:
            fails.append(f"forbidden tool used: {t}")

    for chk in a.get("arg_checks", []):
        for tc in res["tool_calls"]:
            if tc["name"] != chk["tool"]:
                continue
            v = tc["args"].get(chk["arg"])
            if "equals" in chk and v != chk["equals"]:
                fails.append(f"{chk['tool']}.{chk['arg']}={v!r} != {chk['equals']!r}")
            if "not_equals" in chk and v == chk["not_equals"]:
                fails.append(f"{chk['tool']}.{chk['arg']}={v!r} should != {chk['not_equals']!r}")
            if "in" in chk:
                vv = v.lower() if isinstance(v, str) else v
                allowed = [x.lower() if isinstance(x, str) else x for x in chk["in"]]
                if vv not in allowed:
                    fails.append(f"{chk['tool']}.{chk['arg']}={v!r} not in {chk['in']}")
            if "gte" in chk and (v is None or v < chk["gte"]):
                fails.append(f"{chk['tool']}.{chk['arg']}={v!r} not >= {chk['gte']}")
            if "lte" in chk and (v is None or v > chk["lte"]):
                fails.append(f"{chk['tool']}.{chk['arg']}={v!r} not <= {chk['lte']}")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", help="JSON file of cases (default: built-in suite)")
    ap.add_argument("--grep", help="only run cases whose name matches this regex")
    ap.add_argument("--json", help="write machine-readable results here")
    ap.add_argument("--timeout", type=int, default=240)
    args = ap.parse_args()

    cases = json.loads(Path(args.cases).read_text()) if args.cases else DEFAULT_CASES
    if args.grep:
        rx = re.compile(args.grep, re.I)
        cases = [c for c in cases if rx.search(c["name"])]
    if not cases:
        raise SystemExit("no cases to run")

    admin_key = _admin_key()
    results, npass = [], 0
    for c in cases:
        print(f"\n=== {c['name']} ===")
        print(f"  Q: {c['query']}")
        res = run_case(c, admin_key, args.timeout)
        fails = check(c, res)
        ok = not fails
        npass += ok
        results.append({"name": c["name"], "ok": ok, "fails": fails, **res})

        for tc in res["tool_calls"]:
            arg_s = ", ".join(f"{k}={v}" for k, v in tc["args"].items())
            print(f"  → {tc['name']}({arg_s})")
        print(f"  {len(res['answer'])} chars, {len(res['tool_calls'])} tool calls, "
              f"{res['elapsed_s']}s")
        snippet = res["answer"][:280].replace("\n", " ")
        print(f"  A: {snippet}{'…' if len(res['answer']) > 280 else ''}")
        print(f"  {'PASS' if ok else 'FAIL'}" + ("" if ok else f": {'; '.join(fails)}"))

    print(f"\n{'='*50}\n{npass}/{len(cases)} passed")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"results → {args.json}")
    sys.exit(0 if npass == len(cases) else 1)


if __name__ == "__main__":
    main()
