#!/usr/bin/env python3
"""Headless end-to-end test harness for the /chat AI agent.

Why this exists: the real chat pipeline can only be exercised through the
running production web process — it is the process that holds the model
credentials and the MCP tool config, and a sandboxed shell cannot reach them.
So we drive the production process's own HTTP API instead of trying to run the
model ourselves.

Auth: requests authenticate as admin with the `x-admin-key` header (a
constant-time check against ADMIN_KEY), so the harness is not subject to the
per-user login and rate limits. ADMIN_KEY comes from the environment, falling
back to web/.env.local (the same value the web process loads).

Usage:
    python3 selftest_chat.py "距离 clapham common 车站步行 5 分钟以内的三房，90万以内"
    python3 selftest_chat.py --lang en "3-bed near Clapham Common, 5 min walk, under 900k"

(The Chinese example is the same request as the English one: "3-bed within a
5-minute walk of Clapham Common station, under 900k".)

Prints the model's tool calls (name + args), the final answer, and any listing
IDs it returned — enough to then verify results against the DB.
"""
import argparse
import json
import os
import re
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

# Root of the production checkout (holds web/.env.local and data/).
REPO = Path(os.environ.get("WTL_ROOT", Path(__file__).resolve().parents[3]))
ENV_LOCAL = REPO / "web" / ".env.local"
DB_PATH = REPO / "data" / "evaluations.db"
BASE = "http://localhost:3000"


def read_admin_key() -> str:
    if os.environ.get("ADMIN_KEY"):
        return os.environ["ADMIN_KEY"]
    for line in ENV_LOCAL.read_text().splitlines():
        if line.startswith("ADMIN_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("ADMIN_KEY not found (set env or web/.env.local)")


def run_turn(prompt: str, lang: str, timeout: int = 240) -> str:
    """POST one user turn, drain the SSE stream, return the session_id."""
    admin_key = read_admin_key()
    # Unique, identifiable session so we can read this exact turn back from the DB
    # and never collide with real user traffic.
    session_id = f"selftest-{int(time.time())}"
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "language": lang,
        "sessionId": session_id,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/api/chat",
        data=body,
        headers={"content-type": "application/json", "x-admin-key": admin_key},
        method="POST",
    )
    print(f"→ POST /api/chat  session={session_id}  lang={lang}", file=sys.stderr)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        # Drain the stream so the request runs to completion (and persists).
        for _ in resp:
            pass
    print(f"← stream closed in {time.time() - t0:.1f}s", file=sys.stderr)
    return session_id


def read_back(session_id: str, wait: int = 20) -> dict:
    """Persistence happens when the chat turn finishes, which may land just
    after the stream closes — poll briefly for the row keyed by our session_id."""
    deadline = time.time() + wait
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        while time.time() < deadline:
            row = con.execute(
                "SELECT user_text, assistant_text, tool_calls_json, status, error "
                "FROM chat_messages WHERE session_id=? ORDER BY id DESC LIMIT 1",
                (session_id,)).fetchone()
            if row:
                return {"user_text": row[0], "assistant_text": row[1],
                        "tool_calls": json.loads(row[2] or "[]"),
                        "status": row[3], "error": row[4]}
            time.sleep(1)
    finally:
        con.close()
    sys.exit(f"No chat_messages row for session {session_id} after {wait}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--lang", default="zh", choices=["zh", "en"])
    args = ap.parse_args()

    sid = run_turn(args.prompt, args.lang)
    res = read_back(sid)

    print(f"\n=== status: {res['status']}" + (f"  error: {res['error']}" if res["error"] else ""))
    print("\n=== TOOL CALLS ===")
    for tc in res["tool_calls"]:
        print(f"- {tc.get('name')}  {json.dumps(tc.get('args', {}), ensure_ascii=False)}")
    print("\n=== ASSISTANT ANSWER ===")
    print(res["assistant_text"])
    ids = sorted(set(re.findall(r"/properties/(\d+)", res["assistant_text"] or "")))
    print(f"\n=== listing IDs in answer: {ids}")


if __name__ == "__main__":
    main()
