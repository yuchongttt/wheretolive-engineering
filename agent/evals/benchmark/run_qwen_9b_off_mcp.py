#!/usr/bin/env python3
"""Re-run the 50-query baseline on Qwen3.5:9b thinking-OFF *with MCP tools*.

This is the apples-to-apples Phase 2 baseline:
  - Same 50 queries as gold v2 / qwen_baseline_full_9b_off.jsonl
  - Same author system prompt as the gold pipeline (build_gold_answers.AUTHOR_SYSTEM)
  - Same 2 MCP tools (search_properties, get_postcode_scores) — but executed
    inline in Python via direct SQLite reads (matching chat-skills/*.py output)
  - Only difference vs gold: model is Qwen3.5:9b via Ollama /api/chat tool-call loop
    instead of Claude via the agent CLI

Output: data/benchmark/qwen_baseline_full_9b_off_mcp.jsonl

The previous baseline (qwen_baseline_full_9b_off.jsonl) had no tools — answers
were pure model knowledge. This one tests "with the same tool surface as gold,
how close can Qwen get to Sonnet quality?".
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request as urlreq

# Reuse gold pipeline's AUTHOR_SYSTEM + prior-turn injection logic
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_gold_answers import AUTHOR_SYSTEM, make_author_prompt  # noqa: E402

# Root of the production checkout (data/benchmark/ + data/evaluations.db).
REPO_ROOT = Path(os.environ.get("WTL_ROOT", Path(__file__).resolve().parents[3]))
DATA_DIR = REPO_ROOT / "data" / "benchmark"
DB_PATH = REPO_ROOT / "data" / "evaluations.db"
QUERIES_PATH = DATA_DIR / "queries_filtered.jsonl"
REVIEW_PATH = DATA_DIR / "queries_review.json"

OLLAMA_URL = "http://localhost:11434/api/chat"  # NOT /api/generate — chat supports tools
MODEL = "qwen3.5:9b"
# Batch 2 changes vs batch 1:
#   - num_predict 1024 → 2048 below: avoid mid-answer truncation
#   - temperature 0.7 → 0.3 below: less "creative" = less invention
#   - MAX_TOOL_ROUNDS reverted to 5 — capping at 1 (attempted in
#     batch 2a) killed 26/50 answers because Qwen typically does
#     search → get_scores → answer (3 rounds). Hypothesis "fewer rounds
#     = less context bloat" was wrong: rounds are functionally needed.
MAX_TOOL_ROUNDS = 5

# ──────────────────────────────────────────────────────────────────────
# Qwen-specific hard rules (prepended to AUTHOR_SYSTEM for the Qwen path
# only; gold/Sonnet pipeline keeps the original prompt unchanged).
# These target the 4 failure patterns from the qwen-9b-mcp-50query report.
# ──────────────────────────────────────────────────────────────────────

# Few-shot demo block used in batch 3 (dropped again in batch 4 — see
# run_with_tools). It is a model-facing prompt written in Chinese because most
# benchmark queries are Chinese, so it is kept verbatim. English gist: four
# demos — (1) search_properties → cite real listings, copying price / sqft / id
# verbatim; (2) get_postcode_scores → cite the real scores; (3) the tool returns
# "not yet evaluated" → say so, never invent a score; (4) a general-knowledge
# question (leasehold) → no tool, hedge and defer to a solicitor.
# Public copy: the listing ids in demo 1 are redacted placeholders.
QWEN_FEW_SHOT = """# 4 个示范（请严格按这种风格答）

## 示范 1：用 search_properties 找房 → cite 真实 listing

**用户**：帮我找 N1 区 1-2 居、£500k 以内的房源
**调用**：search_properties(bedrooms_min=1, bedrooms_max=2, price_max=500000, postcode_prefix="N1")
**工具返回**：
  Found 11 listings (limit 25):
  - id <listing-id-1> · Example Road, London, N1 · £450,000 · 1 bed · Flat, 400 sq. ft.
  - id <listing-id-2> · Sample Street, N1 · £450,000 · 1 bed · Apartment, 467 sq. ft.
  ...
**正确回答**：
N1 区符合条件的房源共 11 套 [tool=search_properties,n=11]，挑 2 套代表：
- **Example Road, London, N1 · £450,000 · 1 bed · 400 sqft** (id <listing-id-1>)
- **Sample Street, N1 · £450,000 · 1 bed · 467 sqft** (id <listing-id-2>)
注意：价格、平米、id 都是 tool 原文复述，**禁止改写**。

## 示范 2：用 get_postcode_scores 查邮编 → cite 真实分数

**用户**：E14 9NA 这个邮编住起来怎么样？
**调用**：get_postcode_scores(postcode="E14 9NA")
**工具返回**：
  Scores for E14 9NA:
  - Price: 52.5/100 (cached 2026-04-25)
  - Schools: 40.8/100 (cached 2026-04-25)
  - Active listings: 4 on market, avg £587,500
**正确回答**：
E14 9NA 评分 [postcode=E14 9NA, Price=52.5, Schools=40.8]：Price 中等偏下、Schools 一般。在售房源 4 套均价 £587,500。

## 示范 3：工具返"not yet evaluated" → 必须照实说

**用户**：SW16 怎么样？
**调用**：get_postcode_scores(postcode="SW16")
**工具返回**：
  Postcode SW16: not yet evaluated. ... Do NOT make up scores.
**正确回答**：
SW16 我们的评分数据库**暂未评估**——没有分数可引用。基于 UK 一般知识，SW16 是 Streatham 一带，南伦敦 Zone 3，公共交通有 Streatham 国铁站，但具体宜居度需要你触发 /evaluate?postcode=SW16+0AA 取数据。**禁止编出 Transport=70 这类假分数**。

## 示范 4：通识题（如法律 / 流程）→ 不调工具，直接 hedge

**用户**：买 leasehold 注意什么？
**调用**：（无）
**正确回答**：
Leasehold 关键风险：
- 剩余租期：少于 80 年触发 marriage value，延租成本陡增
- Ground rent：2022 年新法把新合同地租设为 peppercorn (£0)；老合同可能有"翻倍条款"
- Service charge：通常 £1,500-£5,000/年
具体条款**必须让 conveyancing solicitor 逐条核对**，不引用具体法条编号。

"""

# Hard rules used both with and without the few-shot demo block. Model-facing
# Chinese prompt, kept verbatim. English gist, highest priority:
#   1. A tool result is a real signal, not reference material — if it says "not
#      yet evaluated" / "No matches found" / "Missing" / "Error", say "no data"
#      and never backfill a score / range / ranking from general knowledge.
#   2. Never cite a tool you did not call this turn — [tool=...] /
#      [postcode=...] tags are only allowed for tools actually called; they are
#      not decoration.
#   3. Copy key fields from tool output verbatim — price, sqft, beds, postcode,
#      id: no paraphrasing or rounding.
#   4. Without calling search_properties, give no specific listings — area-level
#      advice is fine, fabricated listings are not.
QWEN_HARD_RULES = """---

# 最高优先级规则（违反任何一条都是严重错误）

1. **工具结果是真实信号，不是参考资料** —— 如果某次 tool 调用返回的结果包含 "not yet evaluated"、"No matches found"、"Missing"、"Error" 等"无数据"标记，你**只能照实说"暂无该数据"或"工具未返回结果"**，**绝不允许**把它当成不存在然后用通识知识补一个分数 / 区间 / 排序。

2. **没有调用过的 tool，禁止 cite** —— `[tool=search_properties,n=N]`、`[postcode=X,Transport=Y]`、`[get_postcode_scores,...]` 这类引用标签**只能在你本轮真的 call 了对应 tool 时才使用**。如果你没调 tool 就不要写任何这种格式的标签——绝不能把 cite 当成"专业回答的格式装饰"。

3. **复述工具数据时逐字保留关键字段** —— 价格、平米、卧室数、邮编、ID 等具体字段**直接 copy 自 tool 结果，禁止 paraphrase / rounding / 改写**。例如 tool 说 "£450,000 · 1 bed · 400 sq. ft."，答案里就要写完全一致的数字，不能写成 "£425k 1bed 507sqft"。

4. **不调工具时不能给具体 listing** —— 没有调 search_properties 就不要给具体地址、价格、平米这类需要 listing 数据才能确定的内容。可以给区域级建议（"东伦敦 Hackney 通常 £x-y"），但不能伪造具体房源。

---

"""

# OpenAI-style tool schemas — must match chat-skills/*.py and lib/chat-tools.ts
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_properties",
            "description": (
                "Search active UK property listings (rm_sales_overview). "
                "Filter by bedrooms, asking price, postcode prefix, property type. "
                "Returns up to 25 listings sorted by most recently updated. ALWAYS "
                "use this instead of writing SQL — it's the deterministic path."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bedrooms_min": {"type": "integer"},
                    "bedrooms_max": {"type": "integer"},
                    "price_min": {"type": "integer"},
                    "price_max": {"type": "integer"},
                    "postcode_prefix": {"type": "string"},
                    "property_type": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_postcode_scores",
            "description": (
                "Fetch the 5-dimension livability score for a UK postcode "
                "(Transport, Community, Environment, Price, Schools). Returns "
                "each dimension's raw score (0-100) plus key underlying metrics."
            ),
            "parameters": {
                "type": "object",
                "properties": {"postcode": {"type": "string"}},
                "required": ["postcode"],
            },
        },
    },
]


# ──────────────────────────────────────────────────────────────────────
# Tool implementations (same SQL as chat-skills/*.py — keep in sync)
# ──────────────────────────────────────────────────────────────────────


def search_properties_impl(args: dict) -> str:
    limit = max(1, min(int(args.get("limit", 10)), 25))
    where = ["delisted_date IS NULL"]
    params: list[Any] = []
    if args.get("bedrooms_min") is not None:
        where.append("bedrooms >= ?"); params.append(int(args["bedrooms_min"]))
    if args.get("bedrooms_max") is not None:
        where.append("bedrooms <= ?"); params.append(int(args["bedrooms_max"]))
    if args.get("price_min") is not None:
        where.append("asking_price >= ?"); params.append(int(args["price_min"]))
    if args.get("price_max") is not None:
        where.append("asking_price <= ?"); params.append(int(args["price_max"]))
    pcp = (args.get("postcode_prefix") or "").strip().upper()
    if pcp:
        where.append("(postcode = ? OR postcode LIKE ?)")
        params.extend([pcp, pcp + " %"])
    ptype = (args.get("property_type") or "").strip()
    if ptype:
        where.append("LOWER(property_type) LIKE ?")
        params.append("%" + ptype.lower() + "%")
    sql = (
        "SELECT id, postcode, address, asking_price, bedrooms, bathrooms, "
        "property_type, sq_ft, detail_url, date_listed FROM rm_sales_overview "
        f"WHERE {' AND '.join(where)} ORDER BY COALESCE(updated_at, created_at) DESC LIMIT ?"
    )
    params.append(limit)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    if not rows:
        return "No matches found."
    out = [f"Found {len(rows)} listings (limit {limit}):"]
    for r in rows:
        sqft = f", {r['sq_ft']} sqft" if r["sq_ft"] else ""
        out.append(
            f"- id {r['id']} · {r['address'] or r['postcode']} · "
            f"£{r['asking_price']:,} · {r['bedrooms']} bed · "
            f"{r['property_type'] or 'n/a'}{sqft} · {r['detail_url']}"
        )
    return "\n".join(out)


DIMS = [
    ("dim_commute", "Transport / Commute"),
    ("dim_transit", "Transport / Transit"),
    ("dim_safety", "Community / Safety"),
    ("dim_demographics", "Community / Demographics"),
    ("dim_price_analysis", "Price"),
    ("dim_schools", "Schools"),
]


def normalise_postcode(pc: str) -> str:
    cleaned = "".join(pc.split()).upper()
    if len(cleaned) < 5:
        return cleaned
    return cleaned[:-3] + " " + cleaned[-3:]


def get_postcode_scores_impl(args: dict) -> str:
    raw = (args.get("postcode") or "").strip()
    if not raw:
        return "Missing postcode parameter."
    pc = normalise_postcode(raw)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        dims: dict[str, Any] = {}
        for table, label in DIMS:
            row = conn.execute(
                f"SELECT score, data_json, created_at FROM {table} "
                f"WHERE postcode = ? ORDER BY created_at DESC LIMIT 1",
                (pc,),
            ).fetchone()
            if row:
                try:
                    data = json.loads(row["data_json"]) if row["data_json"] else {}
                except json.JSONDecodeError:
                    data = {}
                trimmed = {k: v for k, v in data.items() if k not in ("raw", "intermediate", "debug")}
                dims[label] = {
                    "score": round(row["score"], 1) if row["score"] is not None else None,
                    "as_of": row["created_at"],
                    "details": trimmed,
                }
        listings = conn.execute(
            "SELECT COUNT(*) AS n, AVG(asking_price) AS avg_p FROM rm_sales_overview "
            "WHERE postcode = ? AND delisted_date IS NULL",
            (pc,),
        ).fetchone()
    finally:
        conn.close()
    if not dims:
        return (
            f"Postcode {pc}: not yet evaluated. The user can trigger a first-time "
            f"evaluation by visiting /evaluate?postcode={pc.replace(' ', '+')} "
            f"on wheretolive.xyz. Do NOT make up scores."
        )
    out = [f"Scores for {pc}:"]
    for _, label in DIMS:
        d = dims.get(label)
        if d and d["score"] is not None:
            out.append(f"- {label}: {d['score']}/100 (cached {d['as_of'][:10]})")
    if listings and listings["n"] > 0:
        avg = f", avg £{int(listings['avg_p']):,}" if listings["avg_p"] else ""
        out.append(f"- Active listings: {listings['n']} on market{avg}")
    out.append("")
    out.append("--- structured ---")
    structured = {"postcode": pc, "dimensions": dims}
    if listings and listings["n"] > 0:
        structured["active_listings"] = {
            "count": listings["n"],
            "avg_price": int(listings["avg_p"]) if listings["avg_p"] else None,
        }
    out.append(json.dumps(structured, indent=2, ensure_ascii=False))
    return "\n".join(out)


def execute_tool(name: str, args: dict) -> str:
    try:
        if name == "search_properties":
            return search_properties_impl(args)
        if name == "get_postcode_scores":
            return get_postcode_scores_impl(args)
        return f"Error: unknown tool {name}"
    except Exception as e:
        return f"Error executing {name}: {e}"


def trim_tool_result_for_model(name: str, full: str) -> str:
    """Batch 4 (B4): give model a more focused view of tool output.

    Hypothesis: long tool results bloat context and degrade Qwen's
    field-fidelity (failure pattern ②). Trim:
      - search_properties: keep first 8 listings instead of up to 25
      - get_postcode_scores: keep score lines, drop verbose JSON tail
    Returns the same text format but shorter. The full result is still
    stored in tool_calls for the critic / record.
    """
    if not full:
        return full
    if name == "search_properties":
        # Header + first 8 bullet lines is enough for grounding most queries.
        lines = full.split("\n")
        header = lines[0] if lines else ""
        bullets = [l for l in lines[1:] if l.strip()]
        kept = bullets[:8]
        if len(bullets) > 8:
            kept.append(f"... ({len(bullets) - 8} more listings omitted; ask if you need them)")
        return "\n".join([header] + kept)
    if name == "get_postcode_scores":
        # Keep everything before the "--- structured ---" trailer.
        marker = "--- structured ---"
        if marker in full:
            return full.split(marker, 1)[0].rstrip()
    return full


# ──────────────────────────────────────────────────────────────────────
# Ollama /api/chat one round + tool loop
# ──────────────────────────────────────────────────────────────────────


def chat_round(messages: list[dict], with_tools: bool, timeout_s: int = 600) -> dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.3, "num_predict": 2048},
    }
    if with_tools:
        payload["tools"] = TOOLS
    body = json.dumps(payload).encode("utf-8")
    req = urlreq.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    start = time.time()
    with urlreq.urlopen(req, timeout=timeout_s) as resp:
        d = json.loads(resp.read().decode("utf-8"))
    msg = d.get("message") or {}
    return {
        "content": msg.get("content", "") or "",
        "tool_calls": msg.get("tool_calls") or [],
        "eval_count": d.get("eval_count", 0),
        "eval_ns": d.get("eval_duration", 0),
        "wall_s": round(time.time() - start, 2),
    }


def run_with_tools(query_text: str, candidate_id: str, intent_subtype: str | None) -> dict:
    user_prompt = make_author_prompt(query_text, candidate_id, intent_subtype)
    # Qwen-specific hard rules go FIRST (top of prompt, highest attention).
    # Batch 4: drop few-shot demos (batch 3 showed they encouraged
    # MORE cite invention not less). Keep just the hard rules.
    system_content = QWEN_HARD_RULES + AUTHOR_SYSTEM
    messages: list[dict] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_prompt},
    ]
    all_tool_calls: list[dict] = []
    total_eval = 0
    rounds = 0
    answer = ""
    overall_start = time.time()
    error: str | None = None

    for r_idx in range(MAX_TOOL_ROUNDS + 1):
        try:
            r = chat_round(messages, with_tools=True)
        except Exception as e:
            error = str(e)[:200]
            break
        rounds += 1
        total_eval += r["eval_count"]
        answer = r["content"]
        if not r["tool_calls"]:
            break
        if r_idx == MAX_TOOL_ROUNDS:
            error = "max_tool_rounds_exceeded"
            break
        # Append assistant tool_call message + tool results
        messages.append({
            "role": "assistant",
            "content": r["content"],
            "tool_calls": r["tool_calls"],
        })
        for tc in r["tool_calls"]:
            fn = tc.get("function", {})
            name = fn.get("name", "?")
            args = fn.get("arguments", {}) or {}
            result_full = execute_tool(name, args)
            # Batch 4 (B4): trim what model sees, but record the full result
            # for the critic so it can verify cite fidelity end-to-end.
            result_for_model = trim_tool_result_for_model(name, result_full)
            result = result_full  # alias: record stores the full text
            # Store full result (was truncated to 300 chars in earlier runs;
            # critic-calibration study found that caused ~10-15 false-positive
            # tool_hallucination flags because critic couldn't see past the
            # first 1-2 listings of a 25-result search).
            all_tool_calls.append({"name": name, "args": args, "result": result})
            messages.append({"role": "tool", "content": result_for_model})

    cleaned, removed = _clean_floating_cites(answer, all_tool_calls)
    return {
        "ok": error is None and bool(cleaned),
        "response": cleaned,
        "response_raw": answer if removed else None,  # only kept when cleanup did something
        "cites_stripped": removed,
        "tool_calls": all_tool_calls,
        "tool_rounds": rounds,
        "eval_count": total_eval,
        "wall_s": round(time.time() - overall_start, 2),
        "error": error,
    }


# Cite-format markers Qwen learned to emit. Each match is the full bracketed
# token. We only strip a token if its "tool name" (or postcode for
# get_postcode_scores style cites) doesn't appear in the actual tool calls
# this turn. Keeps legit cites intact while removing fabricated ones.
_CITE_RE = re.compile(r'\[(?:tool=|postcode=)[^\]]+\]')


def _clean_floating_cites(text: str, tool_calls: list[dict]) -> tuple[str, int]:
    """Strip [tool=...] / [postcode=X,...] tokens that don't correspond to
    an actual tool call this turn. Returns (cleaned_text, removed_count).
    """
    if not text:
        return text, 0
    # Build the set of "valid" cites the model could legitimately emit
    called_tool_names = {tc.get("name", "") for tc in tool_calls}
    called_postcodes = set()
    for tc in tool_calls:
        if tc.get("name") == "get_postcode_scores":
            pc = (tc.get("args", {}) or {}).get("postcode", "")
            if pc:
                # Normalise for fuzzy match — e.g. tool args "E14 0GQ" vs
                # cite "E140GQ" both reduce to the same key.
                called_postcodes.add(pc.replace(" ", "").upper())

    removed = 0

    def keep(match: re.Match) -> str:
        nonlocal removed
        token = match.group(0)
        inner = token[1:-1]  # strip [ and ]
        # Two recognised forms:
        # [tool=search_properties,n=N]   → check against called tool names
        # [postcode=N1 9DT,Transport=72] → check against called postcodes
        if inner.startswith("tool="):
            tool_name = inner.split("=", 1)[1].split(",", 1)[0].strip()
            if tool_name not in called_tool_names:
                removed += 1
                return ""
        elif inner.startswith("postcode="):
            pc = inner.split("=", 1)[1].split(",", 1)[0].strip().replace(" ", "").upper()
            if pc not in called_postcodes:
                removed += 1
                return ""
        return token

    cleaned = _CITE_RE.sub(keep, text)
    return cleaned, removed


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def load_queries() -> list[dict]:
    decisions = json.loads(REVIEW_PATH.read_text())
    by_id = {}
    for line in QUERIES_PATH.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            by_id[rec["candidate_id"]] = rec
    out = []
    for cid in sorted(decisions):
        if decisions[cid].get("status") != "kept":
            continue
        rec = by_id.get(cid)
        if not rec:
            continue
        out.append({
            "candidate_id": cid,
            "query": rec["primary_query"],
            "query_class": decisions[cid].get("final_class") or rec.get("filter_class"),
            "intent_subtype": rec.get("intent_subtype"),
        })
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, help="Process at most N queries (debug)")
    p.add_argument("--single", help="Run only this candidate_id (debug)")
    args = p.parse_args()

    queries = load_queries()
    if args.single:
        queries = [q for q in queries if q["candidate_id"] == args.single]
    if args.limit:
        queries = queries[: args.limit]

    out_path = DATA_DIR / "qwen_baseline_full_9b_off_mcp.jsonl"
    print(f"Running {len(queries)} queries through {MODEL} + MCP → {out_path.name}")
    out_path.unlink(missing_ok=True)

    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as f:
        for i, q in enumerate(queries, start=1):
            print(f"[{i}/{len(queries)}] {q['candidate_id']} ({q['query_class']}/{q.get('intent_subtype','-')})", flush=True)
            r = run_with_tools(q["query"], q["candidate_id"], q.get("intent_subtype"))
            rec = {
                "candidate_id": q["candidate_id"],
                "query": q["query"],
                "query_class": q["query_class"],
                "intent_subtype": q.get("intent_subtype"),
                "model": MODEL,
                "set": "full_9b_off_mcp",
                "thinking_enabled": False,
                "ran_at": datetime.now(timezone.utc).isoformat(),
                **r,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            preview = (r.get("response") or "")[:80].replace("\n", " ")
            print(f"   wall={r['wall_s']:.1f}s rounds={r['tool_rounds']} tools={len(r['tool_calls'])} → {preview}…", flush=True)

    print(f"\nDone in {time.time()-t0:.0f}s. Output: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
