#!/usr/bin/env python3
"""Capability-gap mining — pull out the agent's own admissions that "I don't
have this data / tool".

**Why this exists**: on 2026-08-18, reviewing benchmark round R2-9 (market
timing), we found that the gap had already been stated by the agent on
2026-07-11, in language precise enough to start work from. Its reply (in
Chinese) said, in effect:

    "My tools have no field like 'historical listing volume by month', so I am
     answering this part from publicly known seasonal patterns of the UK
     property market, not from specific numbers looked up in my tools."

It had done the requirements analysis, written it into assistant_text, and then
the sentence sat in the table unread. Five weeks later (during which our own
weekly simulation hit it again and waved it through as normal) it was
rediscovered by a hand-run benchmark question.

So the cost of discovering a gap was "one person hand-running one benchmark
question", one gap per round — while the agent produces these diagnoses for
free, continuously. This module brings the cost of collecting them to zero.

**Why the source is chat_messages, not chat_telemetry**: telemetry's
tool_timings_json only started being written on 2026-07-13, and the R2-9 turn
was on 07-11 — mining telemetry would make the founding case itself invisible.
chat_messages.tool_calls_json has 100% coverage for the month. ((session_id,
turn_idx) is unique in neither table, so they cannot be joined.)

The assistant is bilingual (EN/ZH); the admission patterns below match the
Chinese and English phrasings seen in real replies and are kept verbatim.

Usage:
    python3 capability_gaps.py                 # all open gaps
    python3 capability_gaps.py --since 2026-08-01T00:00:00Z
    python3 capability_gaps.py --json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

# Root of the production checkout (holds data/evaluations.db).
REPO = Path(os.environ.get("WTL_ROOT", Path(__file__).resolve().parents[3]))
DB_PATH = REPO / "data" / "evaluations.db"

# My own benchmark runs and contract runs — I already know the answers; they
# are not a discovery channel.
BENCH_SESSION = re.compile(r"^(cmp-|bhv-|selftest-|test-)")
# The weekly user simulation. **Kept on purpose**: it is the second sensor — it
# hit the same seasonal gap on 2026-08-12 (and waved it through). Tagged sim and
# ranked after real users.
SIM_SESSION = re.compile(r"^sim-")

# Phrasings the agent uses to admit a gap. Every one is taken from a real reply,
# not imagined wording. Chinese meanings, in order:
#   no_tool:       "(my) tools don't have", "no field like this", "not looked up
#                  from the tools";
#   no_data:       "I don't have … data / records / fields", "we don't have …
#                  data / records";
#   scope_limit:   "my data covers …", "data coverage … not", "coverage … not";
#   cannot_verify: "unable to verify / check / find", "cannot find … field /
#                  data / record".
GAP_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("no_tool", re.compile(r"(我的)?工具里没有|没有[^。；\n]{0,14}这类字段|不是从工具里查")),
    ("no_data", re.compile(r"我没有[^。；\n]{0,16}(数据|记录|字段)|我们没有[^。；\n]{0,16}(数据|记录)")),
    ("scope_limit", re.compile(r"我的数据覆盖的是|数据覆盖[^。；\n]{0,10}不|覆盖范围[^。；\n]{0,10}不")),
    ("cannot_verify", re.compile(r"无法(查证|核实|查到)|查不到[^。；\n]{0,10}(字段|数据|记录)")),
    ("en_no_tool", re.compile(r"(don't|do not) have (a )?tool|no tool (for|that)|not something (I|we) (can|hold)", re.I)),
]

# Second kind of signal: the user explicitly asked us to search, and this turn
# called no tool at all.
# Measured on 120 real turns, naive "zero tools = suspicious" was far too
# imprecise — most of the 26 zero-tool turns were legitimate: greetings, the
# user adding information ("the service charge is 0"), follow-ups on existing
# results ("give me the link"). So three conditions must hold together: an
# explicit search imperative + status='ok' + zero tools.
# **The status filter is not optional**: the three "announced a search but
# didn't run it" turns from 2026-05 were actually error / timeout (exit 143);
# counting failed turns would turn this detector into a disguised error
# detector.
# (Chinese imperatives: "help me find / search / recommend", "find a few",
# "give me a few", "recommend a few", "see if there are any".)
SEARCH_IMPERATIVE = re.compile(
    r"帮我(找|搜|推荐)|找几套|给我几套|推荐几套|帮我看看有没有|"
    r"find me|show me|recommend .{0,16}(propert|flat|home|house)", re.I)
# With a search imperative but missing constraints, asking back is legitimate:
# a miss is only flagged when the user **gave enough constraints**.
# Requests like "find a similar home" are defined by a referent (an image or a
# link); if the referent was not uploaded, no search can run and asking back is
# correct. In practice this was the first version's only false positive.
# (Chinese: "similar", "like this", "this picture", "a home like this".)
REFERENT_REQUIRED = re.compile(r"类似|像这|这张图|这样的房子|like this|similar to (this|it)", re.I)
ENOUGH_CONSTRAINTS = re.compile(
    # `\d\s*k\b`, not a bare `k\b`: under re.I the latter took the "ok" in "ok
    # find me…" as a budget (that is how explore-20260923 P25 was counted as
    # "enough constraints given").
    # (Chinese: "budget", "10k GBP" units, "N-bedroom", "East/West/South/North/
    # Central London".)
    r"(预算|budget|£|万镑|万英镑|\d\s*k\b|万)|(\d\s*(居|房|卧|bed))|"
    r"([A-Z]{1,2}\d[A-Z\d]?\b)|(东|西|南|北|中)伦敦|east|west|north|south london", re.I)
# The reply explicitly drew a capability/coverage boundary ("for-sale only",
# "no rental inventory", "outside coverage") — refusing to fabricate is correct
# behaviour, not a skipped search. explore-20260923 P25: there is no search lane
# for rentals, and the model said "my search_properties tool only covers
# for-sale listings … Anything I invented here would be a fabricated listing",
# yet it was scored the top-severity hard, polluting the priority of the daily
# signal. Such turns do not enter search_not_run; the rental-search gap itself
# is tracked separately by a human.
# `only covers` alone was too wide («our data only covers London — what's your
# budget?» is a dodge, not a boundary): the phrase must name what it covers
# (for-sale / sales / listings / buying) or the sentence must name rentals.
# (Chinese: "only covers / includes / has for-sale", "no rentals", "does not
# cover / include rentals".)
_BOUNDARY_OBJECT = r"[^.\n]{0,40}(for[- ]sale|sales|listings|buy(ing)?|purchase)"
DECLARED_BOUNDARY = re.compile(
    r"only covers?\b" + _BOUNDARY_OBJECT + r"|covers? only\b" + _BOUNDARY_OBJECT + r"|"
    r"no inventory of|"
    r"(don'?t|do not|cannot|can'?t) (have|hold|carry|index|pull|search)[^.\n]{0,40}"
    r"(rental|lettings|to-let|rent\b)|outside (our|my) (data )?coverage|"
    r"只(覆盖|收录|有)在售|没有租房|不(覆盖|收录)租", re.I)

# Negated negation: "it's not that I don't have data" is not a gap admission.
# A bare regex matching a negated sentence is a known self-inflicted failure
# (R2-8 lesson 8b); it is blocked at the sentence level here.
# (Chinese: "it is not / rather than / cannot say … (I / we) don't have /
# cannot / can't find".)
NEGATION_GUARD = re.compile(r"(并?不是|并非|而不是|不能说)[^。；\n]{0,6}(我|我们)?(没有|无法|查不到)")

# Real agent output mostly uses **ASCII commas** as separators (often a single
# full stop at the very end), so splitting on full stops would take the whole
# paragraph as evidence. Splitting into clauses makes the evidence land on the
# "what is missing" clause.
_CLAUSE_SPLIT = re.compile(r"[，,。！？；;\n]")
_NORM = re.compile(r"[\s，。、？！,.?!；;：:\"'「」『』（）()]+")


# Evidence clauses often sit in markdown headings or table cells; carrying
# "## " / "|" through verbatim is hard to read.
_MD_NOISE = re.compile(r"^[#*>\-\s|]+|[\s|*]+$")


def _clauses(text: str) -> list[str]:
    out = []
    for raw in _CLAUSE_SPLIT.split(text or ""):
        s = _MD_NOISE.sub("", raw.strip())
        if s:
            out.append(s)
    return out


def classify_session(session_id: str | None, user_id: str | None) -> str | None:
    """real / sim / None (discard)."""
    sid = session_id or ""
    if BENCH_SESSION.match(sid):
        return None
    if SIM_SESSION.match(sid):
        return "sim"
    if (user_id or "") == "admin":
        return None
    return "real"


def gap_key(label: str, question: str) -> str:
    """Same question (ignoring punctuation/whitespace) + same kind of admission
    → the same gap."""
    norm = _NORM.sub("", (question or "").lower())
    return hashlib.sha1(f"{label}|{norm}".encode()).hexdigest()[:12]


def detect(assistant_text: str) -> tuple[str, str] | None:
    """Does the reply admit a gap? Returns (label, evidence clause)."""
    clauses = _clauses(assistant_text)
    for i, clause in enumerate(clauses):
        # The guard looks at "previous clause + this clause": with fine-grained
        # splitting the negation may land in the previous clause ("this is not
        # to say, I have no data"); looking at this clause alone would miss it.
        window = (clauses[i - 1] + "," + clause) if i else clause
        if NEGATION_GUARD.search(window):
            continue
        for label, pat in GAP_PATTERNS:
            if pat.search(clause):
                return label, clause
    return None


def _tool_count(raw: str | None) -> int:
    try:
        val = json.loads(raw or "[]")
        return len(val) if isinstance(val, (list, dict)) else 0
    except Exception:
        return 0


def scan(db_path: Path | str = DB_PATH, since: str | None = None) -> list[dict]:
    """Find gaps, ordered by severity (hard first) and then by time."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    # The status column may not exist (old DB / test skeleton); when it does it
    # must be read — failed turns must not take part in behaviour judgements.
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(chat_messages)")}
    except sqlite3.OperationalError:
        cols = set()
    status_col = ", status" if "status" in cols else ""
    sql = (f"SELECT ts, user_id, session_id, user_text, assistant_text, tool_calls_json"
           f"{status_col}"
           " FROM chat_messages WHERE assistant_text IS NOT NULL AND assistant_text <> ''")
    args: list = []
    if since:
        sql += " AND datetime(ts) > datetime(?)"
        args.append(since)
    try:
        rows = conn.execute(sql + " ORDER BY ts", args).fetchall()
    except sqlite3.OperationalError:
        # Table missing (minimal test skeleton / new DB). Gap mining is a bonus
        # signal and must not take down a whole self-review run.
        rows = []
    finally:
        conn.close()

    out: list[dict] = []
    for r in rows:
        source = classify_session(r["session_id"], r["user_id"])
        if source is None:
            continue
        # Failed turns never take part in behaviour judgements (the status
        # column may be missing; a missing column counts as ok).
        status = r["status"] if status_col else "ok"
        if status not in (None, "", "ok"):
            continue

        n_tools = _tool_count(r["tool_calls_json"])
        question = (r["user_text"] or "").strip()
        hit = detect(r["assistant_text"])
        if not hit and n_tools == 0 and SEARCH_IMPERATIVE.search(question) \
                and ENOUGH_CONSTRAINTS.search(question) \
                and not REFERENT_REQUIRED.search(question) \
                and not DECLARED_BOUNDARY.search(r["assistant_text"] or ""):
            hit = ("search_not_run",
                   "user explicitly asked for listings with enough constraints; "
                   "zero tool calls this turn (reply was only a question / explanation)")
        if not hit:
            continue
        label, evidence = hit
        out.append({
            "key": gap_key(label, question),
            "label": label,
            # hard = answered without touching any data this turn;
            # partial = tools were called but coverage was incomplete
            "severity": "hard" if n_tools == 0 else "partial",
            "source": source,
            "ts": r["ts"],
            "session_id": r["session_id"],
            "question": question,
            "evidence": evidence,
            "tools_used": n_tools,
        })

    rank = {"hard": 0, "partial": 1}
    src = {"real": 0, "sim": 1}
    out.sort(key=lambda g: (rank[g["severity"]], src[g["source"]], g["ts"]), reverse=False)
    return out


def summarise(gaps: list[dict]) -> dict:
    """Compact summary of new gaps (output of a zero-token scan)."""
    by_key: dict[str, dict] = {}
    for g in gaps:
        cur = by_key.setdefault(g["key"], {**g, "hits": 0, "last_ts": g["ts"]})
        cur["hits"] += 1
        cur["last_ts"] = max(cur["last_ts"], g["ts"])
    return {
        "total": len(gaps),
        "distinct": len(by_key),
        "hard": sum(1 for g in by_key.values() if g["severity"] == "hard"),
        "real": sum(1 for g in by_key.values() if g["source"] == "real"),
        "items": [
            {k: v[k] for k in ("key", "severity", "source", "label", "question",
                               "evidence", "hits", "last_ts")}
            for v in list(by_key.values())[:20]
        ],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Mine capability gaps the agent admits to")
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--since", help="ISO timestamp; only look at later turns")
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args(argv)

    gaps = scan(args.db, since=args.since)
    if args.as_json:
        print(json.dumps(summarise(gaps), ensure_ascii=False, indent=2))
        return 0

    if not gaps:
        print("No self-reported gaps" + (f" (since {args.since})" if args.since else ""))
        return 0
    s = summarise(gaps)
    print(f"Self-reported capability gaps: {s['total']} hits / {s['distinct']} distinct"
          f" (hard {s['hard']}, from real users {s['real']})\n")
    for it in s["items"]:
        tag = "[HARD]" if it["severity"] == "hard" else "[partial]"
        who = "real user" if it["source"] == "real" else "weekly simulation"
        print(f"{tag} [{it['label']}] {who} · {it['hits']} hits · last {it['last_ts']}")
        print(f"   Q: {it['question'][:70]}")
        print(f"   admitted: {it['evidence'][:110]}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
