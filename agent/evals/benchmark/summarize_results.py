#!/usr/bin/env python3
"""Summarise a directory of benchmark JSONL files into Markdown (or JSON) tables.

Written for the public write-up (RESULTS.md): the raw run files contain listing
URLs and addresses inside tool results, so they are not published — only the
aggregate numbers this script prints.

It recognises three record shapes by content (not by file name):

  critic   — output of critic_score.py / the gold critic pass:
             {candidate_id, verdict, issues[{category, severity, detail}],
              high_count, medium_count, low_count, scored_at, error?, _cached?}
  answers  — model answer files (Qwen baselines, gold_answers.jsonl):
             {candidate_id, query, response|final_answer, tool_calls?, wall_s?,
              ran_at|generated_at, ok?, status?, critic_history?, ...}
  thinking — thinking-mode A/B files: {candidate_id, model, thinking: bool,
              wall_s, eval_count, response, thinking_text}

Anything else (e.g. the query-set files) is listed as skipped.

Two bookkeeping rules matter for reading the numbers honestly:

* A critic reply that is not valid JSON is recorded by parse_critic() as a
  single low-severity "incomplete" issue whose detail starts with
  "critic output was unparseable", with verdict needs_revision. That is critic
  noise, not a property of the answer, so it is counted separately
  ("parse_failures"), excluded from the issue tables, and excluded from the
  denominator of `pass_rate_judged`.
* An empty model answer is scored by critic_score.py without calling the critic
  ("empty response — skipped critic call", one high "format" issue). That IS a
  model failure and stays in every count.

Usage:
    python3 summarize_results.py DIR              # Markdown to stdout
    python3 summarize_results.py DIR --json       # machine-readable
    python3 summarize_results.py DIR --only-ids-of DIR/some_run.jsonl
        # restrict every file to the candidate_ids present in that file, e.g.
        # to compare the 210-query gold set with 50-query baseline runs
    python3 summarize_results.py DIR --group-by DIR/queries.jsonl:intent_subtype \
        --detail-regex '585,000|788 sqft'
        # also split each critic file by a per-case field looked up in another
        # JSONL (joined on candidate_id), and count tool_hallucination issues
        # whose detail text matches the regex
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

PARSE_FAIL_PREFIX = "critic output was unparseable"
CRITIC_CALL_FAIL_PREFIX = "critic call failed"
SEVERITIES = ("high", "medium", "low")
# Rubric order from build_gold_answers.CRITIC_SYSTEM; unknown categories are appended.
CATEGORIES = ("tool_hallucination", "general_hallucination", "missing_hedge",
              "off_domain_answered", "format", "incomplete")


def load_jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def kind_of(records: list[dict]) -> str:
    if not records:
        return "empty"
    first = records[0]
    if "verdict" in first and "issues" in first:
        return "critic"
    if isinstance(first.get("thinking"), bool) and "thinking_text" in first:
        return "thinking"
    if "response" in first or "final_answer" in first or "error" in first and "query" in first:
        return "answers"
    return "unknown"


def _date_range(values: list[str]) -> str:
    days = sorted(v[:10] for v in values if v)
    if not days:
        return ""
    return days[0] if days[0] == days[-1] else f"{days[0]}..{days[-1]}"


def _is_parse_failure(rec: dict) -> bool:
    return any(str(i.get("detail", "")).startswith(PARSE_FAIL_PREFIX)
               for i in rec.get("issues") or [])


def summarize_critic(records: list[dict], detail_regex: re.Pattern | None = None) -> dict:
    n = len(records)
    verdicts = Counter(r.get("verdict") for r in records)
    parse_fail = [r for r in records if _is_parse_failure(r)]
    call_fail = [r for r in records
                 if str(r.get("error", "")).startswith(CRITIC_CALL_FAIL_PREFIX)]
    empty_answer = [r for r in records if "empty response" in str(r.get("error", ""))]

    sev: Counter = Counter()
    cat: Counter = Counter()
    cases_with_high = 0
    cases_with_tool_halluc = 0
    tool_halluc_matching = 0
    for r in records:
        real_issues = [i for i in (r.get("issues") or [])
                       if not str(i.get("detail", "")).startswith(PARSE_FAIL_PREFIX)]
        for i in real_issues:
            sev[i.get("severity", "?")] += 1
            cat[i.get("category", "?")] += 1
            if (detail_regex is not None and i.get("category") == "tool_hallucination"
                    and detail_regex.search(str(i.get("detail", "")))):
                tool_halluc_matching += 1
        if any(i.get("severity") == "high" for i in real_issues):
            cases_with_high += 1
        if any(i.get("category") == "tool_hallucination" for i in real_issues):
            cases_with_tool_halluc += 1

    judged = n - len(parse_fail) - len(call_fail)
    n_pass = verdicts.get("pass", 0)
    out = {
        "cases": n,
        "scored": _date_range([r.get("scored_at", "") for r in records]),
        "verdicts": dict(verdicts),
        "pass": n_pass,
        "pass_rate": round(n_pass / n, 3) if n else None,
        "parse_failures": len(parse_fail),
        "critic_call_failures": len(call_fail),
        "empty_answers": len(empty_answer),
        "pass_rate_judged": round(n_pass / judged, 3) if judged else None,
        "issues_by_severity": {s: sev.get(s, 0) for s in SEVERITIES},
        "issues_by_category": {c: cat.get(c, 0) for c in
                               list(CATEGORIES) + sorted(set(cat) - set(CATEGORIES))},
        "cases_with_high": cases_with_high,
        "cases_with_tool_hallucination": cases_with_tool_halluc,
        "cache_hits": sum(1 for r in records if r.get("_cached")),
    }
    if detail_regex is not None:
        out["tool_hallucination_matching_regex"] = tool_halluc_matching
    return out


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.mean(xs), 1) if xs else None


def _median(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.median(xs), 1) if xs else None


def summarize_answers(records: list[dict]) -> dict:
    n = len(records)
    has_tools = any("tool_calls" in r for r in records)
    tool_counts = [len(r.get("tool_calls") or []) for r in records]
    responses = [(r.get("response") if "response" in r else r.get("final_answer")) or ""
                 for r in records]
    out = {
        "cases": n,
        "models": sorted({str(r["model"]) for r in records if r.get("model")}),
        "thinking": sorted({str(r["thinking_enabled"]) for r in records
                            if "thinking_enabled" in r}),
        "ran": _date_range([r.get("ran_at") or r.get("generated_at") or "" for r in records]),
        "ok": sum(1 for r in records if r.get("ok")) if any("ok" in r for r in records) else None,
        "errors": sum(1 for r in records if r.get("error")),
        "empty_responses": sum(1 for x in responses if not x.strip()),
        "mean_response_chars": _mean([len(x) for x in responses]),
        "mean_wall_s": _mean([r.get("wall_s") for r in records]),
        "median_wall_s": _median([r.get("wall_s") for r in records]),
        "mean_eval_tokens": _mean([r.get("eval_count") for r in records]),
    }
    if has_tools:
        stored = Counter()
        for r in records:
            for tc in r.get("tool_calls") or []:
                stored["full" if "result" in tc else
                       "preview" if "result_preview" in tc else "none"] += 1
        out.update({
            "mean_tool_calls": _mean(tool_counts),
            "zero_tool_cases": sum(1 for c in tool_counts if c == 0),
            "tool_result_storage": dict(stored),
        })
    if any("cites_stripped" in r for r in records):
        out["cites_stripped"] = sum(r.get("cites_stripped") or 0 for r in records)
    if any("status" in r for r in records):
        out["status"] = dict(Counter(r.get("status") for r in records))
    if any("critic_history" in r for r in records):
        out["critic_rounds"] = dict(sorted(Counter(
            len(r.get("critic_history") or []) for r in records).items()))
    if any("split" in r for r in records):
        out["split"] = dict(Counter(r.get("split") or "(none)" for r in records))
    return out


def summarize_thinking(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault((str(r.get("model")), bool(r.get("thinking"))), []).append(r)
    rows = []
    for (model, thinking), rs in sorted(groups.items()):
        rows.append({
            "model": model, "thinking": thinking, "cases": len(rs),
            "mean_wall_s": _mean([r.get("wall_s") for r in rs]),
            "mean_eval_tokens": _mean([r.get("eval_count") for r in rs]),
            "mean_answer_chars": _mean([len(r.get("response") or "") for r in rs]),
            "empty_answers": sum(1 for r in rs if not (r.get("response") or "").strip()),
            "ran": _date_range([r.get("ran_at", "") for r in rs]),
        })
    return rows


def summarize_dir(directory: Path, only_ids: set | None = None,
                  groups: dict | None = None, detail_regex: re.Pattern | None = None) -> dict:
    """groups: optional {candidate_id: group value}; critic files are then also
    summarised per group (cases without a group value go to "(none)")."""
    files = sorted(p for p in Path(directory).glob("*.jsonl") if p.is_file())
    digests: dict[str, str] = {}
    result = {"critic": {}, "critic_by_group": {}, "answers": {}, "thinking": {},
              "skipped": {}, "duplicates": {}}
    for p in files:
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h in digests:
            result["duplicates"][p.name] = digests[h]
        else:
            digests[h] = p.name
        try:
            recs = load_jsonl(p)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            result["skipped"][p.name] = f"unreadable: {e}"
            continue
        if only_ids is not None:
            recs = [r for r in recs if r.get("candidate_id") in only_ids]
        kind = kind_of(recs)
        if kind == "critic":
            result["critic"][p.name] = summarize_critic(recs, detail_regex)
            if groups is not None:
                by: dict[str, list] = {}
                for r in recs:
                    by.setdefault(str(groups.get(r.get("candidate_id"), "(none)")), []).append(r)
                result["critic_by_group"][p.name] = {
                    g: summarize_critic(rs, detail_regex) for g, rs in sorted(by.items())}
        elif kind == "answers":
            result["answers"][p.name] = summarize_answers(recs)
        elif kind == "thinking":
            result["thinking"][p.name] = summarize_thinking(recs)
        else:
            result["skipped"][p.name] = kind
    return result


def _fmt(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".") if v < 1 else f"{v:g}"
    if isinstance(v, dict):
        return ", ".join(f"{k}: {x}" for k, x in v.items()) or "–"
    if isinstance(v, list):
        return ", ".join(v) or "–"
    return str(v)


def render_markdown(s: dict) -> str:
    lines: list[str] = []
    if s["critic"]:
        lines += ["### Critic verdicts", "",
                  "| file | scored | cases | pass | needs_revision | parse failures | "
                  "pass rate | pass rate (judged) | cases w/ high | cases w/ tool_halluc |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for name, c in s["critic"].items():
            lines.append(
                f"| {name} | {c['scored']} | {c['cases']} | {c['pass']} | "
                f"{c['verdicts'].get('needs_revision', 0)} | {c['parse_failures']} | "
                f"{_fmt(c['pass_rate'])} | {_fmt(c['pass_rate_judged'])} | "
                f"{c['cases_with_high']} | {c['cases_with_tool_hallucination']} |")
        lines += ["", "### Critic issues (parse-failure markers excluded)", ""]
        cats = []
        for c in s["critic"].values():
            for k in c["issues_by_category"]:
                if k not in cats:
                    cats.append(k)
        lines += ["| file | high | medium | low | " + " | ".join(cats) + " |",
                  "|---|---|---|---|" + "---|" * len(cats)]
        for name, c in s["critic"].items():
            sev = c["issues_by_severity"]
            lines.append(f"| {name} | {sev['high']} | {sev['medium']} | {sev['low']} | "
                         + " | ".join(str(c["issues_by_category"].get(k, 0)) for k in cats)
                         + " |")
        lines.append("")
    if s.get("critic_by_group"):
        with_regex = any("tool_hallucination_matching_regex" in c
                         for groups in s["critic_by_group"].values() for c in groups.values())
        lines += ["### Critic verdicts by group", "",
                  "| file | group | cases | pass | parse failures | tool_hallucination issues"
                  + (" | … matching --detail-regex" if with_regex else "") + " |",
                  "|---|---|---|---|---|---|" + ("---|" if with_regex else "")]
        for name, groups in s["critic_by_group"].items():
            for g, c in groups.items():
                lines.append(
                    f"| {name} | {g} | {c['cases']} | {c['pass']} | {c['parse_failures']} | "
                    f"{c['issues_by_category'].get('tool_hallucination', 0)}"
                    + (f" | {c.get('tool_hallucination_matching_regex', 0)}" if with_regex else "")
                    + " |")
        lines.append("")
    if s["answers"]:
        lines += ["### Answer files", "",
                  "| file | ran | cases | model | thinking | ok | errors | empty | "
                  "tool calls/case | zero-tool | result storage | mean wall s | "
                  "mean answer chars |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, a in s["answers"].items():
            lines.append(
                f"| {name} | {a['ran']} | {a['cases']} | {_fmt(a['models'])} | "
                f"{_fmt(a['thinking'])} | {_fmt(a['ok'])} | {a['errors']} | "
                f"{a['empty_responses']} | {_fmt(a.get('mean_tool_calls'))} | "
                f"{_fmt(a.get('zero_tool_cases'))} | {_fmt(a.get('tool_result_storage'))} | "
                f"{_fmt(a['mean_wall_s'])} | {_fmt(a['mean_response_chars'])} |")
        extra = [(n, a) for n, a in s["answers"].items() if "status" in a]
        if extra:
            lines += ["", "| file | status | critic rounds (count of cases) | split |",
                      "|---|---|---|---|"]
            for name, a in extra:
                lines.append(f"| {name} | {_fmt(a.get('status'))} | "
                             f"{_fmt(a.get('critic_rounds'))} | {_fmt(a.get('split'))} |")
        lines.append("")
    if s["thinking"]:
        lines += ["### Thinking-mode A/B", "",
                  "| file | model | thinking | cases | mean wall s | mean eval tokens | "
                  "mean answer chars | empty answers | ran |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name, rows in s["thinking"].items():
            for r in rows:
                lines.append(
                    f"| {name} | {r['model']} | {'on' if r['thinking'] else 'off'} | "
                    f"{r['cases']} | {_fmt(r['mean_wall_s'])} | {_fmt(r['mean_eval_tokens'])} | "
                    f"{_fmt(r['mean_answer_chars'])} | {r['empty_answers']} | {r['ran']} |")
        lines.append("")
    if s["duplicates"]:
        lines += ["Byte-identical files: " + "; ".join(
            f"{a} = {b}" for a, b in s["duplicates"].items()), ""]
    if s["skipped"]:
        lines += ["Skipped: " + "; ".join(f"{k} ({v})" for k, v in s["skipped"].items()), ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("directory", type=Path)
    ap.add_argument("--json", action="store_true", dest="as_json")
    ap.add_argument("--only-ids-of", type=Path, metavar="JSONL",
                    help="only count records whose candidate_id appears in this file")
    ap.add_argument("--group-by", metavar="JSONL:FIELD",
                    help="split critic files by FIELD, looked up by candidate_id in JSONL")
    ap.add_argument("--detail-regex", metavar="REGEX",
                    help="count tool_hallucination issues whose detail matches REGEX")
    args = ap.parse_args(argv)
    if not args.directory.is_dir():
        print(f"not a directory: {args.directory}", file=sys.stderr)
        return 1
    only_ids = None
    if args.only_ids_of:
        only_ids = {r.get("candidate_id") for r in load_jsonl(args.only_ids_of)}
    groups = None
    if args.group_by:
        path, _, field = args.group_by.rpartition(":")
        groups = {r.get("candidate_id"): r.get(field) for r in load_jsonl(Path(path))}
    rx = re.compile(args.detail_regex) if args.detail_regex else None
    s = summarize_dir(args.directory, only_ids, groups, rx)
    if args.as_json:
        print(json.dumps(s, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
