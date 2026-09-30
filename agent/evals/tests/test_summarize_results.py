"""summarize_results.py on a tiny synthetic fixture (tests/fixtures/results_sample).

The fixture has one of each record shape: an answer file (3 cases: grounded,
zero-tool, failed with a 300-char preview), its critic file (pass / real
failure / critic parse failure / empty-answer shortcut), a thinking A/B file,
and a query file that must be skipped.
"""
import json
import shutil
from pathlib import Path

import pytest

import summarize_results as sr

FIX = Path(__file__).resolve().parent / "fixtures" / "results_sample"


@pytest.fixture(scope="module")
def summary():
    return sr.summarize_dir(FIX)


def test_critic_verdicts_and_parse_failures(summary):
    c = summary["critic"]["run_a.critic.jsonl"]
    assert c["cases"] == 4
    assert c["verdicts"] == {"pass": 1, "needs_revision": 3}
    assert c["parse_failures"] == 1
    assert c["empty_answers"] == 1
    assert c["critic_call_failures"] == 0
    assert c["pass_rate"] == 0.25
    # a critic parse failure is critic noise: excluded from the judged denominator
    assert c["pass_rate_judged"] == round(1 / 3, 3)
    assert c["scored"] == "2026-05-10..2026-05-11"
    assert c["cache_hits"] == 1


def test_critic_issue_counts_exclude_parse_markers(summary):
    c = summary["critic"]["run_a.critic.jsonl"]
    # the empty-answer "format/high" issue IS a model failure and stays counted
    assert c["issues_by_severity"] == {"high": 2, "medium": 1, "low": 1}
    cats = c["issues_by_category"]
    assert cats["tool_hallucination"] == 1
    assert cats["general_hallucination"] == 1
    assert cats["format"] == 2
    assert cats["incomplete"] == 0          # only the parse-failure marker had this category
    assert list(cats)[:6] == list(sr.CATEGORIES)
    assert c["cases_with_high"] == 2
    assert c["cases_with_tool_hallucination"] == 1


def test_answer_file_summary(summary):
    a = summary["answers"]["run_a.jsonl"]
    assert a["cases"] == 3
    assert a["models"] == ["toy-9b"]
    assert a["thinking"] == ["False"]
    assert a["ran"] == "2026-05-10..2026-05-11"
    assert (a["ok"], a["errors"], a["empty_responses"]) == (2, 1, 1)
    assert a["mean_tool_calls"] == 0.7
    assert a["zero_tool_cases"] == 1
    assert a["tool_result_storage"] == {"full": 1, "preview": 1}
    assert (a["mean_wall_s"], a["median_wall_s"]) == (20.0, 20.0)
    assert a["cites_stripped"] == 1


def test_thinking_ab_grouped_by_model_and_mode(summary):
    rows = summary["thinking"]["thinking_ab.jsonl"]
    off, on = rows  # sorted: thinking False first
    assert (off["thinking"], on["thinking"]) == (False, True)
    assert (off["cases"], on["cases"]) == (2, 2)
    assert (off["mean_wall_s"], on["mean_wall_s"]) == (15.0, 150.0)
    assert on["empty_answers"] == 1 and off["empty_answers"] == 0
    assert on["mean_eval_tokens"] == 2548.0


def test_unknown_shapes_are_skipped_not_guessed(summary):
    assert summary["skipped"] == {"queries.jsonl": "unknown"}


def test_byte_identical_files_are_reported(tmp_path):
    for p in FIX.glob("*.jsonl"):
        shutil.copy(p, tmp_path / p.name)
    shutil.copy(FIX / "run_a.critic.jsonl", tmp_path / "run_b.critic.jsonl")
    s = sr.summarize_dir(tmp_path)
    assert s["duplicates"] == {"run_b.critic.jsonl": "run_a.critic.jsonl"}
    assert s["critic"]["run_b.critic.jsonl"] == s["critic"]["run_a.critic.jsonl"]


def test_only_ids_restricts_every_file():
    s = sr.summarize_dir(FIX, only_ids={"q1", "q2"})
    assert s["critic"]["run_a.critic.jsonl"]["cases"] == 2
    assert s["answers"]["run_a.jsonl"]["cases"] == 2


def test_group_by_and_detail_regex():
    groups = {r["candidate_id"]: r["filter_class"] for r in sr.load_jsonl(FIX / "queries.jsonl")}
    import re
    s = sr.summarize_dir(FIX, groups=groups, detail_regex=re.compile(r"score of \d+"))
    by = s["critic_by_group"]["run_a.critic.jsonl"]
    assert set(by) == {"advice", "filter", "follow_up"}
    fu = by["follow_up"]
    assert (fu["cases"], fu["pass"]) == (2, 0)
    assert fu["issues_by_category"]["tool_hallucination"] == 1
    assert fu["tool_hallucination_matching_regex"] == 1
    assert by["advice"]["parse_failures"] == 1


def test_markdown_render_has_every_section(summary):
    md = sr.render_markdown(summary)
    assert "### Critic verdicts" in md
    assert "| run_a.critic.jsonl | 2026-05-10..2026-05-11 | 4 | 1 | 3 | 1 |" in md
    assert "### Answer files" in md and "| run_a.jsonl |" in md
    assert "### Thinking-mode A/B" in md
    assert "Skipped: queries.jsonl (unknown)" in md


def test_cli(capsys, tmp_path):
    assert sr.main([str(tmp_path / "missing")]) == 1
    assert sr.main([str(FIX), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["critic"]["run_a.critic.jsonl"]["pass"] == 1
    assert sr.main([str(FIX), "--group-by", f"{FIX / 'queries.jsonl'}:filter_class",
                    "--detail-regex", "72"]) == 0
    assert "### Critic verdicts by group" in capsys.readouterr().out
