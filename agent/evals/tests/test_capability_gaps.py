"""Capability-gap mining: when the agent admits in a reply that "I don't have
this data / tool", pull it out.

Founding evidence (2026-08-18): the R2-9 seasonality gap had been stated by the
agent on 2026-07-11 in language precise enough to start work from ("my tools
have no field like 'historical listing volume by month'"); that sentence was
written to assistant_text, never read by anyone, and rediscovered five weeks
later by a hand-run benchmark question.

The fixtures are Chinese on purpose — the assistant is bilingual and the
detectors match Chinese replies. English glosses are given next to each one.
Session ids are synthetic.
"""
import sqlite3
from pathlib import Path

import capability_gaps as cg


def _db(tmp_path: Path, rows) -> Path:
    p = tmp_path / "evaluations.db"
    c = sqlite3.connect(p)
    c.execute("""CREATE TABLE chat_messages (id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, user_id TEXT, session_id TEXT, turn_idx INTEGER,
        user_text TEXT, assistant_text TEXT, tool_calls_json TEXT)""")
    c.executemany(
        "INSERT INTO chat_messages (ts,user_id,session_id,turn_idx,user_text,"
        "assistant_text,tool_calls_json) VALUES (?,?,?,?,?,?,?)", rows)
    c.commit()
    c.close()
    return p


# A representative question and reply (paraphrased).
# Q: "Which month usually has the most listings and the best prices?"
R2_9_Q = "哪个月份挂牌最多、价格最合适"
# A: "This is a question about market patterns, not a data lookup for a specific
# listing/area; my tools have no field like 'historical listing volume by
# month', so I'm answering this part from publicly known UK seasonal patterns,
# not from specific numbers looked up in my tools — just to be clear."
R2_9_A = ("这是个市场规律性问题,不是某个具体房源/地区的数据查询,我的工具里没有"
          "「按月份统计历史挂牌量」这类字段,所以这部分我用英国房产市场公开的"
          "季节性规律来回答,不是从工具里查出来的具体数字,跟你说清楚。")

REAL = ("2026-01-05 12:00:00", "u-test", "web-0000000000001-aaaaaa", 9)


def test_catches_the_founding_case(tmp_path):
    db = _db(tmp_path, [(*REAL, R2_9_Q, R2_9_A, "[]")])
    gaps = cg.scan(db)
    assert len(gaps) == 1
    g = gaps[0]
    assert g["question"] == R2_9_Q
    assert g["source"] == "real"
    # zero tools + an admission = a hard gap: the turn never touched any data
    assert g["severity"] == "hard"


def test_evidence_is_the_sentence_not_the_whole_answer(tmp_path):
    db = _db(tmp_path, [(*REAL, R2_9_Q, R2_9_A, "[]")])
    ev = cg.scan(db)[0]["evidence"]
    assert "按月份统计历史挂牌量" in ev          # the missing capability must be in the evidence
    assert len(ev) < len(R2_9_A)                  # but not a copy of the whole answer
    assert "跟你说清楚" not in ev                 # only that clause ("just to be clear" is cut)


def test_tools_were_called_means_partial_not_hard(tmp_path):
    # Tools called and still admitting a gap = incomplete coverage, milder than
    # "never touched the data".
    # Q: "price trends for every UK area"; A: "my data covers London, not the whole UK."
    db = _db(tmp_path, [(*REAL, "英国所有区房价趋势",
                         "我的数据覆盖的是伦敦,不是全英国。", '[{"name":"get_price_trend"}]')])
    assert cg.scan(db)[0]["severity"] == "partial"


def test_negated_sentence_does_not_fire(tmp_path):
    # A bare regex matching a negated sentence is a known self-inflicted failure
    # (R2-8 lesson 8b).
    # A: "It's not that I don't have data — on the contrary, this home's sales
    # history is complete."
    db = _db(tmp_path, [(*REAL, "这套房怎么样",
                         "并不是我没有数据,恰恰相反,这套房的成交史很完整。", "[]")])
    assert cg.scan(db) == []


def test_bench_traffic_excluded_but_simulation_kept(tmp_path):
    """sim- is our own weekly simulation — the second sensor, not noise.

    The 2026-08-12 simulation hit the same seasonality gap and waved it through;
    cmp-/bhv-/selftest- are my own benchmark and contract runs, whose answers I
    already know, so they are dropped.
    """
    rows = [
        ("2026-08-12 07:04:07", "admin", "sim-explore-20260812-0700-a", 0, R2_9_Q, R2_9_A, "[]"),
        ("2026-08-18 13:28:36", "admin", "cmp-round-02-r2-9-zh-1787", 0, R2_9_Q, R2_9_A, "[]"),
        ("2026-08-18 14:00:00", "admin", "bhv-len-2000", 0, R2_9_Q, R2_9_A, "[]"),
    ]
    gaps = cg.scan(_db(tmp_path, rows))
    assert [g["source"] for g in gaps] == ["sim"]


def test_same_gap_collapses_to_one_key_across_repeats(tmp_path):
    # In the real data the same brief was resent verbatim once.
    rows = [(*REAL, R2_9_Q, R2_9_A, "[]"),
            ("2026-01-07 12:00:00", "u-real", "web-other", 0, R2_9_Q + "？", R2_9_A, "[]")]
    keys = {g["key"] for g in cg.scan(_db(tmp_path, rows))}
    assert len(keys) == 1, "the same question (punctuation-only difference) should fold into one gap"


def test_different_gaps_keep_separate_keys(tmp_path):
    rows = [(*REAL, R2_9_Q, R2_9_A, "[]"),
            ("2026-01-08 12:00:00", "u-real", "web-x", 0, "英国所有区房价趋势",
             "我的数据覆盖的是伦敦,不是全英国。", "[]")]
    assert len({g["key"] for g in cg.scan(_db(tmp_path, rows))}) == 2


def test_since_filter_only_returns_newer_turns(tmp_path):
    rows = [(*REAL, R2_9_Q, R2_9_A, "[]"),
            ("2026-02-01 12:00:00", "u-real", "web-new", 0, R2_9_Q, R2_9_A, "[]")]
    db = _db(tmp_path, rows)
    assert len(cg.scan(db, since="2026-01-15T00:00:00Z")) == 1
    assert len(cg.scan(db)) == 2


def test_clean_answer_produces_nothing(tmp_path):
    # Q: "W5 price trend"; A: "W5 3-year CAGR 4.2%, sample of 812 sales."
    db = _db(tmp_path, [(*REAL, "W5 房价趋势",
                         "W5 近三年 CAGR 4.2%,样本 812 笔。", '[{"name":"get_price_trend"}]')])
    assert cg.scan(db) == []


def test_ranks_hard_gaps_before_partial(tmp_path):
    rows = [("2026-07-01 10:00:00", "u", "web-a", 0, "问A", "我的数据覆盖的是伦敦。",
             '[{"name":"x"}]'),
            (*REAL, R2_9_Q, R2_9_A, "[]")]
    assert [g["severity"] for g in cg.scan(_db(tmp_path, rows))] == ["hard", "partial"]


def test_missing_table_is_not_fatal(tmp_path):
    """The self-review's minimal test skeleton has no chat_messages; gap mining
    must not take down a whole self-review run."""
    p = tmp_path / "bare.db"
    sqlite3.connect(p).close()
    assert cg.scan(p) == []


# --- Second kind of signal: explicitly asked to search, zero tools (2026-08-18) ---
# Measured on 120 real turns: 26 had zero tools, but most were entirely
# legitimate (greetings / the user adding information / follow-ups on existing
# results). **Naive "zero tools = suspicious" is far too imprecise, and it gets
# dominated by failed turns**: the three "announced a search but didn't run it"
# turns from 2026-05 were actually status=error / timeout (exit 143), not a
# behaviour problem. So this signal requires all of: an explicit search
# instruction from the user + status=ok this turn + zero tools.

_DB2_SEQ = [0]


def _db2(tmp_path, rows):
    """A DB with a status column (scan uses it to drop failed turns). A new file
    per call — reusing the path when building DBs in a loop inside one test
    would collide on CREATE TABLE."""
    _DB2_SEQ[0] += 1
    p = tmp_path / f"s{_DB2_SEQ[0]}.db"
    c = sqlite3.connect(p)
    c.execute("""CREATE TABLE chat_messages (id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, user_id TEXT, session_id TEXT, turn_idx INTEGER, user_text TEXT,
        assistant_text TEXT, tool_calls_json TEXT, status TEXT)""")
    c.executemany("INSERT INTO chat_messages (ts,user_id,session_id,turn_idx,user_text,"
                  "assistant_text,tool_calls_json,status) VALUES (?,?,?,?,?,?,?,?)", rows)
    c.commit(); c.close()
    return p


REAL2 = ("2026-01-06 12:00:00", "u-test", "web-0000000000002-bbbbbb", 0)


def test_explicit_search_request_answered_with_questions_is_flagged(tmp_path):
    """Real pattern: the user gave budget + area + unit type and said "find me a
    few"; the reply was a question and zero listings."""
    # Q: "Budget £600k, want a 2-bed in South London with an easy commute — find
    # me a few." A: "East London is broad — which direction do you usually commute?"
    db = _db2(tmp_path, [(*REAL2, "预算 60 万镑,想在南伦敦买个通勤方便的两居,帮我找几套",
                          "南伦敦范围较广,您平时通勤主要去哪个方向?", "[]", "ok")])
    gaps = cg.scan(db)
    assert [g["label"] for g in gaps] == ["search_not_run"]
    assert gaps[0]["severity"] == "hard"


def test_a_failed_turn_is_never_reported_as_a_behaviour_gap(tmp_path):
    """The 2026-05 "announced a search but didn't run it" turns were timeouts /
    exit 143. Treating errors as behaviour problems would drown the signal."""
    # Q: "Find me a few 2-beds in East London"; A: "Let me look that up for you…"
    for st in ("error", "timeout"):
        db = _db2(tmp_path, [(*REAL2, "帮我找几套东伦敦的两居", "让我帮你查…", "[]", st)])
        assert cg.scan(db) == [], st


def test_a_search_request_that_did_call_tools_is_clean(tmp_path):
    # A: "Found 6…"
    db = _db2(tmp_path, [(*REAL2, "帮我找几套东伦敦的两居", "找到 6 套…",
                          '[{"name":"search_properties"}]', "ok")])
    assert cg.scan(db) == []


def test_chitchat_and_user_supplied_facts_are_not_flagged(tmp_path):
    """Legitimate zero-tool turns: greetings, the user adding information,
    follow-ups on existing results."""
    # "hello", "the service charge is 0, ground rent 150 a year", "give me the
    # link", "so if you had to pick, which one?"; reply: "OK."
    for q in ("哈喽", "service charge 是 0，ground rent 每年 150", "链接给我", "所以要你选你选哪个"):
        db = _db2(tmp_path, [(*REAL2, q, "好的。", "[]", "ok")])
        assert cg.scan(db) == [], q


def test_similarity_search_without_its_referent_is_not_a_miss(tmp_path):
    """"Find a similar home" is defined by an image; if the image was not
    uploaded no search can run, and asking back is correct."""
    # Q: "Find me a similar home: commute within 30 min, 600k, 2 beds";
    # A: "The image couldn't be read — could you send it again?"
    db = _db2(tmp_path, [(*REAL2, "帮我找一个类似的房子，要求：通勤 30 分钟内，60 万，两房",
                          "图片没能读取，能重新发一下吗？", "[]", "ok")])
    assert cg.scan(db) == []


# --------------------------------------------------------------------------
# Self-review F5 (2026-09-24): a correct refusal is not a skipped search
# --------------------------------------------------------------------------
SIM_RENT = ("2026-09-23 07:05:00", "u-sim", "sim-explore-20260923-0700-1", 6)
BOUNDARY_A = ("I have to be upfront again here: my `search_properties` tool only covers "
              "**for-sale** listings — I have no inventory of live **rental** listings, so I "
              "genuinely cannot pull you actual flats-to-let with viewing availability this "
              "weekend. Anything I invented here would be a fabricated listing, which I won't do.")


def test_a_correct_refusal_that_declares_the_boundary_is_not_a_miss(tmp_path):
    """explore-20260923 P25: there is no search lane for rentals; refusing to
    fabricate is correct and must not be scored hard."""
    db = _db2(tmp_path, [(*SIM_RENT,
                          "find me 2 bed flats in E14 under £2,000 a month to view this weekend",
                          BOUNDARY_A, "[]", "ok")])
    assert cg.scan(db) == []


def test_ok_is_not_a_budget_constraint(tmp_path):
    """"ok find me some actual places…" was once treated as "enough constraints"
    because `k\\b` matched the "ok"."""
    db = _db2(tmp_path, [(*SIM_RENT, "ok find me some actual places I can go and view this weekend",
                          "Which area and budget are you thinking of?", "[]", "ok")])
    assert cg.scan(db) == []


def test_a_real_k_budget_still_counts(tmp_path):
    db = _db2(tmp_path, [(*REAL2, "find me a 2 bed in E14 for 500k",
                          "Which direction do you commute?", "[]", "ok")])
    assert [g["label"] for g in cg.scan(db)] == ["search_not_run"]


def test_only_covers_london_does_not_excuse_a_skipped_for_sale_search(tmp_path):
    """Review (minor): `only covers?` was too wide — "our data only covers London"
    is not a capability boundary; the user gave enough constraints and wants to
    buy, yet got zero tools and a budget question: still a skipped search."""
    db = _db2(tmp_path, [(*REAL2, "find me a 2 bed flat in E14 for 500k",
                          "Our data only covers London — what's your budget and move-in date?",
                          "[]", "ok")])
    assert [g["label"] for g in cg.scan(db)] == ["search_not_run"]
