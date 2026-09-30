"""Subject-selection guard for the "named building" cases in chat_behavior_suite.

T7 and WEB5 are both **named-building** contracts: T7 requires that "when the
user names a building, the agent must recognise it in OUR OWN data, with no
detour to the web"; WEB5 requires that "a named building gets one public-records
due-diligence sweep". Both pick their subject from live stock with
`ORDER BY RANDOM()`. The upside is that they never go stale; the cost is that
**occasionally they draw a question that makes no sense**.

Measured 2026-08-19: of T7's 626 candidates, 1 had a building name that was
literally "Building" (0.2%). When drawn, the question became "What does
Building cost now?". The agent not calling a tool was reasonable, yet the case
went red, twice in a row (hard_retry picks the subject before retrying).
Similar cases: 'The Tower' x21 / 'The Manor' / 'The Lodge' / 'The Mansions',
plus 'Penthouse' (a unit type, not a building name) and 'Houseboat' (a boat),
which merely happen to contain "House".

The guard removes **names that cannot uniquely identify a building**, with two
rules:
  1. at least two words — removes single-word subjects such as 'Building' /
     'Penthouse' / 'Houseboat';
  2. after dropping the article, not equal to the generic word itself — removes
     names such as 'The Tower'.

Cost: 45 of 8,686 candidates removed (0.5%), so the pool is essentially
unchanged; the benefit is that the suite no longer has a 0.5% chance of going
red for no reason. **A false green and a false red are equally harmful** — a red
that cannot be reproduced teaches people to ignore it.
"""
from __future__ import annotations

# Generic words WEB5 uses to recognise a "named building". T7 only uses 'Building'.
NAMED_BUILDING_KEYWORDS = (
    "Court", "House", "Manor", "Tower", "Building", "Mansions", "Wharf", "Lodge",
)

# Degenerate names: "after dropping the article, just the generic word". Compared upper-case.
_DEGENERATE = tuple(f"THE {k.upper()}" for k in NAMED_BUILDING_KEYWORDS)

#: Guard appended to the subject-selection WHERE clause. The caller's table must
#: have an `address` column and must already ensure `instr(address, ',') > 0`
#: (building name = first comma-separated segment).
NON_DEGENERATE_BUILDING_SQL = (
    # 1) at least two words
    " AND instr(TRIM(substr(address, 1, instr(address, ',') - 1)), ' ') > 0 "
    # 2) after dropping the article, not equal to the generic word itself
    " AND UPPER(TRIM(substr(address, 1, instr(address, ',') - 1))) NOT IN ("
    + ", ".join(f"'{d}'" for d in _DEGENERATE)
    + ") "
)


def is_degenerate_building_name(name: str) -> bool:
    """The same rule on the Python side — for tests and ad-hoc investigation."""
    n = " ".join((name or "").split())          # collapse whitespace
    if not n or " " not in n:
        return True
    return n.upper() in _DEGENERATE


_NAME_SQL = "UPPER(TRIM(substr(address, 1, instr(address, ',') - 1)))"


def name_contains_word_sql(word: str) -> str:
    """Require `word` to appear in the building name **as a whole word**, not
    embedded in another word.

    Only cases that **ask by building name** need this (T7 asks "What does
    <building> cost now?"). `LIKE '%Building%'` matches 'Shipbuilding Way' —
    that is a street; 10 of T7's 628 candidates (1.6%). When drawn, the question
    becomes "What does Shipbuilding Way cost now?"; the agent correctly has no
    building to report, yet the case goes red.

    Conversely, names like 'The Cooper Building Wharf Road' — **missing a comma,
    but the building name really is in there** — must be kept. So the rule is a
    word boundary, not a blanket "the last word must not be a street type"
    (the latter would wrongly drop 5 real building names).

    WEB5 does not need this: it asks by listing URL, not by name, and a
    street-facing address (building name present only in the Land Registry row)
    is exactly the scenario it is meant to test.
    """
    w = word.upper()
    return f" AND ({_NAME_SQL} LIKE '% {w}%' OR {_NAME_SQL} LIKE '{w} %') "
