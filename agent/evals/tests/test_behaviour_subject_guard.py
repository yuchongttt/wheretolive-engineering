"""Degenerate-name guard for named-building subjects (behaviour/behaviour_subjects.py).

Why it exists: T7 and WEB5 pick their subject from live stock with
`ORDER BY RANDOM()` — never stale, but occasionally they draw a question that
makes no sense. Measured 2026-08-19: a listing whose building name was
literally "Building" was drawn (1 in 626), the question became "What does
Building cost now?", the agent reasonably called no tool, and the case went red
twice in a row (hard_retry picks the subject before retrying).

**A false red is as harmful as a false green** — a red that cannot be
reproduced teaches people to ignore the suite.

Public copy: the private version also runs the guard against the live listings
table (the word-boundary rule drops 'Shipbuilding Way', keeps comma-less real
names, removes < 5% of the pool, and the pool still contains degenerate names
so the checks are not vacuous). Those tests need the production DB and are not
included; the pure-Python and source-scan tests below run offline.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from behaviour_subjects import is_degenerate_building_name

SUITE = Path(__file__).resolve().parents[1] / "behaviour" / "chat_behavior_suite.py"


@pytest.mark.parametrize("name", [
    "Building", "The Tower", "the manor", "  The   Lodge  ",   # the generic word itself
    "Penthouse", "Houseboat",                                   # one word, and not a building name at all
    "", "   ",
])
def test_degenerate_names_are_rejected(name: str) -> None:
    assert is_degenerate_building_name(name)


@pytest.mark.parametrize("name", [
    "The Makers Building", "Legacy Building", "East Lodge",
    "Princess Park Manor", "Ontario Tower",
])
def test_real_building_names_are_kept(name: str) -> None:
    assert not is_degenerate_building_name(name)


def test_every_named_building_subject_query_uses_the_guard() -> None:
    """Stops someone adding a third named-building case without the guard (or
    deleting the guard).

    Only subject queries that **filter on the building name** are checked: T15
    deliberately picks a house-number address (to avoid WEB5's web branch) and
    neither should nor needs to carry this guard.
    """
    src = SUITE.read_text()
    offenders = []
    for m in re.finditer(r"ORDER BY RANDOM\(\) LIMIT 1", src):
        window = src[max(0, m.start() - 2000): m.start()]
        # take the text after the nearest db_one( so the window does not run
        # into the previous query
        cut = window.rfind("db_one(")
        q = window[cut:] if cut >= 0 else window
        selects_on_building_name = "instr(address, ',') - 1) LIKE" in q
        if selects_on_building_name and "NON_DEGENERATE_BUILDING_SQL" not in q:
            line = src[:m.start()].count("\n") + 1
            offenders.append(line)
    assert not offenders, (
        f"named-building subject query near line(s) {offenders} lacks the degenerate-name guard")


def test_the_source_scan_is_not_vacuous() -> None:
    """The scan above must actually find named-building subject queries."""
    src = SUITE.read_text()
    assert src.count("ORDER BY RANDOM() LIMIT 1") >= 3
    assert src.count("NON_DEGENERATE_BUILDING_SQL") >= 2
