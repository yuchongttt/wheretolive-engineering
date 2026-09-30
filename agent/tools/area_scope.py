"""Resolve how wide an area question is, and aggregate scores across it.

A postcode score is computed per UNIT postcode (SW11 3RA — a few dozen
addresses). An outcode (SW11) covers hundreds of them, and their scores are not
close: across the 703 scored units in SW11 the schools dimension runs from 33.6
to 98.0.

Before this module nothing resolved an outcode, so `get_postcode_scores("SW11")`
found no row and reported "not yet evaluated" — leaving the chat model to invent
a representative point. It picked a different one every run. The same question
("is SW11 worth buying? all five dimensions", asked in Chinese) asked three
times in four hours returned Schools 84.2,
60.4 and 59.8, which reads as a system that cannot make up its mind; the user
re-asked twice and then rephrased.

The scope is the honest unit of answer: an outcode question gets an aggregate
over every scored unit inside it, with the spread attached so the internal
variation is stated rather than hidden behind one arbitrary point. Median rather
than mean — one £16m mansion block should not move an area's price score.
"""
from __future__ import annotations

import re
from statistics import median
from typing import Iterable, Optional

# An outcode is 1-2 letters, 1-2 digits, optionally a trailing letter
# (SW11, E14, EC1V, W1A). A sector appends one digit; a unit appends a digit
# plus two letters.
_OUTCODE = re.compile(r"^[A-Z]{1,2}\d{1,2}[A-Z]?$")
_UNIT_UNSPACED = re.compile(r"^([A-Z]{1,2}\d{1,2}[A-Z]?)(\d[A-Z]{2})$")

Scope = str  # "outcode" | "sector" | "unit"


def classify_postcode_scope(raw: str) -> tuple[Scope, str]:
    """Classify a user-supplied postcode and return (scope, normalised form).

    The internal space is load-bearing and must not be stripped before parsing:
    unspaced "SW11" is ambiguous on its own — outcode SW11, or sector SW1 1 —
    and the two are different places. UK convention writes a sector WITH the
    space ("SW1 1") and an outcode without, so the space is what disambiguates.
    Reading "SW11" as SW1 1 would answer a Battersea question with Westminster
    data.

    Case and surrounding whitespace are still normalised, so "sw113ra",
    "SW11 3RA" and " SW113RA " all resolve identically. Anything that parses as
    neither a sector nor a unit is treated as an outcode — the widest reading —
    because a partial postcode is a question about an area, not a lookup miss.
    """
    text = " ".join((raw or "").upper().split())

    head, _, tail = text.partition(" ")
    if tail:
        if re.fullmatch(r"\d[A-Z]{2}", tail):
            return "unit", f"{head} {tail}"
        if re.fullmatch(r"\d", tail):
            return "sector", f"{head} {tail}"
        return "outcode", head

    m = _UNIT_UNSPACED.match(text)
    if m:
        return "unit", f"{m.group(1)} {m.group(2)}"
    return "outcode", text


def like_pattern(scope: Scope, normalised: str) -> str:
    """SQL LIKE pattern selecting every unit postcode inside `normalised`.

    The trailing space on an outcode pattern is load-bearing: "SW1%" would also
    match SW11 and SW19, silently answering a question about SW1 with data from
    three different districts.
    """
    if scope == "unit":
        return normalised
    if scope == "sector":
        return f"{normalised}%"
    return f"{normalised} %"


def summarise_dimension(scores: Iterable[Optional[float]]) -> Optional[dict]:
    """Median + spread for one dimension across a scope's unit postcodes.

    Returns None when nothing in the scope carries this dimension, so callers
    can omit it instead of publishing a zero. Missing values are dropped rather
    than counted — a unit postcode with no schools data must not read as a
    school-less area.
    """
    vals = [float(s) for s in scores if s is not None]
    if not vals:
        return None
    return {
        "median": round(median(vals), 1),
        "min": round(min(vals), 1),
        "max": round(max(vals), 1),
        "n": len(vals),
    }
