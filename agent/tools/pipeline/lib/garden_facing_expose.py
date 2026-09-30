"""S4 exposure layer: merge the floorplan-derived garden aspect with the
agent-stated one into a searchable form (pure functions).

Two rules, both settled after evidence on 2026-09-03:

1. **Neither vetoes the other.** Spec §3 S1 originally said "stated and
   derived disagree → abstain for the whole property"; in a 165-property live
   cross-validation, both ≥135° disagreements turned out, on visual inspection
   of each plan, to be **the agent's copy being wrong** (the compass symbol and
   the garden block position were both correct). The original rule would have
   let the wrong side veto the right one. Instead both values coexist, each
   with its source, and disagreements are surfaced explicitly.
2. **A derived value may be two adjacent directions** (when θ falls within ±7°
   of a 45° boundary we emit "S|SW"; the truth is one of them), so the stored
   form is a **delimited set** `|south|south-west|` rather than a single value,
   and predicates match set members with LIKE.

Agreement follows the cross-validation's ≤45° rule: stated south + derived
south-west agree; + derived north is a conflict.
"""
from __future__ import annotations

CODE_TO_WORD = {
    "N": "north", "NE": "north-east", "E": "east", "SE": "south-east",
    "S": "south", "SW": "south-west", "W": "west", "NW": "north-west",
}
VALUES = frozenset(CODE_TO_WORD.values())


def derived_to_words(facing_8way):
    """8-way code in the `floorplan_north` convention (possibly "S|SW") → list of words from the stated vocabulary."""
    if not facing_8way:
        return []
    return [CODE_TO_WORD[c] for c in facing_8way.split("|") if c in CODE_TO_WORD]


def facing_set_literal(words):
    """Word list → set string with leading/trailing delimiters; empty list → None (store NULL, not an empty string)."""
    return "|" + "|".join(words) + "|" if words else None


def garden_facing_set_sql(value, col: str) -> str:
    """One direction value → a param-free predicate over the set string.

    Same caliber as `radar_vocab.garden_facing_sql`: a cardinal direction
    includes its compounds (south matches south-east/west); a compound matches
    exactly. The value passes the allowlist before being inlined.
    """
    if value not in VALUES:
        raise ValueError(f"bad garden_facing value: {value!r} (one of {sorted(VALUES)})")
    if value in ("south", "north"):
        return f"{col} LIKE '%|{value}%'"        # |south| and |south-west| both start with |south
    if value in ("east", "west"):
        return f"{col} LIKE '%{value}|%'"        # |east| and |south-east| both end with east|
    return f"{col} LIKE '%|{value}|%'"


def agrees(stated, derived_words):
    """Do stated and derived agree (≤45°, same rule as the 165-property cross-validation)? None if either is missing."""
    if not stated or not derived_words:
        return None
    for w in derived_words:
        if w == stated or w.startswith(stated + "-") or stated.startswith(w + "-"):
            return True
        if w.endswith("-" + stated) or stated.endswith("-" + w):
            return True
    return False


def agreement_sql(stated_col: str, derived_set_col: str) -> str:
    """Agreement predicate, stated vs derived set (≤45°), pinned against
    `agrees()` over all 8×72 combinations.

    The five branches map one-to-one onto agrees()'s five kinds of hit:
      1 exact member                          south      ∈ |south|
      2 derived is a compound of the stated   south      ⊂ |south-west|   (prefix)
      3 derived is a compound of the stated   east       ⊂ |south-east|   (suffix)
      4 stated is a compound of the derived   south-west ⊃ |south|        (stated's first half)
      5 stated is a compound of the derived   south-east ⊃ |east|         (stated's second half)
    """
    d, s_ = derived_set_col, stated_col
    return (
        f"({d} LIKE '%|' || {s_} || '|%'"
        f" OR {d} LIKE '%|' || {s_} || '-%'"
        f" OR {d} LIKE '%-' || {s_} || '|%'"
        f" OR (instr({s_}, '-') > 0 AND {d} LIKE '%|' || substr({s_}, 1, instr({s_}, '-') - 1) || '|%')"
        f" OR (instr({s_}, '-') > 0 AND {d} LIKE '%|' || substr({s_}, instr({s_}, '-') + 1) || '|%'))"
    )


def garden_facing_union_sql(value, stated_col: str = "garden_facing",
                            id_col: str = "id") -> str:
    """Search predicate for "stated **or** derived" (param-free).

    Same semantics as `v_garden_facing` but avoids its full scan of
    rm_sales_overview: measured 0.175s for the view vs 0.060s for this
    expression (0.053s for stated-only, so essentially free). Their
    equivalence is pinned against the live DB by tests/test_garden_facing_expose
    (site repo).
    """
    if value not in VALUES:
        raise ValueError(f"bad garden_facing value: {value!r}")
    if value in ("south", "north"):
        stated = f"({stated_col} = '{value}' OR {stated_col} LIKE '{value}-%')"
    elif value in ("east", "west"):
        stated = f"({stated_col} = '{value}' OR {stated_col} LIKE '%-{value}')"
    else:
        stated = f"{stated_col} = '{value}'"
    # Property-level cross-check: several floorplan readings for the same
    # property that disagree → abstain (same rule as the θ lane). Without the
    # HAVING clause this becomes "any one plan matching is a hit", letting in
    # rows that should have been abstained — caught on 2026-09-03 by the
    # equivalence comparison against the view (north 85 vs 86; the difference
    # was exactly that one property).
    # `outdoor_type = 'garden'`: the lane measures the "main outdoor block",
    # which may be a balcony/terrace/courtyard. 2026-09-03 audit: 127 of 341
    # emitted rows (37%) measured something other than a garden, yet all
    # entered search and Radar as garden_facing (one listing's copy said "south
    # facing rear garden"; we had measured its east-facing balcony). A field
    # called garden may only hold gardens; balcony/terrace aspects stay in the
    # table for later use and are not consumed here.
    derived = (f"{id_col} IN (SELECT property_id FROM garden_facing_derived "
               f"WHERE emitted = 1 AND facing_set IS NOT NULL AND outdoor_type = 'garden' "
               f"GROUP BY property_id "
               f"HAVING COUNT(DISTINCT facing_set) = 1 AND "
               f"{garden_facing_set_sql(value, col='MIN(facing_set)')})")
    return f"({stated} OR {derived})"
