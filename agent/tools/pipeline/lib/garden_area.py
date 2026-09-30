"""Garden area: garden size computed from dimensions printed on the floorplan
(zero-false-positive caliber, pure functions).

Source: `floorplan_room_areas` (the VLM reads floorplan labels + metric_dims
conversion — measured, not estimated). 19,492 rows / 17,138 properties have a
label containing "garden"; 15.3% active coverage; median 395 sqft.

**A naive `LIKE '%garden%'` swallows ~6.7% of rows measuring the wrong thing**
(term frequencies measured 2026-09-03) — the same class of bug as the recently
fixed "balcony posing as a garden":
  garden room 564 / winter garden 347  → an **indoor** conservatory; measures a room
  garden office|studio|store|house 271 → a hut in the garden; measures the shed
  communal|shared garden 114           → a shared garden, not this home's

So only labels where "garden" is the head noun and is NOT qualified by the
words above are accepted. The false-positive sentences are pinned in
tests/test_garden_area.py (site repo); the SQL predicate is checked against
this module over the whole DB so the two can't drift.
"""
from __future__ import annotations

import re

# These words before/after "garden" = it is not this home's own garden
_ROOM_LIKE = ("room", "office", "studio", "store", "shed", "house", "cabin", "bar", "gym")
_NOT_OURS = ("communal", "shared", "residents", "winter", "roof")

_RX_GARDEN = re.compile(r"\bgardens?\b", re.I)
_RX_ROOMLIKE = re.compile(r"\bgardens?\s+(" + "|".join(_ROOM_LIKE) + r")\b", re.I)
_RX_NOTOURS = re.compile(r"\b(" + "|".join(_NOT_OURS) + r")\s+gardens?\b", re.I)
_RX_FRONT = re.compile(r"\bfront\s+gardens?\b", re.I)


def is_garden_label(label: str | None) -> bool:
    """Does this label describe this home's own garden?"""
    if not label or not _RX_GARDEN.search(label):
        return False
    return not (_RX_ROOMLIKE.search(label) or _RX_NOTOURS.search(label))


def is_front_garden(label: str | None) -> bool:
    """A front garden is a garden, but usually not the "big garden" a buyer means — flag it and let consumers decide."""
    return bool(label and _RX_FRONT.search(label))


# SQL equivalent of is_garden_label (pinned by comparing against every label in the live DB).
GARDEN_LABEL_SQL = (
    "(lower(label) GLOB '*garden*'"
    + "".join(f" AND lower(label) NOT GLOB '*garden {w}*'" for w in _ROOM_LIKE)
    + "".join(f" AND lower(label) NOT GLOB '*gardens {w}*'" for w in _ROOM_LIKE)
    + "".join(f" AND lower(label) NOT GLOB '*{w} garden*'" for w in _NOT_OURS)
    # GLOB has no word boundary: a run-together "Wintergarden" (an indoor
    # conservatory) must be blocked explicitly; the Python side's \bgarden
    # never matches it. The 2026-09-29 full-table comparison over 23,099
    # labels found only these 3 disagreements.
    + " AND lower(label) NOT GLOB '*wintergarden*'"
    + ")"
)

#: A property's garden area = the largest qualifying garden row (front gardens
#: are usually small, so MAX naturally picks the main garden).
GARDEN_SQFT_SQL = (
    "(SELECT MAX(a.area_sqft) FROM floorplan_room_areas a "
    "WHERE a.rm_uuid = {id_col} AND a.area_sqft IS NOT NULL AND "
    + GARDEN_LABEL_SQL.replace("label", "a.label") + ")"
)
