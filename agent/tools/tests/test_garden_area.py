"""Garden area, zero-false-positive caliber.

`floorplan_room_areas` has 19,492 rows whose label contains "garden", but a
naive match swallows ~6.7% wrong rows — they don't measure a garden at all
(term frequencies measured 2026-09-03):
  garden room 564 / winter garden 347  → an **indoor** conservatory; measures a room
  garden office|studio|store|house 271 → a hut in the garden; measures the shed
  communal|shared garden 114           → a shared garden, not this home's own
Same class of bug as the recently fixed "balcony posing as a garden": a field
called garden may only hold gardens.

(Extract note: the original file also compares the SQL predicate with the
Python rule over every label in the production DB, and checks a
search_properties parameter; neither runs here.)
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pipeline"))
from lib import garden_area as ga  # noqa: E402

ACCEPT = [
    'Garden 32\'8" x 10\' 9.95 x 3.04m',
    "Garden Approximate 18'3 (5.55) x 16'8 (5.08)",
    "Rear Garden 40'0 x 18'0",
    "Back Garden 30' x 20'",
    "Front Garden 12'0 x 10'0",
    "Patio Garden 15' x 9'",
    "Patio/ Garden 15' x 9'",
    "Garden extends to 60'",
    "Walled Garden 25' x 30'",
]
REJECT = [
    "Garden Room 15'1\" x 14'4\"",        # indoor conservatory
    'Winter Garden 15\'1" x 14\'4"',      # same; garden_facing excludes it too
    "Living/Dining/Wintergarden 8.77m x 3.40m",  # run together: SQL GLOB once took it for a garden (2026-09-29)
    "Garden Office 10' x 8'",             # a shed
    "Garden Studio 12' x 10'",
    "Garden Store 6' x 4'",
    "Garden House 10' x 12'",
    "Communal Garden 100' x 50'",         # not this home's
    "Shared Garden 80' x 40'",
    "Kitchen 12' x 10'",                  # no garden at all
]


def test_accepts_real_gardens():
    for s in ACCEPT:
        assert ga.is_garden_label(s), f"should accept: {s}"


def test_rejects_rooms_outbuildings_and_communal():
    for s in REJECT:
        assert not ga.is_garden_label(s), f"should reject: {s}"


def test_front_garden_is_a_garden_but_flagged_as_front():
    """A front garden counts as a garden, but usually isn't the "big garden" a buyer means — flagged separately for consumers to decide."""
    assert ga.is_garden_label("Front Garden 12'0 x 10'0")
    assert ga.is_front_garden("Front Garden 12'0 x 10'0")
    assert not ga.is_front_garden("Rear Garden 40'0 x 18'0")


def test_radar_vocab_has_garden_sqft():
    import json
    v = json.load(open(REPO / "radar_vocab.json"))
    assert "garden_sqft" in v["types"]
