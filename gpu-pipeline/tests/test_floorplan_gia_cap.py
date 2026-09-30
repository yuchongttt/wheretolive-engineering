"""The GIA cap must not compress outdoor spaces (2026-09-29, caught in a review).

apply_gia_cap scales indoor rooms down proportionally when "sum of indoor room
areas > total floor area"; outdoor spaces are not part of GIA and by design
don't take part. But it only looked at room_type, and the VLM prompt's type enum
has **no garden / patio**, so a plain "Garden" / "Patio" / "Driveway" is typed
'other' — and got compressed together with the indoor rooms: one property's
"Garden 9.75 x 3.05m" (320 sqft) was stored as 305, the whole flat x0.9546, even
the kitchen went from 399 to 381. The production table had ~25k 'other' rows with
outdoor labels (garden 14k, "garden approximate" 4.2k, front garden 1.1k,
balcony 1.0k, driveway 616, ...).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mac" / "floorplan_vlm"))
import dispatcher as d  # noqa: E402


def _room(label, typ):
    a = {"label": label, "type": typ, "floor": "ground"}
    a.update(d.extract_room_area(label) or {"area_sqft": None})
    return a


def test_garden_typed_other_is_not_compressed():
    areas = [
        _room("Kitchen/Reception/Diner 8.69 x 4.27m 28'6\" x 14'0\"", "kitchen"),
        _room("Utility 3.10 x 1.22m 10'2\" x 4'0\"", "utility"),
        _room("Bedroom 1 5.94 x 3.43m 19'6\" x 11'3\"", "bedroom"),
        _room("Bedroom 2 4.57 x 4.27m 15'0\" x 14'0\"", "bedroom"),
        _room("Garden 9.75 x 3.05m 32'0\" x 10'0\"", "other"),
    ]
    before = {a["label"].split()[0]: a["area_sqft"] for a in areas}
    capped = d.apply_gia_cap(areas, 1135)
    # indoor sum 399+41+219+210=869 < 1135 — the cap should never have fired
    assert capped is False
    assert {a["label"].split()[0]: a["area_sqft"] for a in areas} == before
    assert areas[-1]["area_sqft"] == 320


def test_outdoor_rows_typed_other_do_not_count_toward_indoor_sum():
    # indoor really does exceed GIA: compress indoor only, keep outdoor
    # (balcony/driveway typed 'other') as-is
    areas = [
        _room("Living Room 6.00 x 5.00m", "living"),     # 323
        _room("Bedroom 5.00 x 4.00m", "bedroom"),        # 215
        _room("Balcony 4.00 x 2.00m", "other"),          # 86
        _room("Driveway 6.00 x 3.00m", "other"),         # 194
    ]
    assert d.apply_gia_cap(areas, 450) is True
    by = {a["label"].split()[0]: a["area_sqft"] for a in areas}
    assert by["Living"] + by["Bedroom"] <= 451
    assert by["Balcony"] == 86 and by["Driveway"] == 194


@pytest.mark.parametrize("label", [
    "Garden 9.75 x 3.05m", "Garden approximate 12.0 x 8.0m", "Front Garden 5.1 x 4.0m",
    "Rear Garden 10.2 x 6.1m", "Back garden 9m x 7m", "Side Garden 3 x 2m",
    "Communal Garden 20 x 10m", "Shared garden 15 x 9m", "Private Garden 6 x 5m",
    "Garden (max dims) 12 x 9m", "Garden extends to 30'0\"", "Patio 4.0 x 3.0m",
    "Patio/ Garden 8 x 6m", "Garden / Driveway 6 x 5m", "Front Patio 3 x 2m",
    "Courtyard 5.0 x 4.0m", "Terrace 6 x 3m", "Roof Terrace 7 x 4m", "Balcony 3.5 x 1.5m",
    "Decking 4 x 3m", "Deck 4 x 3m", "Yard 4 x 3m", "Front Yard 4 x 3m",
    "Driveway 6 x 3m", "Off Street Parking 5 x 2.5m", "Parking Space 5 x 2.5m",
    "Flat Roof 4 x 3m", "Garden\napproximate 12 x 8m", "Drive 6 x 3m", "Front Drive 6 x 3m",
    "Front drive way approximate 6 x 3m", "Forecourt approximate 5 x 4m", "Decked area 4 x 3m",
])
def test_outdoor_labels(label):
    assert d.is_outdoor_label(label), label


@pytest.mark.parametrize("label", [
    # buildings in the garden / indoor sunrooms: not outdoor space, keep the
    # existing handling
    "Garden Room 4.0 x 3.0m", "Garden Office 3 x 3m", "Garden Studio 4 x 3m",
    "Garden House 5 x 4m", "Garden Store 2 x 1m", "Garden Shed 2 x 2m",
    "Garden Storage 2 x 1m", "Winter Garden 4 x 2m", "Kitchen/Garden Room 6 x 5m",
    # indoor rooms that merely mention the outside
    "Bedroom 1 4.2 x 3.1m (garden access)", "Living Room opening to patio 5 x 4m",
    "Kitchen overlooking courtyard 4 x 3m", "Loft 5 x 4m", "Landing 3 x 1m", "",
    "Garage (shared drive) 5 x 3m", "Garden / Gym 4 x 3m",
])
def test_not_outdoor_labels(label):
    assert not d.is_outdoor_label(label), label


def test_indoor_typed_row_mentioning_garden_is_still_indoor():
    # a row already typed as an indoor room still counts as indoor even if its
    # label mentions garden (the label is only consulted for other/empty types)
    areas = [_room("Bedroom 1 6.00 x 5.00m (garden access)", "bedroom"),
             _room("Kitchen 5.00 x 4.00m", "kitchen")]
    assert d.apply_gia_cap(areas, 400) is True
    assert sum(a["area_sqft"] for a in areas) <= 401


def test_garden_room_typed_other_still_counts_as_indoor():
    areas = [_room("Garden Room 5.00 x 4.00m", "other"), _room("Kitchen 5.00 x 4.00m", "kitchen")]
    assert d.apply_gia_cap(areas, 300) is True
    assert areas[0]["area_sqft"] < 215


# ── Two more gates with the same gap (2026-09-29, continued): balconies/terraces
# typed 'other' must go through them too ─────────
# flag_implausible_balcony and outdoor_shape_audit used to key only on
# type=='balcony' / ('terrace','patio'). Measured: 2,646 'other' balcony rows
# (1,053 with an area); 3 of them larger than the whole flat's GIA — the bounding
# rectangle of a wrap-around balcony (e.g. "Balcony 14.92m x 10.83m" = 1,739 sqft
# for a 1,074 sqft flat) — yet they slipped past the gate because of the type.

import sqlite3  # noqa: E402


def _conn(irr_bal=0, irr_terr=0, uid="u1"):
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE outdoor_shape_audit (rm_uuid TEXT PRIMARY KEY, irregular_outdoor INTEGER, "
              "irregular_terrace INTEGER)")
    c.execute("INSERT INTO outdoor_shape_audit VALUES (?,?,?)", (uid, irr_bal, irr_terr))
    return c


@pytest.mark.parametrize("label,typ,kind", [
    ("Balcony 3.5 x 1.5m", "other", "balcony"),
    ("Balcony 3.5 x 1.5m", "balcony", "balcony"),
    ("Roof Terrace 7 x 4m", "other", "terrace"),
    ("Patio 4 x 3m", "other", "patio"),
    ("Patio/ Terrace 4 x 3m", "other", "terrace"),
    ("Patio/ Garden 8 x 6m", "other", "outdoor"),      # mixed: not treated as balcony/terrace
    ("Garden 9.75 x 3.05m", "other", "outdoor"),
    ("Garden Room 4 x 3m", "other", None),
    ("Kitchen 4 x 3m", "kitchen", None),
])
def test_outdoor_kind(label, typ, kind):
    assert d._outdoor_kind({"label": label, "type": typ}) == kind


def test_oversized_balcony_typed_other_is_nulled():
    areas = [_room("Balcony 49'0\" x 35'5\" 14.92m x 10.83m", "other"),
             _room("Reception 6.00 x 5.00m", "reception")]
    assert d.flag_implausible_balcony(areas, 1074) == 1
    assert areas[0]["area_sqft"] is None and areas[0]["method"] == "suspect_oversized"
    assert areas[1]["area_sqft"] == 323


def test_oversized_garden_typed_other_is_not_touched_by_balcony_guard():
    # a garden far larger than the indoor floor area is normal; this gate is
    # only for balconies
    areas = [_room("Garden 29.89 x 14.57m", "other")]
    assert d.flag_implausible_balcony(areas, 1074) == 0
    assert areas[0]["area_sqft"] == 4688


def test_shape_audit_nulls_irregular_balcony_typed_other():
    parsed = {"total_sqft": 900, "_room_areas": [
        _room("Balcony 6.00 x 4.00m", "other"), _room("Garden 10.00 x 8.00m", "other")]}
    out = d.compute_room_areas(_conn(irr_bal=1), "u1", parsed)
    assert out[0]["area_sqft"] is None and out[0]["method"] == "suspect_irregular_shape"
    assert out[1]["area_sqft"] == 861      # the garden is not affected by the balcony audit


def test_shape_audit_nulls_irregular_terrace_typed_other():
    parsed = {"total_sqft": 900, "_room_areas": [_room("Roof Terrace 7.00 x 4.00m", "other")]}
    out = d.compute_room_areas(_conn(irr_terr=1), "u1", parsed)
    assert out[0]["area_sqft"] is None and out[0]["method"] == "suspect_irregular_shape"
