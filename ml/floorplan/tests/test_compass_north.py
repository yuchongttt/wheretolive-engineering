"""compass_north geometry / gate contract: normalised bbox -> pixel window, white-background crop,
equivariance gate, 8-way boundary handling, property-level agreement."""
import sqlite3, sys
from pathlib import Path
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from lib import compass_north as cn  # noqa: E402


def cn_diff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def test_bbox1000_maps_to_native_pixels():
    # real example: native image 930x1288, Qwen box [899,32,969,75] (0-1000)
    cx, cy, r = cn.bbox1000_to_window([899, 32, 969, 75], 930, 1288)
    assert abs(cx - 868.6) < 1 and abs(cy - 68.9) < 1
    assert r == max(40, 0.75 * max((969 - 899) / 1000 * 930, (75 - 32) / 1000 * 1288))


def test_bbox1000_radius_bounds():
    assert cn.bbox1000_to_window([0, 0, 10, 10], 100, 100)[2] == 40        # lower bound
    assert cn.bbox1000_to_window([0, 0, 1000, 1000], 4000, 4000)[2] == 600  # upper bound


def test_crop_window_white_pads_and_flattens_alpha():
    im = Image.new("LA", (200, 200), (0, 0))            # fully transparent -> must become white
    crop = cn.crop_window(cn.flatten_on_white(im), 10, 10, 50, size=64)
    assert crop.mode == "L" and crop.size == (64, 64)
    assert min(crop.getdata()) == 255


def test_gate_decision_table():
    assert cn.gate_decision(True, 5.0, 0.9) == (True, "pass")
    assert cn.gate_decision(True, 10.0, 0.9) == (True, "pass")
    assert cn.gate_decision(True, 10.1, 0.9) == (False, "spread")
    assert cn.gate_decision(True, 5.0, 0.2) == (False, "presence")
    assert cn.gate_decision(True, 5.0, 0.45) == (True, "pass")        # threshold 0.4: a real compass at 0.488 must pass
    assert cn.gate_decision(True, 5.0, 0.39) == (False, "presence")
    assert cn.gate_decision(True, 5.0, 0.4) == (True, "pass")         # threshold is inclusive
    assert cn.gate_decision(True, float("nan"), 0.9) == (False, "unread")
    assert cn.gate_decision(False, None, None) == (False, "absent")
    assert cn.gate_decision(None, None, None) == (False, "detect_failed")
    assert cn.gate_decision(True, None, 0.9) == (False, "unread")


def test_bin8_with_margin():
    """Outside the band (> BIN_MARGIN from a 45-degree boundary) a single direction is returned -- this part
    did not change when the in-band behaviour switched to returning a pair."""
    assert cn.bin8_with_margin(0) == "N" and cn.bin8_with_margin(90) == "E"
    assert cn.bin8_with_margin(44.9) == "NE" and cn.bin8_with_margin(359) == "N"
    assert cn.bin8_with_margin(22.5 + 7.1) == "NE"
    assert cn.bin8_with_margin(292.5 - 7.1) == "W" and cn.bin8_with_margin(292.5 + 7.1) == "NW"  # W/NW boundary at 292.5


def test_bin8_boundary_returns_adjacent_pair():
    """Within +-BIN_MARGIN we no longer abstain; return the clockwise-adjacent pair "A|B" (28% of emitted
    readings used to be lost here)."""
    assert cn.bin8_with_margin(22.5) == "N|NE"                       # exactly on the boundary
    assert cn.bin8_with_margin(22.5 - 6.9) == "N|NE"
    assert cn.bin8_with_margin(22.5 + 6.9) == "N|NE"
    assert cn.bin8_with_margin(292.5) == "W|NW"


def test_bin8_boundary_pair_wraps_at_north():
    """337.5 is the NW/N boundary: lower bin first; must not wrap to "N|NW"."""
    assert cn.bin8_with_margin(337.5) == "NW|N"
    assert cn.bin8_with_margin(337.5 + 6.9) == "NW|N"
    assert cn.bin8_with_margin(337.5 - 6.9) == "NW|N"


def test_bin8_pair_components_are_truly_adjacent():
    """At all eight boundaries both components must be valid 8-way labels and clockwise-adjacent -- guards
    against an off-by-one in a hand-written table."""
    for deg in (22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5):
        a, b = cn.bin8_with_margin(deg).split("|")
        assert a in cn._BINS and b in cn._BINS, (deg, a, b)
        assert (cn._BINS.index(b) - cn._BINS.index(a)) % 8 == 1, (deg, a, b)


def test_combine_property():
    assert cn.combine_property([]) == (None, False)
    d, ok = cn.combine_property([100.0]); assert ok and d == 100.0
    d, ok = cn.combine_property([350.0, 5.0]); assert ok and cn_diff(d, 357.5) < 1e-6
    assert cn.combine_property([0.0, 20.0]) == (None, False)         # more than 15 degrees apart
    assert cn.combine_property([0.0, 10.0, 200.0]) == (None, False)  # any disagreeing pair -> abstain
    assert cn.combine_property([0.0, 14.0, 28.0]) == (None, False)   # neighbours <=15 but ends 28 apart -> must compare all pairs
    d, ok = cn.combine_property([0.0, 15.0]); assert ok and abs(d - 7.5) < 1e-6   # tolerance is inclusive of 15


def test_ensure_schema_idempotent_and_columns():
    conn = sqlite3.connect(":memory:")
    cn.ensure_schema(conn); cn.ensure_schema(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(floorplan_north)")}
    assert {"property_id", "idx", "model_version", "ok", "compass_present", "bbox_json",
            "north_deg", "spread_deg", "presence_prob", "gate", "emitted", "north_8way",
            "checked_at"} <= cols
    pcols = {r[1] for r in conn.execute("PRAGMA table_info(property_north)")}
    assert {"property_id", "north_deg", "north_8way", "n_floorplans", "n_emitted", "agree",
            "model_version", "computed_at"} <= pcols
