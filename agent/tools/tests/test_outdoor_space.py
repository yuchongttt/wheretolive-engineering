import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
from outdoor_space import (  # noqa: E402
    outdoor_display_type, is_ground_level_label, ground_label_exclusion_sql,
)


# --- outdoor_display_type: honest type comes from the LABEL, not the unreliable room_type ---

def test_display_type_prefers_label_over_room_type():
    # The bug: VLM types a ground "Patio" as room_type='terrace'. Display must say patio.
    assert outdoor_display_type("terrace", "Patio 28'3\" x 15'1\" 8.62 x 4.60m") == "patio"
    assert outdoor_display_type("balcony", "Rear Garden 26'1 x 16'10") == "garden"
    assert outdoor_display_type("terrace", "Courtyard 10' x 8'") == "courtyard"
    assert outdoor_display_type("terrace", "Decking 12' x 10'") == "decking"


def test_display_type_keeps_elevated_labels_elevated():
    assert outdoor_display_type("terrace", "Roof Terrace 20' x 10'") == "roof terrace"
    assert outdoor_display_type("balcony", "Roof Garden 12' x 8'") == "roof terrace"
    assert outdoor_display_type("balcony", "Balcony 8' x 4'") == "balcony"
    assert outdoor_display_type("terrace", "Terrace 15' x 12'") == "terrace"


def test_display_type_falls_back_to_room_type_when_label_unhelpful():
    # No outdoor keyword in the label → trust the coarse room_type, normalising the enum.
    assert outdoor_display_type("balcony", "12' x 8'") == "balcony"
    assert outdoor_display_type("roof_terrace", None) == "roof terrace"
    assert outdoor_display_type("terrace", "") == "terrace"


# --- is_ground_level_label: ground spaces excluded from an elevated-'balcony' match ---

def test_ground_level_label_detection():
    assert is_ground_level_label("Patio 28' x 15'")
    assert is_ground_level_label("Rear Garden 26' x 16'")
    assert is_ground_level_label("Courtyard 10' x 8'")
    assert is_ground_level_label("Decking 12' x 10'")
    # roof garden / roof terrace are elevated → NOT ground
    assert not is_ground_level_label("Roof Garden 12' x 8'")
    assert not is_ground_level_label("Roof Terrace 20' x 10'")
    assert not is_ground_level_label("Balcony 8' x 4'")
    assert not is_ground_level_label(None)


# --- ground_label_exclusion_sql: single source for radar_vocab / search_properties clauses ---

def test_exclusion_sql_covers_every_ground_keyword_and_keeps_roof_garden():
    sql = ground_label_exclusion_sql("ra.label").lower()
    for kw in ("patio", "courtyard", "deck", "pergola"):
        assert f"not like '%{kw}%'" in sql
    # ground gardens excluded, roof gardens preserved
    assert "not like '%garden%'" in sql and "like '%roof garden%'" in sql
