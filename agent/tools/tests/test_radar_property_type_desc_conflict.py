"""radar's property_type=house let in listings whose description says they are flats.

Background (2026-09-03): a public Radar (house only, with a price cap)
had three ptype_desc_flat=1 listings in its match set — agents type
property_type by hand, and at the cheap end 17-21% of "houses" are really
flats/maisonettes (measured 2026-08-27: a 9th-floor one-bed labelled
"Terraced"; removing them lifted Tower Hamlets' house P25 from £650k to £685k).

The guard itself was already productised — the ptype_desc_flat column, and
search_properties prints "⚠ TYPE CONFLICT" — but radar_vocab._property_type
only compared LOWER(o.property_type), so this path was never wired up. The
same filter would warn "the description says this is a flat" in chat, and
silently push it to the user as a house in radar.

House direction only: ptype_desc_flat is zero-false-positive by design (set to
1 only when the description's subject calls itself a flat/maisonette and there
is no house evidence), so using it to drop houses is safe; pulling
house-labelled rows INTO flat results would change recall and is out of scope.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import radar_vocab as rv  # noqa: E402


def _where(value):
    w, params, _join, _jp = rv._property_type({"type": "property_type", "value": value})
    return w, params


def test_house_excludes_description_declared_flats():
    w, _ = _where("house")
    assert "ptype_desc_flat" in w, (
        "the house family must drop rows whose description says flat, or radar pushes flats as houses:\n" + w)


def test_house_still_matches_rows_never_checked():
    """Not yet stamped (NULL) does not mean flat — under the zero-FP design NULL
    must pass, or new listings not yet enriched are wrongly killed in bulk."""
    w, _ = _where("house")
    assert "COALESCE" in w.upper() or "IS NULL" in w.upper(), (
        "NULL must be let through explicitly, not silently swallowed by = 0:\n" + w)


def test_flat_family_unchanged():
    """Reverse direction untouched: recall for a flat request stays as it was."""
    w, params = _where("flat")
    assert "ptype_desc_flat" not in w, w
    assert "flat" in [str(p).lower() for p in params]


def test_specific_subtype_unchanged():
    """A specific subtype (maisonette) still matches exactly, without the house-family exclusion."""
    w, params = _where("maisonette")
    assert params == ["maisonette"]
    assert "ptype_desc_flat" not in w, w


def test_mixed_house_and_flat_request_does_not_filter_the_flat_side():
    """Someone asking for both house and flat wants both anyway — dropping is
    pointless here, and an AND on one IN list would narrow the whole condition
    (the flat side included)."""
    w, _ = _where(["house", "flat"])
    assert "ptype_desc_flat" not in w, (
        "with both families requested, nothing should be dropped:\n" + w)
