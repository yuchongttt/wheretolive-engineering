"""The TfL journey planner's default transport modes must include national-rail (2026-08-25).

Trigger: live commute lookups for outer-London postcodes returned times absurd enough to
make a good area look unlivable. Measured: BR5 3AA → London Bridge reported 95 min,
routed as **walk to St Mary Cray station, then three buses in a row** (273→126→132)
round to North Greenwich — while Southeastern runs direct from that station to London
Bridge in about 25 min. Orpington (BR6) → London Bridge reported 101 min.

The root cause was mode-list drift: the deliberately built commute paths (the offline
sector→hub table builder and a rail+walk verifier) put national-rail in their mode
strings, but the **default** of get_journey in apis/tfl.py was missing it — and the live
commute lookup relies on that default. Barred from National Rail, TfL can only return bus
solutions, so the whole outer ring was distorted.

Measured after adding national-rail: BR5 3AA → London Bridge 95→62 min (matching the
65 min in the precomputed sector_hub_commute table), BR6 0AA → London Bridge 101→53 min.

This pins down "default = the shared all-modes constant", so the next drift turns the
tests red rather than users' commute times. (Two further cases that import the offline
builder scripts, which are not part of this module, were dropped.)
"""
import inspect

from apis.tfl import DEFAULT_JOURNEY_MODE, TfLAPI


def _modes(s):
    return {m.strip() for m in s.split(",") if m.strip()}


def test_default_journey_mode_includes_national_rail():
    """Without it, TfL can only cobble an answer together from buses — outer-ring commute times simply double."""
    assert "national-rail" in _modes(DEFAULT_JOURNEY_MODE)


def test_get_journey_default_argument_is_the_shared_constant():
    """Callers that pass no mode get this default — which must be the shared constant itself,
    not another hand-copied string (hand-copying is exactly what caused this drift)."""
    default = inspect.signature(TfLAPI.get_journey).parameters["mode"].default
    assert default == DEFAULT_JOURNEY_MODE
