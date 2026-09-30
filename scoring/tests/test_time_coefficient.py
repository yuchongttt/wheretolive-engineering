#!/usr/bin/env python3
"""
Travel-time coefficient for long-distance destinations (airports, mainline stations).

Design requirement: an airport reached by public transport in 30 min scores full
marks, 60 min scores 70%. Each (destination, mode) pair has its own
"full marks" and "70%" boundary; beyond that the curve keeps falling to a 0.05 floor.
"""

import pytest

from long_distance_travel_evaluator import LongDistanceTravelEvaluator


@pytest.fixture(scope="module")
def evaluator():
    return LongDistanceTravelEvaluator()


# (dest_type, mode, full-marks boundary [min], 0.7 boundary [min])
BOUNDARIES = [
    ("airport", "TRANSIT", 30, 60),
    ("airport", "DRIVE", 20, 40),
    ("train_station", "TRANSIT", 20, 40),
    ("train_station", "DRIVE", 15, 30),
]


@pytest.mark.parametrize("dest_type, mode, full, seventy", BOUNDARIES)
def test_boundaries(evaluator, dest_type, mode, full, seventy):
    coef = evaluator._calculate_travel_time_coefficient
    assert coef(full / 2, mode, dest_type) == 1.0
    assert coef(full, mode, dest_type) == pytest.approx(1.0)
    assert coef(seventy, mode, dest_type) == pytest.approx(0.7)


@pytest.mark.parametrize("dest_type, mode, _full, _seventy", BOUNDARIES)
def test_monotonic_with_floor(evaluator, dest_type, mode, _full, _seventy):
    prev = 1.0
    for minutes in range(0, 241):
        c = evaluator._calculate_travel_time_coefficient(minutes, mode, dest_type)
        assert 0.05 <= c <= 1.0
        assert c <= prev + 1e-12
        prev = c
    assert evaluator._calculate_travel_time_coefficient(600, mode, dest_type) == 0.05


def test_destination_score_scales_with_importance(evaluator):
    # score = importance x time coefficient x 100 (as in _evaluate_single_destination)
    coef = evaluator._calculate_travel_time_coefficient(60, "TRANSIT", "airport")
    heathrow = 1.0 * coef * 100
    gatwick = 0.9 * coef * 100
    assert heathrow == pytest.approx(70.0)
    assert gatwick == pytest.approx(63.0)
