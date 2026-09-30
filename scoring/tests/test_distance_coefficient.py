#!/usr/bin/env python3
"""
Walking-distance coefficient used by the transit score.

Rules (TransitConvenienceEvaluator._calculate_distance_coefficient):
  - 0-5 min:   1.0
  - 5-10 min:  linear 1.0 -> 0.7
  - 10-15 min: linear 0.7 -> 0.4
  - 15-20 min: linear 0.4 -> 0.2
  - >20 min:   keeps falling 0.01/min, floor 0.05
Station score = min(100, line_weight_sum x coefficient).
"""

import pytest

from transit_convenience_evaluator import TransitConvenienceEvaluator


@pytest.fixture(scope="module")
def evaluator():
    return TransitConvenienceEvaluator()


@pytest.mark.parametrize("walk_min, expected", [
    (0, 1.0), (3, 1.0), (5, 1.0),
    (7, 0.88), (10, 0.7),
    (12, 0.58), (15, 0.4),
    (18, 0.28), (20, 0.2),
    (25, 0.15), (40, 0.05), (90, 0.05),
])
def test_key_points(evaluator, walk_min, expected):
    assert evaluator._calculate_distance_coefficient(walk_min) == pytest.approx(expected, abs=1e-9)


def test_monotonic_non_increasing_and_bounded(evaluator):
    prev = 1.0
    for tenth in range(0, 601):  # 0.0 .. 60.0 minutes
        c = evaluator._calculate_distance_coefficient(tenth / 10)
        assert 0.05 <= c <= 1.0
        assert c <= prev + 1e-12
        prev = c


@pytest.mark.parametrize("walk_min, line_weight", [
    (3.1, 117.9),   # multi-line hub, very close
    (7.3, 129.8),   # six-line hub, ~7 min
    (3.0, 70.0), (10.0, 70.0), (15.0, 70.0),
    (8.0, 114.9),
])
def test_station_score_is_capped_at_100(evaluator, walk_min, line_weight):
    coeff = evaluator._calculate_distance_coefficient(walk_min)
    station_score = min(100, line_weight * coeff)
    assert 0 < station_score <= 100
    if walk_min <= 5 and line_weight >= 100:
        assert station_score == 100
