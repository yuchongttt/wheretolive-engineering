#!/usr/bin/env python3
"""
Line-weight sum for a station.

The original version of this file printed a table against an earlier
"diminishing returns" design (x3.5 for the first line, x1.2 for the second, ...)
and asserted nothing. The current code deliberately sums the per-line weights
with no discount ("more lines is a real advantage"), and the cap to 100 happens
later, when the sum is multiplied by the distance coefficient. These tests pin
that current behaviour.
"""

import pytest

from transit_convenience_evaluator import TransitConvenienceEvaluator


@pytest.fixture(scope="module")
def evaluator():
    return TransitConvenienceEvaluator()


@pytest.mark.parametrize("lines, expected", [
    ([], 0),
    (["central"], 40),                                   # top tier
    (["waterloo-city"], 20),                             # lowest tier
    (["central", "jubilee"], 80),                        # two top-tier lines
    (["central", "jubilee", "dlr"], 110),                # sum is not capped here
    (["northern", "victoria", "piccadilly", "metropolitan", "circle"], 185),
])
def test_weights_are_summed_without_discount(evaluator, lines, expected):
    assert evaluator._calculate_line_weight_with_diminishing_returns(lines) == expected


def test_unknown_line_defaults_to_20_and_lookup_is_case_insensitive(evaluator):
    assert evaluator._calculate_line_weight_with_diminishing_returns(["Some New Line"]) == 20
    assert evaluator._calculate_line_weight_with_diminishing_returns(["CENTRAL"]) == 40


def test_two_top_lines_reach_80_and_three_lines_saturate_a_close_station(evaluator):
    # Design note from the weight table: 2 major lines ~ 80 pts, 3 lines = full marks
    # once the (<= 5 min) distance coefficient of 1.0 and the 100 cap are applied.
    two = evaluator._calculate_line_weight_with_diminishing_returns(["central", "northern"])
    three = evaluator._calculate_line_weight_with_diminishing_returns(["central", "northern", "dlr"])
    coeff = evaluator._calculate_distance_coefficient(4)
    assert min(100, two * coeff) == 80
    assert min(100, three * coeff) == 100
