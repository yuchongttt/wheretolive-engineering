#!/usr/bin/env python3
"""
Station de-duplication and redundancy filtering (offline, synthetic stations).

A large interchange shows up as several "stations" (different entrances,
names, or the bus station next door). Two mechanisms keep the transit score
from counting one hub several times:

  * TfL path (used in production with --free):
      _dedupe_stations_tfl      merges entries whose names normalise to the same
                                station and unions their lines;
      _filter_redundant_stations keeps at most 3 stations, nearest first, and
                                drops a station whose lines are all already
                                served by a nearer one.
  * Google Places path: _deduplicate_stations treats two stations < 400 m apart
    as one hub and keeps the higher-scoring one (falls back to a name
    comparison when coordinates are missing).

The original version of this file ran a live evaluation for one postcode and
printed the result; it is replaced by these offline cases taken from the
design notes (303 m apart -> merged, ~500 m apart -> kept).
"""

import pytest

from transit_convenience_evaluator import TransitConvenienceEvaluator

M_PER_DEG_LAT = 111_195.0
BASE_LAT, BASE_LNG = 51.5308, -0.1238


def north_of_base(metres):
    return BASE_LAT + metres / M_PER_DEG_LAT, BASE_LNG


@pytest.fixture(scope="module")
def ev():
    return TransitConvenienceEvaluator()


def _st(name, score, metres_north, lines=("central",)):
    lat, lng = north_of_base(metres_north)
    return {"name": name, "score": score, "lat": lat, "lng": lng,
            "walk_time": 5.0, "lines": list(lines)}


def test_haversine_distance(ev):
    a, b = _st("A", 1, 0), _st("B", 1, 500)
    assert ev._calculate_station_distance(a, b) == pytest.approx(0.5, abs=0.002)


def test_same_hub_within_400m_is_merged_keeping_the_best(ev):
    main = _st("Hub Station", 100, 0)
    bus = _st("Bus Station", 60, 303)
    kept = ev._deduplicate_stations([bus, main])
    assert [s["name"] for s in kept] == ["Hub Station"]


def test_separate_station_500m_away_is_kept(ev):
    main = _st("Hub", 100, 0)
    intl = _st("Hub International", 90, 500)
    kept = ev._deduplicate_stations([main, intl])
    assert {s["name"] for s in kept} == {"Hub", "Hub International"}


def test_missing_coordinates_fall_back_to_core_name(ev):
    a = {"name": "Canonbury Station", "score": 50, "lat": None, "lng": None}
    b = {"name": "Canonbury", "score": 40, "lat": None, "lng": None}
    c = {"name": "Highbury", "score": 45, "lat": None, "lng": None}
    kept = ev._deduplicate_stations([a, b, c])
    assert [s["name"] for s in kept] == ["Canonbury Station", "Highbury"]


def test_tfl_name_dedupe_unions_lines_and_keeps_shortest_walk(ev):
    stations = [
        {"name": "Hub Underground Station", "walk_time": 6.0, "distance": 480,
         "lines": ["central", "jubilee"], "lat": 1, "lng": 1},
        {"name": "Hub DLR Station", "walk_time": 4.0, "distance": 320,
         "lines": ["dlr"], "lat": 1, "lng": 1},
        {"name": "Other Station", "walk_time": 9.0, "distance": 700,
         "lines": ["elizabeth"], "lat": 1, "lng": 1},
    ]
    merged = ev._dedupe_stations_tfl(stations)
    assert len(merged) == 2
    hub = merged[0]
    assert sorted(hub["lines"]) == ["central", "dlr", "jubilee"]
    assert hub["walk_time"] == 4.0          # sorted nearest-first, shortest walk kept


def test_redundant_stations_are_dropped_and_capped_at_three(ev):
    stations = [  # already sorted nearest-first
        {"name": "A", "lines": ["central", "jubilee"]},
        {"name": "B", "lines": ["central"]},            # fully covered by A -> dropped
        {"name": "C", "lines": ["jubilee", "dlr"]},     # adds dlr -> kept
        {"name": "D", "lines": ["elizabeth"]},          # adds elizabeth -> kept (3rd)
        {"name": "E", "lines": ["northern"]},           # would add, but cap is 3
    ]
    kept = ev._filter_redundant_stations(stations, max_stations=3)
    assert [s["name"] for s in kept] == ["A", "C", "D"]
