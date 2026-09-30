#!/usr/bin/env python3
"""
Square query polygon for the data.police.uk `poly=` parameter.

UKPoliceAPI._calculate_square_polygon builds an N x N km square centred on a
point (1 deg latitude ~ 111 km, longitude scaled by cos(lat)). Check the
edges and diagonal against a haversine distance: mean edge error must stay
under 10 m. (The original file also had a live API call; that part was dropped
because the suite must run offline.)
"""

import math

import pytest

from api_helper import UKPoliceAPI


def haversine_km(lat1, lng1, lat2, lng2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlng / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# King's Cross
LAT, LNG = 51.534749, -0.124585


@pytest.mark.parametrize("side_km", [0.5, 1.0, 2.0])
def test_square_polygon_geometry(side_km):
    poly = UKPoliceAPI()._calculate_square_polygon(LAT, LNG, side_km)
    vertices = [tuple(map(float, v.split(","))) for v in poly.split(":")]
    assert len(vertices) == 4  # NW, NE, SE, SW

    edges = [haversine_km(*vertices[i], *vertices[(i + 1) % 4]) for i in range(4)]
    mean_error_km = sum(abs(e - side_km) for e in edges) / 4
    assert mean_error_km < 0.010, f"mean edge error {mean_error_km * 1000:.2f} m"

    diagonal = haversine_km(*vertices[0], *vertices[2])
    assert diagonal == pytest.approx(side_km * math.sqrt(2), abs=0.015)

    # the square is centred on the query point
    assert sum(v[0] for v in vertices) / 4 == pytest.approx(LAT, abs=1e-6)
    assert sum(v[1] for v in vertices) / 4 == pytest.approx(LNG, abs=1e-6)
