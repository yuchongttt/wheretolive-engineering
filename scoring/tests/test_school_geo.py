import math

from school_geo import bng_to_latlon, classify_category


def test_bng_to_latlon_central_london():
    # Trafalgar Square ~ Easting 530047, Northing 180550 -> ~ (51.508, -0.128)
    lat, lon = bng_to_latlon(530047, 180550)
    assert math.isclose(lat, 51.508, abs_tol=0.02), lat
    assert math.isclose(lon, -0.128, abs_tol=0.02), lon


def test_classify_category():
    assert classify_category("Community school") == "state"
    assert classify_category("Other independent school") == "independent"
    assert classify_category("Community special school") == "special"


def test_evaluator_delegates_to_shared():
    # The evaluator's methods must produce identical results post-refactor.
    from school_evaluator import SchoolEvaluator

    ev = SchoolEvaluator.__new__(SchoolEvaluator)  # no CSV load
    assert ev._bng_to_latlon(530047, 180550) == bng_to_latlon(530047, 180550)
    assert SchoolEvaluator.classify_category("Other independent school") == "independent"
