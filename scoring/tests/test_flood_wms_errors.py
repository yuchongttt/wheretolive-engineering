"""NaFRA2 WMS failures must never be cached as "very low".

Before 2026-09-27, _query_nafra2_wms returned None both for a genuine "no
features were found" (point outside every risk band) and for 403 / 5xx /
timeouts / unexpected bodies, and evaluate_flood_risk cached any None as
very_low for 90 days. From 2026-07-22 the share of cache rows with no
rivers-&-sea and no surface-water band jumped from ~60% to ~93% and stayed
there — the EA endpoint started refusing much of our traffic, and every
refusal was stored as a clean bill of health that the mortgage panel and the
report read directly.

Response bodies below are real (captured 2026-09-27).
"""
import os
import sqlite3
import sys

import pytest
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import environment_evaluators as ee  # noqa: E402

NO_FEATURES = "no features were found\n"
HIGH = ("Results for FeatureType 'https://environment.data.gov.uk/spatialdata/"
        "dataset-96ab4342-82c1-4095-87f1-0082e8d84ef1/wfs:rofrs_4band':\n"
        "--------------------------------------------\nobjectid = 266388\n"
        "risk_band = High\nshape = [GEOMETRY (MultiPolygon) with 2499 points]\n"
        "--------------------------------------------\n")


class Resp:
    def __init__(self, status, text=""):
        self.status_code, self.text = status, text


@pytest.fixture
def env(tmp_path, monkeypatch):
    db = tmp_path / "evaluations.db"
    monkeypatch.setattr(ee, "_get_db_path", lambda: str(db))
    monkeypatch.setattr(ee, "record_api_usage", lambda *a, **k: None)
    monkeypatch.setattr(ee.time if hasattr(ee, "time") else __import__("time"), "sleep", lambda s: None)
    calls = []

    def serve(plan):
        """plan: {'rivers_sea': [resp|exc, ...], 'surface_water': [...]} consumed in order."""
        queues = {k: list(v) for k, v in plan.items()}

        def fake_get(url, params=None, timeout=None):
            src = "rivers_sea" if "rivers-and-sea" in url else "surface_water"
            calls.append(src)
            r = queues[src].pop(0)
            if isinstance(r, Exception):
                raise r
            return r
        monkeypatch.setattr(ee.requests, "get", fake_get)
    return {"db": db, "serve": serve, "calls": calls}


def cached_rows(db):
    if not db.exists():
        return []
    c = sqlite3.connect(db)
    try:
        return c.execute("SELECT data_json FROM flood_cache").fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        c.close()


def test_genuine_no_features_is_very_low_and_cached(env):
    env["serve"]({"rivers_sea": [Resp(200, NO_FEATURES)], "surface_water": [Resp(200, NO_FEATURES)]})
    r = ee.evaluate_flood_risk(51.56, -0.16, skip_cache=True)
    assert r["risk_level"] == "very_low" and r["score"] == 100
    assert len(cached_rows(env["db"])) == 1


def test_real_band_is_parsed_and_cached(env):
    env["serve"]({"rivers_sea": [Resp(200, HIGH)], "surface_water": [Resp(200, NO_FEATURES)]})
    r = ee.evaluate_flood_risk(51.431, -0.3266, skip_cache=True)
    assert r["risk_level"] == "high" and r["rivers_sea_risk"] == "High"
    assert len(cached_rows(env["db"])) == 1


@pytest.mark.parametrize("failure", [
    [Resp(403), Resp(403)],
    [Resp(503), Resp(500)],
    [requests.Timeout("t"), requests.ConnectionError("c")],
    [Resp(200, "<html>Service temporarily unavailable</html>"), Resp(200, "")],
])
def test_failed_source_returns_none_and_caches_nothing(env, failure):
    env["serve"]({"rivers_sea": failure, "surface_water": [Resp(200, NO_FEATURES)]})
    assert ee.evaluate_flood_risk(51.5, -0.1, skip_cache=True) is None
    assert cached_rows(env["db"]) == []


def test_one_retry_rescues_a_transient_failure(env):
    env["serve"]({"rivers_sea": [Resp(503), Resp(200, HIGH)], "surface_water": [Resp(200, NO_FEATURES)]})
    r = ee.evaluate_flood_risk(51.431, -0.3266, skip_cache=True)
    assert r["risk_level"] == "high"
    assert env["calls"].count("rivers_sea") == 2


def test_surface_water_failure_alone_is_also_unknown(env):
    # a missing half can't be combined into "the higher of the two"
    env["serve"]({"rivers_sea": [Resp(200, NO_FEATURES)], "surface_water": [Resp(403), Resp(403)]})
    assert ee.evaluate_flood_risk(51.5, -0.1, skip_cache=True) is None
    assert cached_rows(env["db"]) == []


def test_query_distinguishes_no_features_from_failure(env):
    env["serve"]({"rivers_sea": [Resp(200, NO_FEATURES), Resp(403), Resp(403)],
                  "surface_water": []})
    assert ee._query_nafra2_wms(51.5, -0.1, "rivers_sea") is None
    with pytest.raises(ee.NaFRA2Unavailable):
        ee._query_nafra2_wms(51.5, -0.1, "rivers_sea")
