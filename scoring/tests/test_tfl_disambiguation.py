#!/usr/bin/env python3
"""Regression test: TfL get_journey must auto-resolve HTTP 300 disambiguation.

Root cause (2026-05-29): TfL's /Journey/JourneyResults frequently returns
HTTP 300 for a postcode/landmark that maps to several candidate stops. The
old code did `return None` on 300, so get_commute reported "no route" and the
chat agent had to retry with a station name — wasting 5+ extra tool calls per
multi-candidate query. The fix: read the 300 body's disambiguationOptions,
pick the highest matchQuality parameterValue per endpoint, and retry once.

Run:  .venv/bin/python3 tests/test_tfl_disambiguation.py
"""

import sys
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import apis.tfl as tfl_mod
from apis.tfl import TfLAPI


class FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


DISAMBIG_300 = {
    "$type": "Tfl.Api.Presentation.Entities.JourneyPlanner.DisambiguationResult",
    # to-end is ambiguous: two candidate stops, best matchQuality first-ish
    "toLocationDisambiguation": {
        "matchStatus": "list",
        "disambiguationOptions": [
            {"parameterValue": "1000-LOWQ", "matchQuality": 500,
             "place": {"commonName": "Charlton (wrong)"}},
            {"parameterValue": "1000-BEST", "matchQuality": 900,
             "place": {"commonName": "Charlton"}},
        ],
    },
    # from-end resolved cleanly — must be left untouched on retry
    "fromLocationDisambiguation": {"matchStatus": "identified"},
}

JOURNEY_200 = {
    "journeys": [
        {
            "duration": 37,
            "legs": [
                {
                    "mode": {"name": "elizabeth-line"},
                    "duration": 30,
                    "departurePoint": {"commonName": "Charlton"},
                    "arrivalPoint": {"commonName": "Farringdon"},
                    "routeOptions": [{"name": "Elizabeth line"}],
                }
            ],
            "fare": {"totalCost": 480, "fares": []},
        }
    ]
}


def test_disambiguation_auto_resolves():
    """300 on first call → resolve best parameterValue → 200 on retry."""
    calls = []

    # headers= is required: apis/tfl.py sends an identifying User-Agent
    # header. Without accepting it, the fake raised
    # TypeError, which get_journey's broad `except` swallowed into a silent None —
    # so the failure read as "disambiguation was not resolved".
    def fake_get(url, params=None, timeout=None, headers=None):
        calls.append(url)
        # First call (raw postcode) → 300; subsequent (resolved) → 200
        if len(calls) == 1:
            return FakeResp(300, DISAMBIG_300)
        return FakeResp(200, JOURNEY_200)

    with mock.patch.object(tfl_mod, "requests") as m_requests, \
         mock.patch.object(tfl_mod, "record_api_usage", lambda *a, **k: None):
        m_requests.get.side_effect = fake_get
        result = TfLAPI().get_journey("SE10 0DX", "Farringdon")

    assert result is not None, "300 disambiguation was not resolved (got None)"
    assert result["duration_minutes"] == 37, f"bad duration: {result}"
    assert len(calls) == 2, f"expected exactly 2 HTTP calls (300 then retry), got {len(calls)}: {calls}"
    # Retry must use the BEST (matchQuality 900) parameterValue for the to-end,
    # and keep the identified from-end as the original string.
    retry_url = calls[1]
    assert "1000-BEST" in retry_url, f"retry did not use best parameterValue: {retry_url}"
    assert "1000-LOWQ" not in retry_url, f"retry used the low-quality option: {retry_url}"
    assert "SE10 0DX" in retry_url, f"identified from-end should be unchanged: {retry_url}"
    print("PASS test_disambiguation_auto_resolves")


def test_unresolvable_disambiguation_returns_none():
    """300 with an empty option list (no match) → give up gracefully."""
    empty_300 = {
        "toLocationDisambiguation": {"matchStatus": "empty", "disambiguationOptions": []},
        "fromLocationDisambiguation": {"matchStatus": "identified"},
    }

    # headers= is required: apis/tfl.py sends an identifying User-Agent
    # header. Without accepting it, the fake raised
    # TypeError, which get_journey's broad `except` swallowed into a silent None —
    # so the failure read as "disambiguation was not resolved".
    def fake_get(url, params=None, timeout=None, headers=None):
        return FakeResp(300, empty_300)

    with mock.patch.object(tfl_mod, "requests") as m_requests, \
         mock.patch.object(tfl_mod, "record_api_usage", lambda *a, **k: None):
        m_requests.get.side_effect = fake_get
        result = TfLAPI().get_journey("SE10 0DX", "Nowheresville")

    assert result is None, f"unresolvable 300 should return None, got {result}"
    print("PASS test_unresolvable_disambiguation_returns_none")


if __name__ == "__main__":
    test_disambiguation_auto_resolves()
    test_unresolvable_disambiguation_returns_none()
    print("\nAll TfL disambiguation tests passed.")
