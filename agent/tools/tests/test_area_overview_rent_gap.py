"""get_area_overview: the rent/yield section must date its snapshot (2026-09-24).

The rent/yield figures come from the london_outcodes.json precompute, a
SNAPSHOT of asking rents stamped with generated_at — there is no delist lane
for rentals. The section header used to say "live rental listings", and in a
simulated landlord session about E14 the model repeated that verbatim ("based
on 644 live rental listings"). The header now says it is a snapshot and gives
its date; when the date is unknown it says so rather than going blank.

(Extract note: the original file also runs the tool end-to-end against a
fixture whose DDL is copied from the production database, covering the
"no rent data → say so and route to get_area_profile" branch; those cases need
the production DB and are not included here.)
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import get_area_overview as gao  # noqa: E402


def test_precompute_date_reads_generated_at(tmp_path, monkeypatch):
    p = tmp_path / "london_outcodes.json"
    p.write_text('{"version": 1, "generated_at": "2026-07-22T13:27:54.812160+00:00", "outcodes": []}')
    monkeypatch.setattr(gao, "OUTCODES_JSON", p)
    monkeypatch.setattr(gao, "_OUTCODE_METRICS", None)
    monkeypatch.setattr(gao, "_OUTCODE_GENERATED_AT", None)
    assert gao._precompute_date() == "2026-07-22"


def test_precompute_date_is_honest_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(gao, "OUTCODES_JSON", tmp_path / "nope.json")
    monkeypatch.setattr(gao, "_OUTCODE_METRICS", None)
    monkeypatch.setattr(gao, "_OUTCODE_GENERATED_AT", None)
    assert gao._precompute_date() == "unknown date"
