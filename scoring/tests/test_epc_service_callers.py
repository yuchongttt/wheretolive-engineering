"""Callers of the local EPC service (EPC data-source migration, 2026-09-27).

1. /epc/postcode without `limit` returns only 50 rows; in a large block (hundreds of
   flats under one postcode) the target flat may not be in the response at all, so the
   unit match fails or (when matching on a unique street name) picks a neighbour.
   LocalEPCService must request the service maximum of 200.
2. price_analysis_evaluator no longer falls back to epc.opendatacommunities.org
   (retired on 2026-05-30; on failure it also retried with waits).

(A third case covering a listing-enrichment script outside this module was dropped.)
"""
import sys
from unittest.mock import MagicMock

from api_helper import LocalEPCService  # noqa: E402
from price_analysis_evaluator import PriceAnalysisEvaluator  # noqa: E402


def test_local_epc_service_asks_for_the_service_maximum(monkeypatch):
    resp = MagicMock(status_code=200)
    resp.json.return_value = []
    get = MagicMock(return_value=resp)
    monkeypatch.setattr(sys.modules[LocalEPCService.__module__].requests, "get", get)
    LocalEPCService("http://epc.local:8400").search_by_postcode("E14 4AB")
    assert get.call_args.kwargs["params"]["limit"] == 200


def test_price_analysis_has_no_retired_api_fallback():
    ev = PriceAnalysisEvaluator(params={"epc": {"email": "x@example.com", "api_key": "k"}})
    assert getattr(ev, "epc_api", None) is None
