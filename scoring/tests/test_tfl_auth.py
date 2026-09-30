"""TfL app_key / UA wiring — network-free unit tests.

The registered app_key (free "500 Requests per min" product) raises the rate
limit from ~50 to 500 req/min. It must reach every outgoing request's params,
sourced from the TFL_APP_KEY env var (.env) unless passed explicitly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apis.tfl import TfLAPI


def test_app_key_from_env_merges_into_params(monkeypatch):
    monkeypatch.setenv("TFL_APP_KEY", "testkey123")
    api = TfLAPI()
    assert api._params({"query": "Bank"}) == {"query": "Bank", "app_key": "testkey123"}


def test_explicit_app_key_beats_env(monkeypatch):
    monkeypatch.setenv("TFL_APP_KEY", "envkey")
    assert TfLAPI(app_key="explicit")._params()["app_key"] == "explicit"


def test_no_key_means_no_app_key_param(monkeypatch):
    monkeypatch.delenv("TFL_APP_KEY", raising=False)
    assert "app_key" not in TfLAPI(app_key=None)._params({"query": "x"})


def test_headers_set_identifying_ua():
    # The default python-requests UA is rejected at the edge; every request
    # carries a UA that names the project and a contact address.
    ua = TfLAPI()._headers().get("User-Agent", "")
    assert "wheretolive.xyz" in ua and "python-requests" not in ua
