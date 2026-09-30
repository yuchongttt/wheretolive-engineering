"""Offline checks for the valuation experiment scripts.

Added for this extract: the source repo has no tests for these scripts. Everything here runs on small
synthetic frames -- no database, no GPU box, no network. The model is replaced by a spy where only the
evaluation harness is under test, so nothing is trained.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
pytest.importorskip("sklearn")

SCRIPTS = ["valuation_knn_v1_2", "valuation_v4_gbr", "valuation_v6_gbr", "valuation_v7_explore",
           "valuation_v8_temporal", "valuation_v8_analysis", "valuation_v8_best_detail", "predict_one_v8"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_imports(name):
    importlib.import_module(name)


def _synthetic_pool(n=2400, seed=0):
    """Rows shaped like load_pool() output: every column any feature group reads, random values."""
    import valuation_v7_explore as v7
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2006-01-01") + pd.to_timedelta(rng.integers(0, 17 * 365, n), unit="D")
    df = pd.DataFrame({
        "rm_uuid": [f"p{i % (n // 2)}" for i in range(n)],        # two sales per property
        "sold_date": dates,
        "sold_price": rng.uniform(2e5, 9e5, n),
        "tenure": rng.choice(["Leasehold", "Freehold"], n),
        "type_bucket": rng.choice(["flat", "terraced", "semi"], n),
    })
    df["year_sold"] = df["sold_date"].dt.year
    cols = set(v7.V5_BASE_COLS) | set(v7.V5_LUX_COLS)
    for group_cols in v7.GROUPS.values():
        cols |= set(group_cols)
    for c in sorted(cols - set(df.columns)):
        df[c] = rng.normal(size=n)
    return df


def test_v8_removes_the_leaky_growth_feature():
    import valuation_v7_explore as v7
    import valuation_v8_temporal as v8
    assert v7.GROUPS["prev_cagr"] == ["prev_annualised_growth"]    # v7: uses the row's own sold_price
    assert v8.GROUPS["prev_cagr"] == []
    df = _synthetic_pool(200)
    enabled = v7.V6_BEST | {"prev_cagr", "area_per_bed"}
    assert "prev_annualised_growth" in v7.make_features(df, enabled).columns
    assert "prev_annualised_growth" not in v8.make_features(df, enabled).columns


class _Spy:
    """Stands in for the GBR: records which rows it was fitted and scored on."""
    calls = []

    def fit(self, X, y):
        _Spy.calls.append(("fit", X.index))
        return self

    def predict(self, X):
        _Spy.calls.append(("predict", X.index))
        return np.full(len(X), 13.0)


def test_temporal_walk_forward_never_trains_on_the_test_year_or_later(monkeypatch):
    import valuation_v7_explore as v7
    import valuation_v8_temporal as v8
    df = _synthetic_pool()
    X = v8.make_features(df, v7.V6_BEST)
    _Spy.calls = []
    monkeypatch.setattr(v8, "gbr_default", lambda: _Spy())
    res = v8.evaluate_temporal(X, np.log(df["sold_price"].values), df, "spy")
    fits = [idx for kind, idx in _Spy.calls if kind == "fit"]
    assert len(fits) == len(res["per_period"]) > 0
    for idx, period in zip(fits, res["per_period"]):
        assert df.loc[idx, "sold_date"].max() < pd.Timestamp(f"{period['year']}-01-01")


def test_group_kfold_keeps_every_sale_of_a_property_in_one_fold(monkeypatch):
    import valuation_v7_explore as v7
    import valuation_v8_temporal as v8
    df = _synthetic_pool()
    X = v8.make_features(df, v7.V6_BEST)
    _Spy.calls = []
    monkeypatch.setattr(v8, "gbr_default", lambda: _Spy())
    v8.evaluate_groupkfold(X, np.log(df["sold_price"].values), df, "spy")
    folds = list(zip(_Spy.calls[0::2], _Spy.calls[1::2]))
    assert len(folds) == 5
    for (k1, train_idx), (k2, test_idx) in folds:
        assert (k1, k2) == ("fit", "predict")
        assert not set(df.loc[train_idx, "rm_uuid"]) & set(df.loc[test_idx, "rm_uuid"])
        assert (df.loc[test_idx, "sold_date"] >= "2020-01-01").all()


def test_knn_price_feature_only_looks_at_earlier_sales():
    import valuation_v7_explore as v7
    n = 12
    df = pd.DataFrame({
        "latitude": 51.5 + np.arange(n) * 1e-4, "longitude": np.full(n, -0.1),
        "bedrooms": np.full(n, 2), "type_bucket": ["flat"] * n,
        "sold_date": pd.to_datetime("2020-01-01") + pd.to_timedelta(np.arange(n) * 30, unit="D"),
        "ppsf": np.linspace(5000, 6100, n),
    })
    base = v7.compute_knn_price(df)
    k = 6
    changed = df.copy()
    changed.loc[k, "ppsf"] = 1.0                                   # perturb one mid-pool sale
    after = v7.compute_knn_price(changed)
    # Rows 1..k have at least one earlier comparable and must not see sale k's own price.
    # Row 0 has none and is filled with the pool-wide median of the other rows' values, which can
    # depend on later sales -- a small known leak path, so it is deliberately not asserted here.
    np.testing.assert_array_equal(base[1:k + 1], after[1:k + 1])
    assert after[k + 1] != base[k + 1]                             # later rows do see it
