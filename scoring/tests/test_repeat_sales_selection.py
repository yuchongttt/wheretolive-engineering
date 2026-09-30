# tests/test_repeat_sales_selection.py
"""Tests for the repeat-sales (historical resale transactions) selection logic.

Covers the three changes from 2026-07-08:
1. _select_repeat_sales adds a selection_reason (recent / long_held) to each item
2. When there are enough candidates, the same address appears at most once
3. Pairing uses residential transactions from before outlier filtering (genuine low-priced early sales no longer break pairs apart)
"""
from unittest.mock import MagicMock

import pytest

from price_analysis_evaluator import PriceAnalysisEvaluator


def _pair(addr, prev_date, prev_price, curr_date, curr_price, years_held):
    return {
        "address": addr,
        "previous_sale": {"date": prev_date, "price": prev_price},
        "current_sale": {"date": curr_date, "price": curr_price},
        "price_change": curr_price - prev_price,
        "price_change_pct": (curr_price - prev_price) / prev_price * 100,
        "years_held": years_held,
        "annualized_return": 5.0,
    }


def _txn(addr, date, price, ptype="T"):
    return {
        "address": addr,
        "postcode": "E1 1AA",
        "price": price,
        "date": date,
        "property_type": ptype,
        "tenure": "F",
        "new_build": False,
    }


class TestSelectRepeatSales:
    def test_empty(self):
        assert PriceAnalysisEvaluator._select_repeat_sales([]) == []

    def test_few_pairs_all_returned_newest_first_labeled_recent(self):
        pairs = [
            _pair("1 Foo St", "2010-01-01", 100_000, "2015-01-01", 150_000, 5.0),
            _pair("2 Foo St", "2012-01-01", 100_000, "2020-01-01", 180_000, 8.0),
        ]
        out = PriceAnalysisEvaluator._select_repeat_sales(pairs)
        assert [s["address"] for s in out] == ["2 Foo St", "1 Foo St"]
        assert all(s["selection_reason"] == "recent" for s in out)

    def test_small_set_still_dedups_including_whitespace_variants(self):
        # ≤5 items also go through the same dedup logic, consistent with what the tooltip promises (no behaviour cliff at the boundary);
        # the address key lower().strip() matches _identify_repeat_sales
        pairs = [
            _pair("9 Serial Sellers Rd", "2021-01-01", 100_000, "2025-06-01", 110_000, 4.4),
            _pair("9 Serial Sellers Rd", "2018-01-01", 90_000, "2021-01-01", 100_000, 3.0),
            _pair("9 Serial Sellers Rd ", "2015-01-01", 80_000, "2018-01-01", 90_000, 3.0),
            _pair("1 Vintage Villas", "1997-01-01", 60_000, "2024-01-01", 300_000, 27.0),
        ]
        out = PriceAnalysisEvaluator._select_repeat_sales(pairs)
        # The same address (incl. the trailing-whitespace variant) appears only once; both addresses are within the 3 most recent → both recent
        assert [(s["address"].strip(), s["selection_reason"]) for s in out] == [
            ("9 Serial Sellers Rd", "recent"),
            ("1 Vintage Villas", "recent"),
        ]

    def test_curated_three_recent_plus_two_long_held(self):
        pairs = [
            _pair("1 Recent Rd", "2020-01-01", 100_000, "2025-06-01", 120_000, 5.4),
            _pair("2 Recent Rd", "2019-01-01", 100_000, "2025-05-01", 120_000, 6.3),
            _pair("3 Recent Rd", "2018-01-01", 100_000, "2025-04-01", 120_000, 7.2),
            _pair("4 Mid Rd", "2015-01-01", 100_000, "2020-01-01", 120_000, 5.0),
            _pair("5 Long Ln", "1996-01-01", 50_000, "2024-01-01", 400_000, 28.0),
            _pair("6 Long Ln", "2000-01-01", 80_000, "2023-01-01", 350_000, 23.0),
        ]
        out = PriceAnalysisEvaluator._select_repeat_sales(pairs)
        assert len(out) == 5
        recent = [s for s in out if s["selection_reason"] == "recent"]
        long_held = [s for s in out if s["selection_reason"] == "long_held"]
        assert [s["address"] for s in recent] == ["1 Recent Rd", "2 Recent Rd", "3 Recent Rd"]
        assert [s["address"] for s in long_held] == ["5 Long Ln", "6 Long Ln"]

    def test_same_address_appears_at_most_once_when_curating(self):
        # One property sold 4 times → 3 pairs; they should not take up all the curated slots
        pairs = [
            _pair("9 Serial Sellers Rd", "2021-01-01", 100_000, "2025-06-01", 110_000, 4.4),
            _pair("9 Serial Sellers Rd", "2018-01-01", 90_000, "2021-01-01", 100_000, 3.0),
            _pair("9 Serial Sellers Rd", "2015-01-01", 80_000, "2018-01-01", 90_000, 3.0),
            _pair("1 Other St", "2019-01-01", 100_000, "2025-01-01", 130_000, 6.0),
            _pair("2 Other St", "2010-01-01", 100_000, "2024-01-01", 200_000, 14.0),
            _pair("3 Other St", "1998-01-01", 60_000, "2023-01-01", 300_000, 25.0),
        ]
        out = PriceAnalysisEvaluator._select_repeat_sales(pairs)
        addrs = [s["address"].lower() for s in out]
        assert addrs.count("9 serial sellers rd") == 1
        assert len(addrs) == len(set(addrs))

    def test_input_pairs_not_mutated(self):
        pairs = [_pair("1 Foo St", "2010-01-01", 100_000, "2015-01-01", 150_000, 5.0)]
        PriceAnalysisEvaluator._select_repeat_sales(pairs)
        assert "selection_reason" not in pairs[0]


class TestNominalTransferGuard:
    def test_nominal_transfer_never_pairs(self):
        """A £100 transfer between relatives + a market sale at the same address must not be paired — otherwise the fake +100%/yr pair
        would pollute repeat_sales_return_std, the trend CAGR and notable_gains."""
        txns = [
            _txn("3 Nominal Row", "2005-05-01", 100),
            _txn("3 Nominal Row", "2015-05-01", 400_000),
        ] + [
            _txn(f"{i} Modern Mews", f"20{15 + i % 10}-06-01", 400_000 + i * 1_000)
            for i in range(30)
        ]
        evaluator = PriceAnalysisEvaluator(params={})
        evaluator.epc_api = None
        evaluator.land_registry_api = MagicMock()
        evaluator.land_registry_api.get_transactions_by_postcode.return_value = txns

        result = evaluator.analyze_area_prices("E1 1AA", years=30, filter_outliers=True)

        assert not result.get("error")
        assert result["repeat_sales_count"] == 0
        assert all(rs["address"] != "3 Nominal Row" for rs in result["repeat_sales"])
        assert all(g["address"] != "3 Nominal Row" for g in result["notable_gains"])


class TestPairingBeforeOutlierFilter:
    def test_old_cheap_sale_still_pairs(self):
        """A genuine £45k sale in 1996 falls below the 17% floor of the 30-year mixed median and gets removed by outlier
        filtering — but it should still pair with the 2024 resale as a repeat-sale pair."""
        old_pair_txns = [
            _txn("7 Vintage Villas", "1996-03-01", 45_000),
            _txn("7 Vintage Villas", "2024-03-01", 480_000),
        ]
        # 30 recent high-priced sales push the mixed median up to £400k (17% floor = £68k > £45k)
        recent_txns = [
            _txn(f"{i} Modern Mews", f"20{15 + i % 10}-06-01", 400_000 + i * 1_000)
            for i in range(30)
        ]
        evaluator = PriceAnalysisEvaluator(params={})
        evaluator.epc_api = None
        evaluator.land_registry_api = MagicMock()
        evaluator.land_registry_api.get_transactions_by_postcode.return_value = (
            old_pair_txns + recent_txns
        )

        result = evaluator.analyze_area_prices("E1 1AA", years=30, filter_outliers=True)

        assert not result.get("error")
        paired_addrs = {rs["address"] for rs in result["repeat_sales"]}
        assert "7 Vintage Villas" in paired_addrs
        # Average-price stats are still based on the filtered transactions (the £45k sale is not in the summary sample dragging the average down)
        assert result["summary"]["min_price"] > 45_000
