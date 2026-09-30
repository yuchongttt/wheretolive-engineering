"""get_sold_nearby._is_market judges pre-2005 purchases against the market of their time (2026-09-27).

The "non-market" rules (purchase < £50k, doubling within 8 years, annualised
> +20%) are written at today's price levels. London's median sale price was
about £73k in 1995 and rose 2.3x from 1996 to 2002 — measured over all 2.08M
London repeat-sale pairs, 40-50% of 1995-1999 purchases were dropped as
"non-market", and the rescued portion ran only 1-3 percentage points a year
above the market over the same period, i.e. ordinary sales. So for pre-2005
purchases: a pair the original rules drop still counts as a market sale if the
purchase price is at least 30% of that year's London median (and never above
the £50k floor), the annualised return is no more than 12 points above the
market's over the same years, and the fall is no worse than -35%/yr. This only
ever loosens; after 2005 nothing changes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import get_sold_nearby as g  # noqa: E402


def _pair(first_date, first_price, last_date, last_price):
    from datetime import date
    d1, d2 = date.fromisoformat(first_date), date.fromisoformat(last_date)
    hy = (d2 - d1).days / 365.25
    return {"first_date": first_date, "last_date": last_date, "first_price": first_price,
            "last_price": last_price, "hold_years": hy,
            "pct_change": (last_price / first_price - 1) * 100}


def test_1990s_ex_council_flat_at_the_market_price_counts():
    # an ex-council flat in E2: 1998 £44k → 2018 £305k (+10.4%/yr; market over the same years ≈ +8%/yr)
    assert g._is_market(_pair("1998-10-23", 44_000, "2018-05-10", 305_000))


def test_1990s_boom_resale_counts():
    # 1997 £90k → 2002 £190k: doubled in 5 years (dropped by the old rule); market went 86k→180k
    assert g._is_market(_pair("1997-03-01", 90_000, "2002-03-01", 190_000))


def test_1990s_discounted_first_sale_is_still_non_market():
    # 1996 £15k (under 30% of that year's median) — a discounted / non-market purchase
    assert not g._is_market(_pair("1996-07-12", 15_000, "2002-08-14", 58_000))


def test_1990s_staircasing_pattern_is_still_non_market():
    # 1999 £40k share → 2001 full price £160k: +100%/yr, far above the market
    assert not g._is_market(_pair("1999-06-01", 40_000, "2001-06-01", 160_000))


def test_rules_after_2005_are_unchanged():
    # 2006 £45k: still dropped by the £50k floor; 2010 → 2016 doubling (6 years) still dropped
    assert not g._is_market(_pair("2006-01-01", 45_000, "2012-01-01", 60_000))
    assert not g._is_market(_pair("2010-01-01", 200_000, "2016-01-01", 420_000))
    assert g._is_market(_pair("2010-01-01", 300_000, "2020-01-01", 450_000))


def test_never_stricter_than_before():
    # a 2000 purchase the old rules accepted (steady hold) is still accepted
    assert g._is_market(_pair("2000-05-01", 150_000, "2015-05-01", 450_000))
