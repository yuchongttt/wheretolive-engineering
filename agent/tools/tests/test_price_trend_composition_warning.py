"""get_price_trend: warn when the median YoY and the repeat-sale CAGR diverge sharply.

R2-3 textbook case: EC2Y flats 12m median YoY +13.3% (n=70, sensitive to the
mix of what sold) vs repeat-sale 3Y CAGR +4.6%/yr (175 pairs, same-home
comparisons, immune to mix drift). The agent used +13.3% to say "the market
isn't falling"; the reference agent used a modelled index to say "-14.8% over
ten years" — neither played the strongest card. Beyond the threshold, say it
plainly: judge direction by the CAGR; the median is only a price level.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import get_price_trend as gpt  # noqa: E402


def test_warns_on_sharp_divergence():
    w = gpt._composition_warning(13.3, 4.6)
    assert w is not None
    assert "composition" in w.lower() or "结构" in w
    assert "CAGR" in w


def test_silent_when_aligned():
    assert gpt._composition_warning(5.0, 4.6) is None


def test_silent_when_either_side_missing():
    assert gpt._composition_warning(None, 4.6) is None
    assert gpt._composition_warning(13.3, None) is None


def test_negative_divergence_also_warns():
    """A median plunge with a steady CAGR must be caught too — both directions can mislead."""
    assert gpt._composition_warning(-9.0, 2.0) is not None
