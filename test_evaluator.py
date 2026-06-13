import pandas as pd
from evaluator import Evaluator, Signal


def make_df(close, sma20, sma50):
    """Build a minimal DataFrame whose last row carries the values under test."""
    return pd.DataFrame({
        "Close": [0.0, close],
        "SMA20": [None, sma20],
        "SMA50": [None, sma50],
    })


def test_buy_when_both_indicators_bullish():
    # price > sma20 AND sma20 > sma50
    sig = Evaluator().evaluate(make_df(close=110, sma20=105, sma50=100))
    assert sig.action == "BUY"
    assert sig.confidence == 0.80
    assert len(sig.reasons) == 2


def test_sell_when_both_indicators_bearish():
    # price < sma20 AND sma20 < sma50
    sig = Evaluator().evaluate(make_df(close=90, sma20=95, sma50=100))
    assert sig.action == "SELL"
    assert sig.confidence == 0.80


def test_hold_when_price_above_sma20_but_death_cross():
    # price > sma20 (bullish) but sma20 < sma50 (bearish) → mixed
    sig = Evaluator().evaluate(make_df(close=98, sma20=96, sma50=100))
    assert sig.action == "HOLD"
    assert sig.confidence == 0.50


def test_hold_when_price_below_sma20_but_golden_cross():
    # price < sma20 (bearish) but sma20 > sma50 (bullish) → mixed
    sig = Evaluator().evaluate(make_df(close=102, sma20=105, sma50=100))
    assert sig.action == "HOLD"


def test_reasons_are_human_readable():
    sig = Evaluator().evaluate(make_df(close=110, sma20=105, sma50=100))
    joined = " ".join(sig.reasons)
    assert "above SMA20" in joined
    assert "golden-cross" in joined


def test_ignores_leading_nan_rows():
    # Only the last fully-populated row should drive the decision
    df = pd.DataFrame({
        "Close": [50.0, 60.0, 110.0],
        "SMA20": [None, None, 105.0],
        "SMA50": [None, None, 100.0],
    })
    assert Evaluator().evaluate(df).action == "BUY"


def test_returns_signal_dataclass():
    sig = Evaluator().evaluate(make_df(close=110, sma20=105, sma50=100))
    assert isinstance(sig, Signal)
    assert 0.0 <= sig.confidence <= 1.0
