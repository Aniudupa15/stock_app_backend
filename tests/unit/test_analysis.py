import numpy as np
import pytest

from app.analysis.candlestick import (
    is_bearish_engulfing,
    is_bullish_engulfing,
    is_doji,
    is_hammer,
    is_morning_star,
    is_shooting_star,
)
from app.analysis.levels import detect_breakout_or_breakdown, find_swing_highs, find_swing_lows, support_resistance_levels
from app.analysis.trend import analyze_gap, classify_trend


def test_bullish_engulfing_detects_classic_pattern():
    opens = np.array([10.0, 7.0])
    highs = np.array([10.5, 11.5])
    lows = np.array([7.5, 6.5])
    closes = np.array([8.0, 11.0])
    result = is_bullish_engulfing(opens, highs, lows, closes)
    assert result.tolist() == [False, True]


def test_bearish_engulfing_detects_classic_pattern():
    opens = np.array([8.0, 11.0])
    highs = np.array([10.5, 11.5])
    lows = np.array([7.5, 6.5])
    closes = np.array([10.0, 7.0])
    result = is_bearish_engulfing(opens, highs, lows, closes)
    assert result.tolist() == [False, True]


def test_doji_detects_tiny_body():
    opens = np.array([10.0])
    highs = np.array([11.0])
    lows = np.array([9.0])
    closes = np.array([10.02])
    assert is_doji(opens, highs, lows, closes)[0]


def test_hammer_handles_zero_body_without_crashing():
    """Regression test: a zero-body hammer (open == close) used to be rejected
    because the old thresholds were relative to body size, which shrinks to
    ~0 along with the body itself. Thresholds are now range-relative.
    """
    opens = np.array([10.0])
    highs = np.array([10.2])
    lows = np.array([7.0])
    closes = np.array([10.0])
    assert is_hammer(opens, highs, lows, closes)[0]


def test_hammer_rejects_long_upper_shadow():
    opens = np.array([9.5])
    highs = np.array([11.0])
    lows = np.array([8.0])
    closes = np.array([9.6])
    assert not is_hammer(opens, highs, lows, closes)[0]


def test_shooting_star_handles_zero_body_without_crashing():
    opens = np.array([8.0])
    highs = np.array([11.0])
    lows = np.array([7.9])
    closes = np.array([8.0])
    assert is_shooting_star(opens, highs, lows, closes)[0]


def test_morning_star_requires_gap_down_star():
    opens = np.array([20.0, 15.0, 15.3])
    highs = np.array([20.2, 15.3, 19.5])
    lows = np.array([15.0, 14.9, 15.2])
    closes = np.array([15.2, 15.1, 19.0])
    assert is_morning_star(opens, highs, lows, closes)[2]


def test_morning_star_false_when_star_does_not_gap_down():
    opens = np.array([20.0, 15.5, 16.0])
    highs = np.array([20.2, 15.8, 19.5])
    lows = np.array([15.0, 15.2, 15.9])
    closes = np.array([15.2, 15.6, 19.0])
    assert not is_morning_star(opens, highs, lows, closes)[2]


def test_find_swing_highs_and_lows():
    highs = np.array([10, 11, 12, 15, 12, 11, 10, 9, 8, 9, 10, 13, 10, 9], dtype=float)
    lows = np.array([9, 10, 11, 13, 11, 10, 9, 8, 7, 8, 9, 11, 9, 8], dtype=float)
    swing_highs = find_swing_highs(highs, window=2)
    swing_lows = find_swing_lows(lows, window=2)
    assert np.where(swing_highs)[0].tolist() == [3, 11]
    assert 8 in np.where(swing_lows)[0].tolist()


def test_detect_breakout_with_volume_confirmation():
    highs = np.array([10, 11, 12, 15, 12, 11, 10, 9, 8, 9, 10, 13, 10, 9], dtype=float)
    lows = np.array([9, 10, 11, 13, 11, 10, 9, 8, 7, 8, 9, 11, 9, 8], dtype=float)
    levels = support_resistance_levels(highs, lows, window=2, lookback=20)

    closes = np.array([9, 9.5, 10, 10.2, 10.5, 10.8, 11, 11.5, 15.5])
    volumes = np.array([1000] * 8 + [5000])
    assert detect_breakout_or_breakdown(closes, volumes, levels, volume_multiplier=1.5, lookback_avg=8) == "breakout"


def test_detect_breakout_none_without_volume_confirmation():
    highs = np.array([10, 11, 12, 15, 12, 11, 10, 9, 8, 9, 10, 13, 10, 9], dtype=float)
    lows = np.array([9, 10, 11, 13, 11, 10, 9, 8, 7, 8, 9, 11, 9, 8], dtype=float)
    levels = support_resistance_levels(highs, lows, window=2, lookback=20)

    closes = np.array([9, 9.5, 10, 10.2, 10.5, 10.8, 11, 11.5, 15.5])
    volumes = np.array([1000] * 9)  # no volume spike on the breakout bar
    assert detect_breakout_or_breakdown(closes, volumes, levels, volume_multiplier=1.5, lookback_avg=8) is None


def test_classify_trend_uptrend_on_strongly_rising_series():
    n = 60
    closes = 100 + np.arange(n, dtype=float) * 2
    highs = closes + 1
    lows = closes - 1
    assert classify_trend(highs, lows, closes) == "Uptrend"


def test_classify_trend_insufficient_data():
    closes = np.array([100.0, 101.0, 102.0])
    highs = closes + 1
    lows = closes - 1
    assert classify_trend(highs, lows, closes) == "Insufficient Data"


def test_analyze_gap_up():
    opens = np.array([100.0, 108.0])
    closes = np.array([100.0, 108.0])
    gap = analyze_gap(opens, closes)
    assert gap.gap_type == "gap_up"
    assert gap.gap_percent == pytest.approx(8.0)


def test_analyze_gap_none_for_small_move():
    opens = np.array([100.0, 100.2])
    closes = np.array([100.0, 100.2])
    gap = analyze_gap(opens, closes)
    assert gap.gap_type == "none"
