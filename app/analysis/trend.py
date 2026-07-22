"""Trend classification and gap analysis. Reuses `app.indicators` for the
underlying SMA/ADX math rather than recomputing it.
"""

from dataclasses import dataclass

import numpy as np

from app.indicators.moving_averages import sma
from app.indicators.trend import adx as compute_adx

_TREND_ADX_THRESHOLD = 20.0
_GAP_THRESHOLD_PERCENT = 0.5


def classify_trend(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, short_period: int = 20, long_period: int = 50
) -> str:
    """'Uptrend'/'Downtrend' when price and moving averages align with ADX
    confirming trend strength (>= 20); 'Sideways' otherwise.
    """
    if len(closes) < long_period:
        return "Insufficient Data"

    sma_short = sma(closes, short_period)
    sma_long = sma(closes, long_period)
    adx_values = compute_adx(highs, lows, closes, 14)

    last_close = closes[-1]
    last_short = sma_short[-1]
    last_long = sma_long[-1]
    last_adx = adx_values[-1]

    if np.isnan(last_short) or np.isnan(last_long):
        return "Insufficient Data"

    trending = not np.isnan(last_adx) and last_adx >= _TREND_ADX_THRESHOLD

    if last_close > last_short > last_long and trending:
        return "Uptrend"
    if last_close < last_short < last_long and trending:
        return "Downtrend"
    return "Sideways"


@dataclass(frozen=True, slots=True)
class GapInfo:
    gap_type: str  # "gap_up" | "gap_down" | "none"
    gap_percent: float


def analyze_gap(opens: np.ndarray, closes: np.ndarray) -> GapInfo:
    """Compares the latest bar's open against the previous bar's close."""
    if len(closes) < 2:
        return GapInfo(gap_type="none", gap_percent=0.0)

    prev_close = closes[-2]
    today_open = opens[-1]
    if prev_close == 0:
        return GapInfo(gap_type="none", gap_percent=0.0)

    gap_percent = (today_open - prev_close) / prev_close * 100
    if gap_percent >= _GAP_THRESHOLD_PERCENT:
        return GapInfo(gap_type="gap_up", gap_percent=round(float(gap_percent), 2))
    if gap_percent <= -_GAP_THRESHOLD_PERCENT:
        return GapInfo(gap_type="gap_down", gap_percent=round(float(gap_percent), 2))
    return GapInfo(gap_type="none", gap_percent=round(float(gap_percent), 2))
