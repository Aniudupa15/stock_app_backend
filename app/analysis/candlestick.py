"""Candlestick pattern detection. Pure numpy, shape-based (no trend-context
requirement) - each function returns a boolean array, True at bar i if the
pattern completes at that bar. Reusable by any feature needing pattern
recognition (Phase 3's intraday assistant, later chart annotations, etc.).
"""

import numpy as np

_DOJI_BODY_RATIO = 0.1
_HAMMER_MAX_BODY_RATIO = 0.3
_HAMMER_MIN_SHADOW_RATIO = 0.6
_HAMMER_MAX_OPPOSITE_SHADOW_RATIO = 0.1


def _body(opens: np.ndarray, closes: np.ndarray) -> np.ndarray:
    return np.abs(closes - opens)


def _range(highs: np.ndarray, lows: np.ndarray) -> np.ndarray:
    return highs - lows


def is_doji(opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    rng = _range(highs, lows)
    body = _body(opens, closes)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(rng > 0, body / rng, 0.0)
    return ratio <= _DOJI_BODY_RATIO


def is_hammer(opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    """Small body near the top of the range, long lower shadow, little/no upper shadow.

    Thresholds are relative to the bar's total range (high-low), not the body -
    a body-relative threshold breaks down when the body is ~0 (open == close),
    since the threshold itself then shrinks to ~0 and rejects any shadow at all.
    """
    body = _body(opens, closes)
    rng = _range(highs, lows)
    upper_shadow = highs - np.maximum(opens, closes)
    lower_shadow = np.minimum(opens, closes) - lows
    with np.errstate(divide="ignore", invalid="ignore"):
        body_ratio = np.where(rng > 0, body / rng, 0.0)
        lower_ratio = np.where(rng > 0, lower_shadow / rng, 0.0)
        upper_ratio = np.where(rng > 0, upper_shadow / rng, 0.0)
    return (
        (rng > 0)
        & (body_ratio <= _HAMMER_MAX_BODY_RATIO)
        & (lower_ratio >= _HAMMER_MIN_SHADOW_RATIO)
        & (upper_ratio <= _HAMMER_MAX_OPPOSITE_SHADOW_RATIO)
    )


def is_shooting_star(opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    """Mirror of the hammer: small body near the bottom of the range, long upper shadow."""
    body = _body(opens, closes)
    rng = _range(highs, lows)
    upper_shadow = highs - np.maximum(opens, closes)
    lower_shadow = np.minimum(opens, closes) - lows
    with np.errstate(divide="ignore", invalid="ignore"):
        body_ratio = np.where(rng > 0, body / rng, 0.0)
        upper_ratio = np.where(rng > 0, upper_shadow / rng, 0.0)
        lower_ratio = np.where(rng > 0, lower_shadow / rng, 0.0)
    return (
        (rng > 0)
        & (body_ratio <= _HAMMER_MAX_BODY_RATIO)
        & (upper_ratio >= _HAMMER_MIN_SHADOW_RATIO)
        & (lower_ratio <= _HAMMER_MAX_OPPOSITE_SHADOW_RATIO)
    )


def is_bullish_engulfing(opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    """Bar i-1 bearish, bar i bullish, bar i's body fully engulfs bar i-1's body."""
    n = len(closes)
    result = np.zeros(n, dtype=bool)
    for i in range(1, n):
        prev_bearish = closes[i - 1] < opens[i - 1]
        curr_bullish = closes[i] > opens[i]
        engulfs = opens[i] < closes[i - 1] and closes[i] > opens[i - 1]
        result[i] = prev_bearish and curr_bullish and engulfs
    return result


def is_bearish_engulfing(opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    """Bar i-1 bullish, bar i bearish, bar i's body fully engulfs bar i-1's body."""
    n = len(closes)
    result = np.zeros(n, dtype=bool)
    for i in range(1, n):
        prev_bullish = closes[i - 1] > opens[i - 1]
        curr_bearish = closes[i] < opens[i]
        engulfs = opens[i] > closes[i - 1] and closes[i] < opens[i - 1]
        result[i] = prev_bullish and curr_bearish and engulfs
    return result


def is_morning_star(opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    """3-bar bullish reversal: long bearish bar, small-bodied "star" that gaps
    down, then a bullish bar closing back above the midpoint of the first bar.
    """
    n = len(closes)
    result = np.zeros(n, dtype=bool)
    body = _body(opens, closes)
    rng = _range(highs, lows)
    for i in range(2, n):
        first_bearish = closes[i - 2] < opens[i - 2]
        first_long = rng[i - 2] > 0 and body[i - 2] / rng[i - 2] > 0.5
        star_small = rng[i - 1] > 0 and body[i - 1] / rng[i - 1] < _DOJI_BODY_RATIO * 3
        star_gaps_down = max(opens[i - 1], closes[i - 1]) < closes[i - 2]
        third_bullish = closes[i] > opens[i]
        first_midpoint = (opens[i - 2] + closes[i - 2]) / 2
        closes_above_midpoint = closes[i] > first_midpoint
        result[i] = first_bearish and first_long and star_small and star_gaps_down and third_bullish and closes_above_midpoint
    return result


def detect_latest_patterns(opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> list[str]:
    """Convenience aggregator: which patterns (if any) complete on the most recent bar."""
    if len(closes) == 0:
        return []

    detectors = {
        "Doji": is_doji,
        "Hammer": is_hammer,
        "Shooting Star": is_shooting_star,
        "Bullish Engulfing": is_bullish_engulfing,
        "Bearish Engulfing": is_bearish_engulfing,
        "Morning Star": is_morning_star,
    }
    found = []
    for name, fn in detectors.items():
        result = fn(opens, highs, lows, closes)
        if result[-1]:
            found.append(name)
    return found
