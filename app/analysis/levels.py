"""Support/resistance level detection and breakout/breakdown detection.
Pure numpy, operates on OHLCV arrays already loaded by the caller.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Level:
    price: float
    kind: str  # "support" | "resistance"


def find_swing_highs(highs: np.ndarray, window: int = 3) -> np.ndarray:
    """Boolean array, True where bar i's high is the max within [i-window, i+window]."""
    n = len(highs)
    result = np.zeros(n, dtype=bool)
    for i in range(window, n - window):
        result[i] = highs[i] == np.max(highs[i - window : i + window + 1])
    return result


def find_swing_lows(lows: np.ndarray, window: int = 3) -> np.ndarray:
    """Boolean array, True where bar i's low is the min within [i-window, i+window]."""
    n = len(lows)
    result = np.zeros(n, dtype=bool)
    for i in range(window, n - window):
        result[i] = lows[i] == np.min(lows[i - window : i + window + 1])
    return result


def support_resistance_levels(highs: np.ndarray, lows: np.ndarray, window: int = 3, lookback: int = 60) -> list[Level]:
    """Candidate support/resistance levels from swing points in the most recent `lookback` bars."""
    n = len(highs)
    start = max(0, n - lookback)
    swing_high_mask = find_swing_highs(highs, window)
    swing_low_mask = find_swing_lows(lows, window)

    levels: list[Level] = []
    for i in range(start, n):
        if swing_high_mask[i]:
            levels.append(Level(price=float(highs[i]), kind="resistance"))
        if swing_low_mask[i]:
            levels.append(Level(price=float(lows[i]), kind="support"))
    return levels


def detect_breakout_or_breakdown(
    closes: np.ndarray,
    volumes: np.ndarray,
    levels: list[Level],
    volume_multiplier: float = 1.5,
    lookback_avg: int = 20,
) -> str | None:
    """'breakout' if the latest close clears the nearest resistance level above the
    prior close with above-average volume; 'breakdown' for the mirror case against
    support; None otherwise.
    """
    if len(closes) < 2:
        return None

    latest_close = closes[-1]
    prev_close = closes[-2]

    history = volumes[:-1]
    window = history[-lookback_avg:] if len(history) > lookback_avg else history
    avg_volume = np.mean(window) if len(window) > 0 else 0.0
    volume_confirmed = avg_volume > 0 and volumes[-1] > volume_multiplier * avg_volume

    resistances = sorted(lvl.price for lvl in levels if lvl.kind == "resistance" and lvl.price > prev_close)
    if resistances and latest_close > resistances[0] and volume_confirmed:
        return "breakout"

    supports = sorted((lvl.price for lvl in levels if lvl.kind == "support" and lvl.price < prev_close), reverse=True)
    if supports and latest_close < supports[0] and volume_confirmed:
        return "breakdown"

    return None
