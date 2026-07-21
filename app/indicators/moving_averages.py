"""SMA and EMA. Pure numpy, no framework/DB/HTTP dependency - reusable
by any feature that needs price averaging (indicator endpoints, Phase 3's
AI engines, etc.).
"""

import numpy as np


def sma(values: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average. Result[i] is NaN until index period-1."""
    n = len(values)
    result = np.full(n, np.nan)
    if period <= 0:
        return result
    for i in range(period - 1, n):
        result[i] = np.mean(values[i - period + 1 : i + 1])
    return result


def ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average, seeded with the SMA of the first `period` values."""
    n = len(values)
    result = np.full(n, np.nan)
    if period <= 0 or n < period:
        return result

    multiplier = 2 / (period + 1)
    result[period - 1] = np.mean(values[:period])
    for i in range(period, n):
        result[i] = (values[i] - result[i - 1]) * multiplier + result[i - 1]
    return result
