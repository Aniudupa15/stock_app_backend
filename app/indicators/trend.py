import numpy as np

from app.indicators.moving_averages import ema
from app.indicators.volatility import atr


def macd(
    closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = ema_fast - ema_slow

    n = len(closes)
    valid_start = slow - 1
    signal_line = np.full(n, np.nan)
    if n > valid_start:
        signal_line[valid_start:] = ema(macd_line[valid_start:], signal)

    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's Average Directional Index."""
    n = len(closes)
    result = np.full(n, np.nan)
    if n < period * 2 + 1:
        return result

    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))

    atr_smooth = float(np.mean(tr[1 : period + 1]))
    plus_dm_smooth = float(np.mean(plus_dm[1 : period + 1]))
    minus_dm_smooth = float(np.mean(minus_dm[1 : period + 1]))

    dx_values: list[float] = []
    for i in range(period + 1, n):
        atr_smooth = (atr_smooth * (period - 1) + tr[i]) / period
        plus_dm_smooth = (plus_dm_smooth * (period - 1) + plus_dm[i]) / period
        minus_dm_smooth = (minus_dm_smooth * (period - 1) + minus_dm[i]) / period

        plus_di = 100 * plus_dm_smooth / atr_smooth if atr_smooth else 0.0
        minus_di = 100 * minus_dm_smooth / atr_smooth if atr_smooth else 0.0
        di_sum = plus_di + minus_di
        dx = 100 * abs(plus_di - minus_di) / di_sum if di_sum else 0.0
        dx_values.append(dx)

    if len(dx_values) < period:
        return result

    dx_array = np.array(dx_values)
    first_adx = float(np.mean(dx_array[:period]))
    idx = period + 1 + period - 1  # position of the first ADX value in `result`
    result[idx] = first_adx

    prev_adx = first_adx
    for j in range(period, len(dx_array)):
        prev_adx = (prev_adx * (period - 1) + dx_array[j]) / period
        idx += 1
        result[idx] = prev_adx

    return result


def supertrend(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 10, multiplier: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (supertrend_line, direction). `line[i]` is NaN during ATR warmup
    (see `atr()`); `direction[i]` is `1` (uptrend), `-1` (downtrend), or `0`
    during that same warmup period - `0` is a "not yet computed" sentinel
    (int arrays can't hold NaN), not a real neutral/sideways signal.
    """
    n = len(closes)
    atr_values = atr(highs, lows, closes, period)
    hl2 = (highs + lows) / 2
    upper_band = hl2 + multiplier * atr_values
    lower_band = hl2 - multiplier * atr_values

    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    line = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)

    for i in range(n):
        if np.isnan(atr_values[i]):
            continue

        if np.isnan(final_upper[i - 1]) if i > 0 else True:
            final_upper[i] = upper_band[i]
            final_lower[i] = lower_band[i]
            direction[i] = 1 if closes[i] > final_upper[i] else -1
            line[i] = final_lower[i] if direction[i] == 1 else final_upper[i]
            continue

        final_upper[i] = (
            upper_band[i] if (upper_band[i] < final_upper[i - 1] or closes[i - 1] > final_upper[i - 1]) else final_upper[i - 1]
        )
        final_lower[i] = (
            lower_band[i] if (lower_band[i] > final_lower[i - 1] or closes[i - 1] < final_lower[i - 1]) else final_lower[i - 1]
        )

        if closes[i] > final_upper[i - 1]:
            direction[i] = 1
        elif closes[i] < final_lower[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

        line[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    return line, direction
