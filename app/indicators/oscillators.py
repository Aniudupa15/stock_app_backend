import numpy as np

from app.indicators.moving_averages import sma


def rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's RSI. Result[i] is NaN until index `period`."""
    n = len(closes)
    result = np.full(n, np.nan)
    if n < period + 1:
        return result

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    result[period] = _rsi_from_averages(avg_gain, avg_loss)

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        result[i + 1] = _rsi_from_averages(avg_gain, avg_loss)

    return result


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def stochastic_rsi(
    closes: np.ndarray,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Stochastic RSI: RSI's own position within its rolling high/low range,
    then smoothed into %K and %D lines. Returns (percent_k, percent_d).
    """
    rsi_values = rsi(closes, rsi_period)
    n = len(closes)
    stoch = np.full(n, np.nan)

    for i in range(stoch_period - 1, n):
        window = rsi_values[i - stoch_period + 1 : i + 1]
        if np.isnan(window).any():
            continue
        lo, hi = np.min(window), np.max(window)
        stoch[i] = 0.0 if hi == lo else (rsi_values[i] - lo) / (hi - lo) * 100

    percent_k = sma(stoch, k_smooth)
    percent_d = sma(percent_k, d_smooth)
    return percent_k, percent_d
