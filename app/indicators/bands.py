import numpy as np

from app.indicators.moving_averages import sma


def bollinger_bands(closes: np.ndarray, period: int = 20, num_std: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (upper, middle, lower) bands. Middle is the SMA; upper/lower
    are `num_std` rolling standard deviations away from it.
    """
    n = len(closes)
    middle = sma(closes, period)
    std = np.full(n, np.nan)
    for i in range(period - 1, n):
        std[i] = np.std(closes[i - period + 1 : i + 1], ddof=0)

    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower
