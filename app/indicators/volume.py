from dataclasses import dataclass

import numpy as np


def vwap(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray, period: int = 20) -> np.ndarray:
    """Rolling volume-weighted average price over `period` bars.

    True VWAP resets every trading session using intraday ticks; Phase 2 only
    has daily EOD bars (from the NSE Bhavcopy archive), so this is a rolling
    multi-day approximation - typical price weighted by volume over a trailing
    window - not a same-day session VWAP.
    """
    n = len(closes)
    typical = (highs + lows + closes) / 3
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        vol_window = volumes[i - period + 1 : i + 1]
        tp_window = typical[i - period + 1 : i + 1]
        total_vol = np.sum(vol_window)
        result[i] = np.sum(tp_window * vol_window) / total_vol if total_vol else np.nan
    return result


@dataclass(frozen=True, slots=True)
class VolumeProfileBin:
    price_low: float
    price_high: float
    volume: float


def volume_profile(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray, num_bins: int = 20
) -> list[VolumeProfileBin]:
    """Buckets traded volume into `num_bins` price ranges across the given window."""
    if len(closes) == 0:
        return []

    price_min = float(np.min(lows))
    price_max = float(np.max(highs))
    if price_max <= price_min:
        return [VolumeProfileBin(price_low=price_min, price_high=price_max, volume=float(np.sum(volumes)))]

    edges = np.linspace(price_min, price_max, num_bins + 1)
    bin_volumes = np.zeros(num_bins)
    typical = (highs + lows + closes) / 3
    bin_indices = np.clip(np.digitize(typical, edges) - 1, 0, num_bins - 1)
    for idx, vol in zip(bin_indices, volumes, strict=False):
        bin_volumes[idx] += vol

    return [
        VolumeProfileBin(price_low=float(edges[i]), price_high=float(edges[i + 1]), volume=float(bin_volumes[i]))
        for i in range(num_bins)
    ]


def point_of_control(bins: list[VolumeProfileBin]) -> VolumeProfileBin | None:
    """The price bin with the most traded volume - the level the market spent the most time/volume at."""
    if not bins:
        return None
    return max(bins, key=lambda b: b.volume)
