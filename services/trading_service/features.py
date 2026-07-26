"""Compute the strategy feature dict from bars using the data-service's real
12-indicator engine (`app.indicators`, pure numpy).

Produces a `FeatureBuilder` for the backtester/engine: index i -> (features at
i, features at i-1). NaN indicator warmup values are omitted, so a rule
referencing an indicator that isn't ready yet simply reads it as missing (and
the DSL treats that leaf as False - the conservative reading).

Feature keys strategies can reference:
  close open high low volume
  SMA_20 SMA_50 SMA_200 EMA_20 EMA_50 RSI_14
  MACD MACD_SIGNAL MACD_HIST BB_UPPER BB_MIDDLE BB_LOWER
  VWAP_20 ADX_14 ATR_14 SUPERTREND SUPERTREND_DIR STOCH_K STOCH_D
"""

from __future__ import annotations

import math

import numpy as np

from app.indicators.bands import bollinger_bands
from app.indicators.moving_averages import ema, sma
from app.indicators.oscillators import rsi, stochastic_rsi
from app.indicators.trend import adx, macd, supertrend
from app.indicators.volatility import atr
from app.indicators.volume import vwap
from libs.backtest.bar import Bar


def _put(row: dict, key: str, arr: np.ndarray, i: int) -> None:
    if i < len(arr):
        value = float(arr[i])
        if not math.isnan(value):
            row[key] = value


def build_feature_rows(bars: list[Bar]) -> list[dict]:
    """One feature dict per bar (index-aligned with `bars`)."""
    if not bars:
        return []

    closes = np.array([float(b.close) for b in bars])
    highs = np.array([float(b.high) for b in bars])
    lows = np.array([float(b.low) for b in bars])
    volumes = np.array([float(b.volume) for b in bars])

    sma20, sma50, sma200 = sma(closes, 20), sma(closes, 50), sma(closes, 200)
    ema20, ema50 = ema(closes, 20), ema(closes, 50)
    rsi14 = rsi(closes, 14)
    macd_line, macd_signal, macd_hist = macd(closes)
    bb_u, bb_m, bb_l = bollinger_bands(closes)
    vwap20 = vwap(highs, lows, closes, volumes, 20)
    adx14 = adx(highs, lows, closes, 14)
    atr14 = atr(highs, lows, closes, 14)
    st_line, st_dir = supertrend(highs, lows, closes)
    stoch_k, stoch_d = stochastic_rsi(closes)

    rows: list[dict] = []
    for i, bar in enumerate(bars):
        row: dict = {
            "close": float(bar.close),
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "volume": float(bar.volume),
        }
        _put(row, "SMA_20", sma20, i)
        _put(row, "SMA_50", sma50, i)
        _put(row, "SMA_200", sma200, i)
        _put(row, "EMA_20", ema20, i)
        _put(row, "EMA_50", ema50, i)
        _put(row, "RSI_14", rsi14, i)
        _put(row, "MACD", macd_line, i)
        _put(row, "MACD_SIGNAL", macd_signal, i)
        _put(row, "MACD_HIST", macd_hist, i)
        _put(row, "BB_UPPER", bb_u, i)
        _put(row, "BB_MIDDLE", bb_m, i)
        _put(row, "BB_LOWER", bb_l, i)
        _put(row, "VWAP_20", vwap20, i)
        _put(row, "ADX_14", adx14, i)
        _put(row, "ATR_14", atr14, i)
        _put(row, "SUPERTREND", st_line, i)
        # direction uses 0 as a "not computed" sentinel - omit it, not report 0.
        if i < len(st_dir) and int(st_dir[i]) != 0:
            row["SUPERTREND_DIR"] = int(st_dir[i])
        _put(row, "STOCH_K", stoch_k, i)
        _put(row, "STOCH_D", stoch_d, i)
        rows.append(row)
    return rows


class IndicatorFeatureBuilder:
    """Callable FeatureBuilder for the backtester: `builder(i) -> (features_i,
    features_{i-1} | None)`."""

    def __init__(self, bars: list[Bar]) -> None:
        self._rows = build_feature_rows(bars)

    def __call__(self, index: int) -> tuple[dict, dict | None]:
        prev = self._rows[index - 1] if index > 0 else None
        return self._rows[index], prev
