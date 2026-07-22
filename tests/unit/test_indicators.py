import numpy as np
import pytest

from app.indicators.bands import bollinger_bands
from app.indicators.levels import pivot_points
from app.indicators.moving_averages import ema, sma
from app.indicators.oscillators import rsi, stochastic_rsi
from app.indicators.trend import adx, macd, supertrend
from app.indicators.volatility import atr
from app.indicators.volume import point_of_control, volume_profile, vwap


def test_sma_matches_hand_calculation():
    closes = np.array([1, 2, 3, 4, 5], dtype=float)
    result = sma(closes, period=3)
    assert np.isnan(result[0]) and np.isnan(result[1])
    assert result[2] == pytest.approx(2.0)
    assert result[3] == pytest.approx(3.0)
    assert result[4] == pytest.approx(4.0)


def test_ema_seeds_with_sma_then_diverges_on_nonlinear_data():
    closes = np.array([10, 10, 10, 10, 10, 20], dtype=float)
    result = ema(closes, period=5)
    assert np.isnan(result[3])
    assert result[4] == pytest.approx(10.0)  # seed = mean of first 5
    assert result[5] > result[4]  # pulled up by the jump to 20


def test_pivot_points_classic_formula():
    levels = pivot_points(high=110, low=90, close=100)
    assert levels.pivot == pytest.approx(100.0)
    assert levels.r1 == pytest.approx(110.0)
    assert levels.s1 == pytest.approx(90.0)
    assert levels.r2 == pytest.approx(120.0)
    assert levels.s2 == pytest.approx(80.0)
    assert levels.r3 == pytest.approx(130.0)
    assert levels.s3 == pytest.approx(70.0)


def test_bollinger_bands_matches_hand_calculation():
    # nine 10s then one 20, period=10 -> mean=11, population std=3
    closes = np.array([10] * 9 + [20], dtype=float)
    upper, middle, lower = bollinger_bands(closes, period=10, num_std=2.0)
    assert middle[-1] == pytest.approx(11.0)
    assert upper[-1] == pytest.approx(17.0)
    assert lower[-1] == pytest.approx(5.0)


def test_rsi_extremes_on_monotonic_series():
    up = np.array([float(i) for i in range(1, 40)])
    down = np.array([float(40 - i) for i in range(1, 40)])
    assert rsi(up, period=14)[-1] == pytest.approx(100.0)
    assert rsi(down, period=14)[-1] == pytest.approx(0.0)


def test_rsi_warmup_length():
    closes = np.array([float(i) for i in range(1, 30)])
    result = rsi(closes, period=14)
    assert np.isnan(result[:14]).all()
    assert not np.isnan(result[14])


def test_atr_warmup_length_and_nonnegative():
    rng = np.random.default_rng(1)
    n = 40
    closes = 100 + np.cumsum(rng.normal(0, 1, n))
    highs = closes + rng.uniform(0.5, 2, n)
    lows = closes - rng.uniform(0.5, 2, n)
    result = atr(highs, lows, closes, period=14)
    assert np.isnan(result[:13]).all()
    assert not np.isnan(result[13:]).any()
    assert (result[13:] >= 0).all()


def test_adx_stays_within_valid_range():
    rng = np.random.default_rng(2)
    n = 60
    closes = 100 + np.cumsum(rng.normal(0, 1, n))
    highs = closes + rng.uniform(0.5, 2, n)
    lows = closes - rng.uniform(0.5, 2, n)
    result = adx(highs, lows, closes, period=14)
    valid = result[~np.isnan(result)]
    assert len(valid) > 0
    assert (valid >= 0).all() and (valid <= 100).all()


def test_adx_higher_for_strong_trend_than_choppy_series():
    n = 60
    trending_closes = 100 + np.arange(n, dtype=float) * 1.5
    trending_highs = trending_closes + 1
    trending_lows = trending_closes - 1

    rng = np.random.default_rng(3)
    choppy_closes = 100 + rng.normal(0, 0.1, n)  # noisy, no direction
    choppy_highs = choppy_closes + 0.5
    choppy_lows = choppy_closes - 0.5

    trend_adx = adx(trending_highs, trending_lows, trending_closes, period=14)
    choppy_adx = adx(choppy_highs, choppy_lows, choppy_closes, period=14)

    assert np.nanmean(trend_adx) > np.nanmean(choppy_adx)


def test_supertrend_direction_is_only_up_down_or_warmup_sentinel():
    rng = np.random.default_rng(4)
    n = 40
    closes = 100 + np.cumsum(rng.normal(0, 1, n))
    highs = closes + rng.uniform(0.5, 2, n)
    lows = closes - rng.uniform(0.5, 2, n)
    line, direction = supertrend(highs, lows, closes, period=10, multiplier=3.0)
    assert set(np.unique(direction)).issubset({-1, 0, 1})
    assert len(line) == n


def test_macd_output_shapes_match_input():
    closes = np.array([float(i) for i in range(1, 60)])
    macd_line, signal_line, histogram = macd(closes)
    assert macd_line.shape == signal_line.shape == histogram.shape == closes.shape


def test_stochastic_rsi_bounded_between_0_and_100():
    rng = np.random.default_rng(5)
    closes = 100 + np.cumsum(rng.normal(0, 1, 60))
    k, d = stochastic_rsi(closes)
    valid_k = k[~np.isnan(k)]
    valid_d = d[~np.isnan(d)]
    assert len(valid_k) > 0
    assert (valid_k >= 0).all() and (valid_k <= 100).all()
    assert (valid_d >= 0).all() and (valid_d <= 100).all()


def test_vwap_within_price_range():
    rng = np.random.default_rng(6)
    n = 30
    closes = 100 + rng.normal(0, 2, n)
    highs = closes + 1
    lows = closes - 1
    volumes = rng.uniform(1000, 5000, n)
    result = vwap(highs, lows, closes, volumes, period=10)
    valid = result[~np.isnan(result)]
    assert len(valid) > 0
    assert (valid >= lows.min()).all() and (valid <= highs.max()).all()


def test_volume_profile_conserves_total_volume():
    highs = np.array([102.0, 103.0, 104.0])
    lows = np.array([98.0, 99.0, 100.0])
    closes = np.array([100.0, 101.0, 102.0])
    volumes = np.array([1000.0, 2000.0, 3000.0])

    bins = volume_profile(highs, lows, closes, volumes, num_bins=5)
    assert sum(b.volume for b in bins) == pytest.approx(6000.0)

    poc = point_of_control(bins)
    assert poc is not None
    assert poc.volume == max(b.volume for b in bins)


def test_point_of_control_empty_bins_returns_none():
    assert point_of_control([]) is None
