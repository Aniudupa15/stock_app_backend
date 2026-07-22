import logging
from datetime import date, timedelta
from decimal import Decimal

import numpy as np

from app.core.exceptions import StockNotFoundError
from app.domain.ports import HistoricalPriceRepositoryPort, StockRepositoryPort
from app.indicators.bands import bollinger_bands
from app.indicators.levels import pivot_points
from app.indicators.moving_averages import ema, sma
from app.indicators.oscillators import rsi, stochastic_rsi
from app.indicators.trend import adx, macd, supertrend
from app.indicators.volatility import atr
from app.indicators.volume import point_of_control, volume_profile, vwap
from app.schemas.indicators import (
    BollingerOut,
    IndicatorsOut,
    MacdOut,
    PivotPointsOut,
    StochRsiOut,
    SupertrendOut,
    VolumeProfileBinOut,
)

logger = logging.getLogger(__name__)

# Generous calendar-day lookback so a 200-trading-day SMA/EMA actually has
# enough bars once weekends/holidays are accounted for (~30% non-trading days).
_LOOKBACK_CALENDAR_DAYS = 450


def _last_or_none(values: np.ndarray) -> Decimal | None:
    if len(values) == 0:
        return None
    last = values[-1]
    if np.isnan(last):
        return None
    return Decimal(str(round(float(last), 4)))


def _to_decimal(value: float) -> Decimal:
    return Decimal(str(round(value, 4)))


class IndicatorService:
    """Loads historical bars and runs them through the technical indicator
    engine (`app.indicators`). Returns a snapshot of the latest reading for
    each indicator, not the full historical series - `/history` is where a
    caller gets raw OHLCV to compute its own overlays if it needs the series.
    """

    def __init__(self, stock_repository: StockRepositoryPort, price_repository: HistoricalPriceRepositoryPort):
        self._stock_repository = stock_repository
        self._price_repository = price_repository

    async def get_indicators(self, symbol: str) -> IndicatorsOut:
        stock = await self._stock_repository.get_by_symbol(symbol)
        if stock is None:
            raise StockNotFoundError(symbol)

        to_date = date.today()
        from_date = to_date - timedelta(days=_LOOKBACK_CALENDAR_DAYS)
        bars = await self._price_repository.get_bars(stock.symbol, from_date, to_date)

        if not bars:
            return IndicatorsOut(symbol=stock.symbol, has_data=False)

        closes = np.array([float(b.close) for b in bars])
        highs = np.array([float(b.high) for b in bars])
        lows = np.array([float(b.low) for b in bars])
        volumes = np.array([float(b.volume) for b in bars])

        macd_line, signal_line, histogram = macd(closes)
        bb_upper, bb_middle, bb_lower = bollinger_bands(closes)
        st_line, st_direction = supertrend(highs, lows, closes)
        stoch_k, stoch_d = stochastic_rsi(closes)

        last_bar = bars[-1]
        pivots = pivot_points(float(last_bar.high), float(last_bar.low), float(last_bar.close))
        vp_bins = volume_profile(highs, lows, closes, volumes)
        poc = point_of_control(vp_bins)
        direction_value = int(st_direction[-1]) if st_direction[-1] != 0 else None

        return IndicatorsOut(
            symbol=stock.symbol,
            as_of=last_bar.trade_date,
            has_data=True,
            sma_20=_last_or_none(sma(closes, 20)),
            sma_50=_last_or_none(sma(closes, 50)),
            sma_200=_last_or_none(sma(closes, 200)),
            ema_20=_last_or_none(ema(closes, 20)),
            ema_50=_last_or_none(ema(closes, 50)),
            rsi_14=_last_or_none(rsi(closes, 14)),
            macd=MacdOut(
                macd=_last_or_none(macd_line), signal=_last_or_none(signal_line), histogram=_last_or_none(histogram)
            ),
            bollinger=BollingerOut(
                upper=_last_or_none(bb_upper), middle=_last_or_none(bb_middle), lower=_last_or_none(bb_lower)
            ),
            vwap_20=_last_or_none(vwap(highs, lows, closes, volumes, 20)),
            adx_14=_last_or_none(adx(highs, lows, closes, 14)),
            atr_14=_last_or_none(atr(highs, lows, closes, 14)),
            supertrend=SupertrendOut(value=_last_or_none(st_line), direction=direction_value),
            stochastic_rsi=StochRsiOut(k=_last_or_none(stoch_k), d=_last_or_none(stoch_d)),
            pivot_points=PivotPointsOut(
                pivot=_to_decimal(pivots.pivot),
                r1=_to_decimal(pivots.r1),
                r2=_to_decimal(pivots.r2),
                r3=_to_decimal(pivots.r3),
                s1=_to_decimal(pivots.s1),
                s2=_to_decimal(pivots.s2),
                s3=_to_decimal(pivots.s3),
            ),
            volume_profile=[
                VolumeProfileBinOut(
                    price_low=_to_decimal(b.price_low), price_high=_to_decimal(b.price_high), volume=int(b.volume)
                )
                for b in vp_bins
            ],
            point_of_control=(
                VolumeProfileBinOut(
                    price_low=_to_decimal(poc.price_low), price_high=_to_decimal(poc.price_high), volume=int(poc.volume)
                )
                if poc
                else None
            ),
        )
