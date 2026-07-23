from datetime import date, timedelta
from decimal import Decimal

import numpy as np

from app.domain.entities import StockIndicatorSnapshot
from app.domain.ports import HistoricalPriceRepositoryPort, ScreenerRepositoryPort, StockRepositoryPort
from app.indicators.moving_averages import sma
from app.indicators.oscillators import rsi

# Enough calendar days for a 200-session SMA to have real data even after
# weekends/holidays are accounted for - same lookback IndicatorService uses.
_LOOKBACK_CALENDAR_DAYS = 450


def _last_or_none(values: np.ndarray) -> Decimal | None:
    if len(values) == 0:
        return None
    last = values[-1]
    if np.isnan(last):
        return None
    return Decimal(str(round(float(last), 4)))


class IndicatorSnapshotSyncService:
    """Refreshes the materialized `stock_indicator_snapshots` table the
    screener reads from. Meant to run once daily right after the price sync
    - every active stock gets its snapshot overwritten with the latest
    close/volume/RSI/SMA values.
    """

    def __init__(
        self,
        stock_repository: StockRepositoryPort,
        price_repository: HistoricalPriceRepositoryPort,
        screener_repository: ScreenerRepositoryPort,
    ):
        self._stock_repository = stock_repository
        self._price_repository = price_repository
        self._screener_repository = screener_repository

    async def sync_all(self) -> int:
        symbols = await self._stock_repository.list_active_symbols()
        to_date = date.today()
        from_date = to_date - timedelta(days=_LOOKBACK_CALENDAR_DAYS)

        snapshots = []
        for symbol in symbols:
            bars = await self._price_repository.get_bars(symbol, from_date, to_date)
            if not bars:
                continue

            closes = np.array([float(b.close) for b in bars])
            latest = bars[-1]
            snapshots.append(
                StockIndicatorSnapshot(
                    symbol=symbol,
                    name="",  # not stored - the repository joins to `stocks` for name at read time
                    as_of=latest.trade_date,
                    close=latest.close,
                    volume=latest.volume,
                    rsi_14=_last_or_none(rsi(closes, 14)),
                    sma_50=_last_or_none(sma(closes, 50)),
                    sma_200=_last_or_none(sma(closes, 200)),
                )
            )

        return await self._screener_repository.bulk_upsert(snapshots)
