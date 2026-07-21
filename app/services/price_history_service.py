import logging
from datetime import date, timedelta

from app.core.exceptions import StockNotFoundError
from app.domain.ports import HistoricalPriceRepositoryPort, StockDataProviderPort, StockRepositoryPort
from app.schemas.history import HistoryOut, OhlcvBarOut

logger = logging.getLogger(__name__)

# 1D/5D can only ever show whatever daily bars fall in that window (1-5 points) -
# the NSE Bhavcopy archive is end-of-day only, no intraday ticks are available.
_RANGE_TO_DAYS = {
    "1D": 1,
    "5D": 5,
    "1M": 30,
    "3M": 90,
    "6M": 182,
    "1Y": 365,
    "3Y": 365 * 3,
    "5Y": 365 * 5,
    "MAX": 365 * 30,
}
_DEFAULT_RANGE = "1Y"


class PriceHistoryService:
    def __init__(
        self,
        repository: HistoricalPriceRepositoryPort,
        provider: StockDataProviderPort,
        stock_repository: StockRepositoryPort,
    ):
        self._repository = repository
        self._provider = provider
        self._stock_repository = stock_repository

    async def backfill_date(self, trade_date: date) -> int:
        """Fetch and store one day's Bhavcopy. Returns rows upserted (0 for
        holidays/weekends, where NSE simply has no file - not an error).
        """
        records = await self._provider.fetch_daily_bars(trade_date)
        if not records:
            return 0
        return await self._repository.bulk_upsert_bars(records)

    async def get_history(self, symbol: str, range_key: str) -> HistoryOut:
        stock = await self._stock_repository.get_by_symbol(symbol)
        if stock is None:
            raise StockNotFoundError(symbol)

        normalized_range = range_key.upper()
        days = _RANGE_TO_DAYS.get(normalized_range, _RANGE_TO_DAYS[_DEFAULT_RANGE])
        to_date = date.today()
        from_date = to_date - timedelta(days=days)

        bars = await self._repository.get_bars(stock.symbol, from_date, to_date)

        return HistoryOut(
            symbol=stock.symbol,
            range=normalized_range if normalized_range in _RANGE_TO_DAYS else _DEFAULT_RANGE,
            bars=[
                OhlcvBarOut(
                    trade_date=b.trade_date, open=b.open, high=b.high, low=b.low, close=b.close, volume=b.volume
                )
                for b in bars
            ],
        )
