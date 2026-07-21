import logging
from datetime import date, timedelta

from app.domain.ports import FinancialResultRepositoryPort, StockDataProviderPort

logger = logging.getLogger(__name__)

_DEFAULT_LOOKBACK_DAYS = 400


class FinancialResultsSyncService:
    """Per-symbol sync: fetches the financial-results filing index for one
    symbol, parses each filing's XBRL, and upserts whatever quarters were
    successfully parsed. The building block the scheduled job iterates over -
    kept separate so it's independently testable and reusable (e.g. an
    on-demand "refresh this stock now" admin action later).
    """

    def __init__(self, provider: StockDataProviderPort, repository: FinancialResultRepositoryPort):
        self._provider = provider
        self._repository = repository

    async def sync_symbol(self, symbol: str, lookback_days: int = _DEFAULT_LOOKBACK_DAYS) -> int:
        to_date = date.today()
        from_date = to_date - timedelta(days=lookback_days)
        filings = await self._provider.fetch_financial_results_index(symbol, from_date, to_date)

        records = []
        for filing in filings:
            record = await self._provider.fetch_financial_result_detail(filing)
            if record is not None:
                records.append(record)

        if not records:
            return 0
        return await self._repository.bulk_upsert(records)
