import logging

from app.domain.ports import StockDataProviderPort, StockRepositoryPort

logger = logging.getLogger(__name__)


class UniverseSyncService:
    """Replaces any hardcoded stock list: pulls the full NSE equity master and
    upserts it into `stocks`, soft-delisting anything no longer present.
    """

    def __init__(self, provider: StockDataProviderPort, repository: StockRepositoryPort):
        self._provider = provider
        self._repository = repository

    async def sync_equity_universe(self) -> dict[str, int]:
        records = await self._provider.fetch_equity_universe()
        upserted = await self._repository.upsert_universe(records)
        active_symbols = {r.symbol for r in records}
        deactivated = await self._repository.deactivate_missing(active_symbols)
        logger.info("Universe sync complete: upserted=%d deactivated=%d", upserted, deactivated)
        return {"upserted": upserted, "deactivated": deactivated}
