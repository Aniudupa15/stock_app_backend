import logging
from datetime import date

from app.core.exceptions import StockNotFoundError
from app.domain.ports import CorporateActionRepositoryPort, StockDataProviderPort, StockRepositoryPort
from app.schemas.corporate_action import CorporateActionOut

logger = logging.getLogger(__name__)


class CorporateActionService:
    def __init__(
        self,
        repository: CorporateActionRepositoryPort,
        provider: StockDataProviderPort,
        stock_repository: StockRepositoryPort,
    ):
        self._repository = repository
        self._provider = provider
        self._stock_repository = stock_repository

    async def sync(self, from_date: date, to_date: date) -> int:
        """Pull corporate actions for a date window and store them.

        Best-effort: raises ProviderUnavailableError on failure (the scheduled
        job that calls this decides how to log it) - the API always serves
        whatever was last synced successfully, even if this call fails.
        """
        records = await self._provider.fetch_corporate_actions(from_date, to_date)
        if not records:
            return 0
        return await self._repository.bulk_upsert(records)

    async def get_for_symbol(self, symbol: str) -> list[CorporateActionOut]:
        stock = await self._stock_repository.get_by_symbol(symbol)
        if stock is None:
            raise StockNotFoundError(symbol)

        records = await self._repository.get_for_symbol(stock.symbol)
        return [
            CorporateActionOut(
                purpose=r.purpose,
                face_value=r.face_value,
                ex_date=r.ex_date,
                record_date=r.record_date,
                book_closure_start=r.book_closure_start,
                book_closure_end=r.book_closure_end,
            )
            for r in records
        ]
