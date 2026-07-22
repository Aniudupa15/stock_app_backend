import logging

from app.core.exceptions import StockNotFoundError
from app.domain.entities import NewsCategory
from app.domain.ports import NewsProviderPort, NewsRepositoryPort, StockRepositoryPort
from app.schemas.news import NewsArticleOut

logger = logging.getLogger(__name__)


def _to_schema(articles) -> list[NewsArticleOut]:
    return [
        NewsArticleOut(
            headline=a.headline,
            summary=a.summary,
            source=a.source,
            url=a.url,
            category=a.category,
            related_symbols=a.related_symbols,
            published_at=a.published_at,
        )
        for a in articles
    ]


class NewsService:
    def __init__(
        self, provider: NewsProviderPort, repository: NewsRepositoryPort, stock_repository: StockRepositoryPort
    ):
        self._provider = provider
        self._repository = repository
        self._stock_repository = stock_repository

    async def sync(self) -> int:
        """Best-effort: raises ProviderUnavailableError only if every feed
        failed (the scheduled job that calls this decides how to log it) -
        the API always serves whatever was last synced successfully.
        """
        known_symbols = set(await self._stock_repository.list_active_symbols())
        articles = await self._provider.fetch_latest(known_symbols)
        if not articles:
            return 0
        return await self._repository.bulk_upsert(articles)

    async def list_latest(
        self, category: NewsCategory | None, symbol: str | None, limit: int, offset: int
    ) -> list[NewsArticleOut]:
        articles = await self._repository.list_latest(category, symbol, limit, offset)
        return _to_schema(articles)

    async def get_for_symbol(self, symbol: str) -> list[NewsArticleOut]:
        stock = await self._stock_repository.get_by_symbol(symbol)
        if stock is None:
            raise StockNotFoundError(symbol)

        articles = await self._repository.list_latest(None, stock.symbol, limit=50, offset=0)
        return _to_schema(articles)
