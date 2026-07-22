import logging

from app.core.config import Settings
from app.core.exceptions import ProviderUnavailableError, StockNotFoundError
from app.domain.ports import CachePort, StockDataProviderPort, StockRepositoryPort
from app.schemas.market_status import IndexQuoteOut, MarketStatusOut
from app.schemas.stock import QuoteOut, StockDetail, StockSearchResult

logger = logging.getLogger(__name__)


class StockService:
    """Orchestrates repository (DB), provider (NSE), and cache to serve search and
    stock-detail requests. Depends only on the abstract ports in `app.domain.ports` -
    never imports a concrete NSE/SQLAlchemy/cache implementation directly.
    """

    def __init__(
        self,
        repository: StockRepositoryPort,
        provider: StockDataProviderPort,
        cache: CachePort,
        settings: Settings,
    ):
        self._repository = repository
        self._provider = provider
        self._cache = cache
        self._settings = settings

    async def search(self, query: str, limit: int) -> list[StockSearchResult]:
        cache_key = f"search:{query.strip().lower()}:{limit}"
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return cached

        stocks = await self._repository.search_by_symbol_or_name(query, limit)
        results = [
            StockSearchResult(
                symbol=s.symbol,
                name=s.name,
                isin=s.isin,
                series=s.series,
                instrument_type=s.instrument_type,
            )
            for s in stocks
        ]
        await self._cache.set(cache_key, results, self._settings.CACHE_SEARCH_TTL_SECONDS)
        return results

    async def get_detail(self, symbol: str) -> StockDetail:
        stock = await self._repository.get_by_symbol(symbol)
        if stock is None:
            raise StockNotFoundError(symbol)

        quote_out: QuoteOut | None = None
        quote_unavailable_reason: str | None = None

        cache_key = f"quote:{stock.symbol}"
        cached_quote = await self._cache.get(cache_key)
        if cached_quote is not None:
            quote_out = cached_quote
        else:
            try:
                quote = await self._provider.get_quote(stock.symbol)
            except ProviderUnavailableError as exc:
                # Graceful degradation: a live-quote failure should never turn a
                # valid stock lookup into a 500 - the caller still gets the DB info.
                logger.warning("Quote unavailable for %s: %s", stock.symbol, exc)
                quote_unavailable_reason = str(exc)
            else:
                quote_out = QuoteOut(
                    last_price=quote.last_price,
                    change=quote.change,
                    change_percent=quote.change_percent,
                    open=quote.open,
                    high=quote.high,
                    low=quote.low,
                    previous_close=quote.previous_close,
                    volume=quote.volume,
                    as_of=quote.as_of,
                )
                await self._cache.set(cache_key, quote_out, self._settings.CACHE_QUOTE_TTL_SECONDS)

        return StockDetail(
            symbol=stock.symbol,
            isin=stock.isin,
            name=stock.name,
            series=stock.series,
            sector=stock.sector,
            industry=stock.industry,
            instrument_type=stock.instrument_type,
            listing_date=stock.listing_date,
            face_value=stock.face_value,
            quote=quote_out,
            quote_unavailable_reason=quote_unavailable_reason,
        )

    async def get_market_status(self) -> list[MarketStatusOut]:
        """Best-effort - returns an empty list (not an error) if NSE's
        cookie-gated status endpoint is unreachable, same graceful-degradation
        shape as `get_detail`'s quote lookup.
        """
        try:
            statuses = await self._provider.fetch_market_status()
        except ProviderUnavailableError as exc:
            logger.warning("Market status unavailable: %s", exc)
            return []
        return [MarketStatusOut(market=s.market, status=s.status, as_of=s.as_of) for s in statuses]

    async def get_indices(self) -> list[IndexQuoteOut]:
        """Best-effort - returns an empty list (not an error) if NSE's
        cookie-gated indices endpoint is unreachable.
        """
        try:
            indices = await self._provider.fetch_indices()
        except ProviderUnavailableError as exc:
            logger.warning("Indices unavailable: %s", exc)
            return []
        return [
            IndexQuoteOut(
                index_name=i.index_name, last_price=i.last_price, change=i.change, change_percent=i.change_percent
            )
            for i in indices
        ]
