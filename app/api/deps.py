from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.domain.ports import (
    CachePort,
    CorporateActionRepositoryPort,
    FinancialResultRepositoryPort,
    HistoricalPriceRepositoryPort,
    StockDataProviderPort,
    StockRepositoryPort,
)
from app.infrastructure.db.session import get_db_session
from app.providers.nse.client import NseClient
from app.providers.nse.nse_provider import NseStockDataProvider
from app.repositories.corporate_action_repository import SqlAlchemyCorporateActionRepository
from app.repositories.financial_result_repository import SqlAlchemyFinancialResultRepository
from app.repositories.historical_price_repository import SqlAlchemyHistoricalPriceRepository
from app.repositories.stock_repository import SqlAlchemyStockRepository
from app.services.corporate_action_service import CorporateActionService
from app.services.fundamentals_service import FundamentalsService
from app.services.indicator_service import IndicatorService
from app.services.intraday_signal_service import IntradaySignalService
from app.services.long_term_signal_service import LongTermSignalService
from app.services.price_history_service import PriceHistoryService
from app.services.stock_service import StockService

# Everything below is the one place where abstract ports get bound to concrete
# adapters. `services/` and `api/v1/*` routes never import a concrete
# implementation directly - only these functions do.


def get_cache(request: Request) -> CachePort:
    return request.app.state.cache


def get_nse_client(request: Request) -> NseClient:
    return request.app.state.nse_client


def get_nse_provider(client: NseClient = Depends(get_nse_client)) -> StockDataProviderPort:
    return NseStockDataProvider(client)


def get_stock_repository(db: AsyncSession = Depends(get_db_session)) -> StockRepositoryPort:
    return SqlAlchemyStockRepository(db)


def get_stock_service(
    repository: StockRepositoryPort = Depends(get_stock_repository),
    provider: StockDataProviderPort = Depends(get_nse_provider),
    cache: CachePort = Depends(get_cache),
    settings: Settings = Depends(get_settings),
) -> StockService:
    return StockService(repository, provider, cache, settings)


def get_historical_price_repository(db: AsyncSession = Depends(get_db_session)) -> HistoricalPriceRepositoryPort:
    return SqlAlchemyHistoricalPriceRepository(db)


def get_corporate_action_repository(db: AsyncSession = Depends(get_db_session)) -> CorporateActionRepositoryPort:
    return SqlAlchemyCorporateActionRepository(db)


def get_price_history_service(
    repository: HistoricalPriceRepositoryPort = Depends(get_historical_price_repository),
    provider: StockDataProviderPort = Depends(get_nse_provider),
    stock_repository: StockRepositoryPort = Depends(get_stock_repository),
) -> PriceHistoryService:
    return PriceHistoryService(repository, provider, stock_repository)


def get_indicator_service(
    stock_repository: StockRepositoryPort = Depends(get_stock_repository),
    price_repository: HistoricalPriceRepositoryPort = Depends(get_historical_price_repository),
) -> IndicatorService:
    return IndicatorService(stock_repository, price_repository)


def get_corporate_action_service(
    repository: CorporateActionRepositoryPort = Depends(get_corporate_action_repository),
    provider: StockDataProviderPort = Depends(get_nse_provider),
    stock_repository: StockRepositoryPort = Depends(get_stock_repository),
) -> CorporateActionService:
    return CorporateActionService(repository, provider, stock_repository)


def get_intraday_signal_service(
    stock_repository: StockRepositoryPort = Depends(get_stock_repository),
    price_repository: HistoricalPriceRepositoryPort = Depends(get_historical_price_repository),
) -> IntradaySignalService:
    return IntradaySignalService(stock_repository, price_repository)


def get_financial_result_repository(db: AsyncSession = Depends(get_db_session)) -> FinancialResultRepositoryPort:
    return SqlAlchemyFinancialResultRepository(db)


def get_fundamentals_service(
    stock_repository: StockRepositoryPort = Depends(get_stock_repository),
    financial_result_repository: FinancialResultRepositoryPort = Depends(get_financial_result_repository),
    price_repository: HistoricalPriceRepositoryPort = Depends(get_historical_price_repository),
    corporate_action_repository: CorporateActionRepositoryPort = Depends(get_corporate_action_repository),
) -> FundamentalsService:
    return FundamentalsService(stock_repository, financial_result_repository, price_repository, corporate_action_repository)


def get_long_term_signal_service(
    stock_repository: StockRepositoryPort = Depends(get_stock_repository),
    fundamentals_service: FundamentalsService = Depends(get_fundamentals_service),
) -> LongTermSignalService:
    return LongTermSignalService(stock_repository, fundamentals_service)
