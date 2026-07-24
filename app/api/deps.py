import uuid

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.security import decode_access_token
from app.domain.ports import (
    AlertRepositoryPort,
    CachePort,
    CorporateActionRepositoryPort,
    FinancialResultRepositoryPort,
    HistoricalPriceRepositoryPort,
    IntradaySignalSnapshotRepositoryPort,
    IpoRepositoryPort,
    LongTermSignalSnapshotRepositoryPort,
    MarketMoverRepositoryPort,
    NewsProviderPort,
    NewsRepositoryPort,
    NotificationRepositoryPort,
    PortfolioRepositoryPort,
    RefreshTokenRepositoryPort,
    ScreenerRepositoryPort,
    SearchHistoryRepositoryPort,
    StockDataProviderPort,
    StockRepositoryPort,
    UserRepositoryPort,
    WatchlistRepositoryPort,
)
from app.infrastructure.db.session import get_db_session
from app.providers.news.news_provider import RssNewsProvider
from app.providers.nse.client import NseClient
from app.providers.nse.nse_provider import NseStockDataProvider
from app.repositories.alert_repository import SqlAlchemyAlertRepository
from app.repositories.corporate_action_repository import SqlAlchemyCorporateActionRepository
from app.repositories.financial_result_repository import SqlAlchemyFinancialResultRepository
from app.repositories.historical_price_repository import SqlAlchemyHistoricalPriceRepository
from app.repositories.intraday_signal_snapshot_repository import SqlAlchemyIntradaySignalSnapshotRepository
from app.repositories.ipo_repository import SqlAlchemyIpoRepository
from app.repositories.long_term_signal_snapshot_repository import SqlAlchemyLongTermSignalSnapshotRepository
from app.repositories.market_mover_repository import SqlAlchemyMarketMoverRepository
from app.repositories.news_repository import SqlAlchemyNewsRepository
from app.repositories.notification_repository import SqlAlchemyNotificationRepository
from app.repositories.portfolio_repository import SqlAlchemyPortfolioRepository
from app.repositories.refresh_token_repository import SqlAlchemyRefreshTokenRepository
from app.repositories.screener_repository import SqlAlchemyScreenerRepository
from app.repositories.search_history_repository import SqlAlchemySearchHistoryRepository
from app.repositories.stock_repository import SqlAlchemyStockRepository
from app.repositories.user_repository import SqlAlchemyUserRepository
from app.repositories.watchlist_repository import SqlAlchemyWatchlistRepository
from app.services.alert_service import AlertService
from app.services.analysis_service import AnalysisService
from app.services.auth_service import AuthService
from app.services.chat_service import ChatService
from app.services.comparison_service import ComparisonService
from app.services.corporate_action_service import CorporateActionService
from app.services.dashboard_service import DashboardService
from app.services.dividend_service import DividendService
from app.services.fundamentals_service import FundamentalsService
from app.services.indicator_service import IndicatorService
from app.services.intraday_signal_service import IntradaySignalService
from app.services.ipo_service import IpoService
from app.services.long_term_signal_service import LongTermSignalService
from app.services.market_mover_service import MarketMoverService
from app.services.news_service import NewsService
from app.services.notification_service import NotificationService
from app.services.portfolio_service import PortfolioService
from app.services.price_history_service import PriceHistoryService
from app.services.screener_service import ScreenerService
from app.services.search_history_service import SearchHistoryService
from app.services.stock_service import StockService
from app.services.watchlist_service import WatchlistService

# Everything below is the one place where abstract ports get bound to concrete
# adapters. `services/` and `api/v1/*` routes never import a concrete
# implementation directly - only these functions do.

_bearer_scheme = HTTPBearer()
_optional_bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> uuid.UUID:
    return decode_access_token(credentials.credentials, settings)


def get_optional_user_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(_optional_bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> uuid.UUID | None:
    """For endpoints that work anonymously but behave differently when
    authenticated (e.g. search history logging on `/stocks/search`). No
    header at all -> None. A present-but-invalid/expired token still 401s -
    silently swallowing a bad token would hide a real client bug.
    """
    if credentials is None:
        return None
    return decode_access_token(credentials.credentials, settings)


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
    return FundamentalsService(
        stock_repository, financial_result_repository, price_repository, corporate_action_repository
    )


def get_long_term_signal_service(
    stock_repository: StockRepositoryPort = Depends(get_stock_repository),
    fundamentals_service: FundamentalsService = Depends(get_fundamentals_service),
) -> LongTermSignalService:
    return LongTermSignalService(stock_repository, fundamentals_service)


def get_dividend_service(
    corporate_action_repository: CorporateActionRepositoryPort = Depends(get_corporate_action_repository),
    price_repository: HistoricalPriceRepositoryPort = Depends(get_historical_price_repository),
    stock_repository: StockRepositoryPort = Depends(get_stock_repository),
) -> DividendService:
    return DividendService(corporate_action_repository, price_repository, stock_repository)


def get_intraday_signal_snapshot_repository(
    db: AsyncSession = Depends(get_db_session),
) -> IntradaySignalSnapshotRepositoryPort:
    return SqlAlchemyIntradaySignalSnapshotRepository(db)


def get_long_term_signal_snapshot_repository(
    db: AsyncSession = Depends(get_db_session),
) -> LongTermSignalSnapshotRepositoryPort:
    return SqlAlchemyLongTermSignalSnapshotRepository(db)


def get_analysis_service(
    intraday_snapshot_repository: IntradaySignalSnapshotRepositoryPort = Depends(
        get_intraday_signal_snapshot_repository
    ),
    long_term_snapshot_repository: LongTermSignalSnapshotRepositoryPort = Depends(
        get_long_term_signal_snapshot_repository
    ),
) -> AnalysisService:
    return AnalysisService(intraday_snapshot_repository, long_term_snapshot_repository)


def get_market_mover_repository(db: AsyncSession = Depends(get_db_session)) -> MarketMoverRepositoryPort:
    return SqlAlchemyMarketMoverRepository(db)


def get_market_mover_service(
    repository: MarketMoverRepositoryPort = Depends(get_market_mover_repository),
) -> MarketMoverService:
    return MarketMoverService(repository)


def get_watchlist_repository(db: AsyncSession = Depends(get_db_session)) -> WatchlistRepositoryPort:
    return SqlAlchemyWatchlistRepository(db)


def get_watchlist_service(
    repository: WatchlistRepositoryPort = Depends(get_watchlist_repository),
    price_repository: MarketMoverRepositoryPort = Depends(get_market_mover_repository),
) -> WatchlistService:
    return WatchlistService(repository, price_repository)


def get_portfolio_repository(db: AsyncSession = Depends(get_db_session)) -> PortfolioRepositoryPort:
    return SqlAlchemyPortfolioRepository(db)


def get_portfolio_service(
    repository: PortfolioRepositoryPort = Depends(get_portfolio_repository),
    price_repository: MarketMoverRepositoryPort = Depends(get_market_mover_repository),
) -> PortfolioService:
    return PortfolioService(repository, price_repository)


def get_news_provider() -> NewsProviderPort:
    return RssNewsProvider()


def get_news_repository(db: AsyncSession = Depends(get_db_session)) -> NewsRepositoryPort:
    return SqlAlchemyNewsRepository(db)


def get_news_service(
    provider: NewsProviderPort = Depends(get_news_provider),
    repository: NewsRepositoryPort = Depends(get_news_repository),
    stock_repository: StockRepositoryPort = Depends(get_stock_repository),
) -> NewsService:
    return NewsService(provider, repository, stock_repository)


def get_alert_repository(db: AsyncSession = Depends(get_db_session)) -> AlertRepositoryPort:
    return SqlAlchemyAlertRepository(db)


def get_alert_service(repository: AlertRepositoryPort = Depends(get_alert_repository)) -> AlertService:
    return AlertService(repository)


def get_notification_repository(db: AsyncSession = Depends(get_db_session)) -> NotificationRepositoryPort:
    return SqlAlchemyNotificationRepository(db)


def get_notification_service(
    repository: NotificationRepositoryPort = Depends(get_notification_repository),
) -> NotificationService:
    return NotificationService(repository)


def get_dashboard_service(
    stock_service: StockService = Depends(get_stock_service),
    market_mover_service: MarketMoverService = Depends(get_market_mover_service),
    news_service: NewsService = Depends(get_news_service),
    cache: CachePort = Depends(get_cache),
    settings: Settings = Depends(get_settings),
) -> DashboardService:
    return DashboardService(stock_service, market_mover_service, news_service, cache, settings)


def get_user_repository(db: AsyncSession = Depends(get_db_session)) -> UserRepositoryPort:
    return SqlAlchemyUserRepository(db)


def get_refresh_token_repository(db: AsyncSession = Depends(get_db_session)) -> RefreshTokenRepositoryPort:
    return SqlAlchemyRefreshTokenRepository(db)


def get_auth_service(
    user_repository: UserRepositoryPort = Depends(get_user_repository),
    refresh_token_repository: RefreshTokenRepositoryPort = Depends(get_refresh_token_repository),
    settings: Settings = Depends(get_settings),
) -> AuthService:
    return AuthService(user_repository, refresh_token_repository, settings)


def get_search_history_repository(db: AsyncSession = Depends(get_db_session)) -> SearchHistoryRepositoryPort:
    return SqlAlchemySearchHistoryRepository(db)


def get_search_history_service(
    repository: SearchHistoryRepositoryPort = Depends(get_search_history_repository),
) -> SearchHistoryService:
    return SearchHistoryService(repository)


def get_comparison_service(
    stock_service: StockService = Depends(get_stock_service),
    indicator_service: IndicatorService = Depends(get_indicator_service),
    fundamentals_service: FundamentalsService = Depends(get_fundamentals_service),
) -> ComparisonService:
    return ComparisonService(stock_service, indicator_service, fundamentals_service)


def get_screener_repository(db: AsyncSession = Depends(get_db_session)) -> ScreenerRepositoryPort:
    return SqlAlchemyScreenerRepository(db)


def get_screener_service(
    repository: ScreenerRepositoryPort = Depends(get_screener_repository),
) -> ScreenerService:
    return ScreenerService(repository)


def get_ipo_repository(db: AsyncSession = Depends(get_db_session)) -> IpoRepositoryPort:
    return SqlAlchemyIpoRepository(db)


def get_ipo_service(
    provider: StockDataProviderPort = Depends(get_nse_provider),
    repository: IpoRepositoryPort = Depends(get_ipo_repository),
) -> IpoService:
    return IpoService(provider, repository)


def get_chat_service(
    stock_repository: StockRepositoryPort = Depends(get_stock_repository),
    stock_service: StockService = Depends(get_stock_service),
    indicator_service: IndicatorService = Depends(get_indicator_service),
    portfolio_service: PortfolioService = Depends(get_portfolio_service),
    watchlist_service: WatchlistService = Depends(get_watchlist_service),
    notification_service: NotificationService = Depends(get_notification_service),
) -> ChatService:
    return ChatService(
        stock_repository, stock_service, indicator_service, portfolio_service, watchlist_service, notification_service
    )
