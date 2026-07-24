from fastapi import APIRouter

from app.api.v1 import (
    alerts,
    analysis,
    auth,
    chat,
    dashboard,
    dividends,
    health,
    ipo,
    live,
    market,
    news,
    notifications,
    portfolios,
    screener,
    search_history,
    stocks,
    watchlists,
)

router = APIRouter()
router.include_router(health.router)
router.include_router(auth.router)
router.include_router(stocks.router)
router.include_router(market.router)
router.include_router(watchlists.router)
router.include_router(portfolios.router)
router.include_router(news.router)
router.include_router(alerts.router)
router.include_router(notifications.router)
router.include_router(dashboard.router)
router.include_router(search_history.router)
router.include_router(screener.router)
router.include_router(ipo.router)
router.include_router(chat.router)
router.include_router(dividends.router)
router.include_router(analysis.router)
router.include_router(live.router)
