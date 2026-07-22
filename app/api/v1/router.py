from fastapi import APIRouter

from app.api.v1 import alerts, auth, dashboard, health, market, news, notifications, portfolios, stocks, watchlists

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
