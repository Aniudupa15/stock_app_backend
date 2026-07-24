import logging
from datetime import date, timedelta
from decimal import Decimal

from app.domain.ports import CorporateActionRepositoryPort, HistoricalPriceRepositoryPort, StockRepositoryPort
from app.schemas.dividend import DividendRecommendationOut
from app.services.dividend_parsing import sum_dividend_amount

logger = logging.getLogger(__name__)

_UPCOMING_WINDOW_DAYS = 90
_RECENT_WINDOW_DAYS = 365
_PRICE_LOOKBACK_DAYS = 14
_HIGH_YIELD_THRESHOLD = Decimal("3.0")
_MODERATE_YIELD_THRESHOLD = Decimal("1.0")


class DividendService:
    """Rule-based dividend recommendation list, composed entirely from
    already-synced local data (corporate_actions + historical_prices) - no
    new NSE data source needed. Deterministic, no ML/LLM, same "reasoning
    names the real computed value" philosophy as the intraday/long-term
    signal services.
    """

    def __init__(
        self,
        corporate_action_repository: CorporateActionRepositoryPort,
        price_repository: HistoricalPriceRepositoryPort,
        stock_repository: StockRepositoryPort,
    ):
        self._corporate_action_repository = corporate_action_repository
        self._price_repository = price_repository
        self._stock_repository = stock_repository

    async def list_dividends(self, upcoming_only: bool, sort: str, limit: int) -> list[DividendRecommendationOut]:
        today = date.today()
        ex_date_from = today if upcoming_only else today - timedelta(days=_RECENT_WINDOW_DAYS)
        ex_date_to = today + timedelta(days=_UPCOMING_WINDOW_DAYS)

        actions = await self._corporate_action_repository.list_dividend_actions(ex_date_from, ex_date_to)

        results: list[DividendRecommendationOut] = []
        for action in actions:
            if action.ex_date is None:
                continue
            amount = sum_dividend_amount(action.purpose)
            if amount <= 0:
                continue

            stock = await self._stock_repository.get_by_symbol(action.symbol)
            if stock is None:
                continue

            price_to = today
            price_from = price_to - timedelta(days=_PRICE_LOOKBACK_DAYS)
            bars = await self._price_repository.get_bars(action.symbol, price_from, price_to)
            current_price = bars[-1].close if bars else None
            if current_price is None or current_price <= 0:
                continue

            dividend_yield = Decimal(str(round(float(amount / current_price) * 100, 2)))
            buy_before_date = action.ex_date - timedelta(days=1)
            recommendation, risk_level, confidence = self._score(dividend_yield, action.ex_date, today)

            results.append(
                DividendRecommendationOut(
                    symbol=action.symbol,
                    name=stock.name,
                    dividend_yield=dividend_yield,
                    dividend_amount=amount,
                    ex_dividend_date=action.ex_date,
                    buy_before_date=buy_before_date,
                    recommendation=recommendation,
                    risk_level=risk_level,
                    confidence=confidence,
                )
            )

        if sort == "yield":
            results.sort(key=lambda r: r.dividend_yield, reverse=True)
        else:
            results.sort(key=lambda r: r.ex_dividend_date)

        return results[:limit]

    def _score(self, dividend_yield: Decimal, ex_date: date, today: date) -> tuple[str, str, int]:
        is_upcoming = ex_date >= today
        if dividend_yield >= _HIGH_YIELD_THRESHOLD:
            recommendation = "Hold for dividend" if is_upcoming else "Hold"
            risk_level = "Low"
            confidence = 80
        elif dividend_yield >= _MODERATE_YIELD_THRESHOLD:
            recommendation = "Hold for dividend" if is_upcoming else "Hold"
            risk_level = "Moderate"
            confidence = 60
        else:
            recommendation = "Hold" if is_upcoming else "Sell"
            risk_level = "Moderate"
            confidence = 40
        return recommendation, risk_level, confidence
