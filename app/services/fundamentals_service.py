import logging
from datetime import date, timedelta
from decimal import Decimal

from app.core.exceptions import StockNotFoundError
from app.domain.entities import FinancialResultRecord
from app.domain.ports import (
    CorporateActionRepositoryPort,
    FinancialResultRepositoryPort,
    HistoricalPriceRepositoryPort,
    StockRepositoryPort,
)
from app.schemas.fundamentals import FundamentalsOut
from app.services.dividend_parsing import sum_dividend_amount

logger = logging.getLogger(__name__)

_RECENT_QUARTERS_LIMIT = 8
_QOQ_TARGET_DAYS = 90
_YOY_TARGET_DAYS = 365
_MATCH_TOLERANCE_DAYS = 20
_PRICE_LOOKBACK_DAYS = 14
_DIVIDEND_LOOKBACK_DAYS = 365


def _growth_percent(current: Decimal | None, prior: Decimal | None) -> Decimal | None:
    if current is None or prior is None or prior == 0:
        return None
    return Decimal(str(round(float((current - prior) / abs(prior)) * 100, 2)))


def _find_comparable_quarter(
    quarters: list[FinancialResultRecord], latest: FinancialResultRecord, days_back: int
) -> FinancialResultRecord | None:
    target_date = latest.period_end - timedelta(days=days_back)
    candidates = [(abs((q.period_end - target_date).days), q) for q in quarters if q is not latest]
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    best_diff, best_quarter = candidates[0]
    return best_quarter if best_diff <= _MATCH_TOLERANCE_DAYS else None


class FundamentalsService:
    """Derives growth rates, TTM EPS, PE, and dividend yield from stored XBRL
    financial results plus existing price/corporate-action data. Book value,
    ROE, ROCE, and debt-to-equity are not computed - no confirmed free source
    for the balance-sheet data they need (see Phase 3 plan) - and are always
    returned as null, not omitted, so API consumers can see the gap explicitly.
    """

    def __init__(
        self,
        stock_repository: StockRepositoryPort,
        financial_result_repository: FinancialResultRepositoryPort,
        price_repository: HistoricalPriceRepositoryPort,
        corporate_action_repository: CorporateActionRepositoryPort,
    ):
        self._stock_repository = stock_repository
        self._financial_result_repository = financial_result_repository
        self._price_repository = price_repository
        self._corporate_action_repository = corporate_action_repository

    async def get_fundamentals(self, symbol: str) -> FundamentalsOut:
        stock = await self._stock_repository.get_by_symbol(symbol)
        if stock is None:
            raise StockNotFoundError(symbol)

        quarters = await self._financial_result_repository.get_recent_quarters(
            stock.symbol, consolidated=False, limit=_RECENT_QUARTERS_LIMIT
        )
        if not quarters:
            # Standalone filings are the norm; consolidated-only filers (holding
            # companies with no standalone operations) fall back to those.
            quarters = await self._financial_result_repository.get_recent_quarters(
                stock.symbol, consolidated=True, limit=_RECENT_QUARTERS_LIMIT
            )

        if not quarters:
            return FundamentalsOut(symbol=stock.symbol, has_data=False)

        latest = quarters[0]
        qoq_quarter = _find_comparable_quarter(quarters, latest, _QOQ_TARGET_DAYS)
        yoy_quarter = _find_comparable_quarter(quarters, latest, _YOY_TARGET_DAYS)

        revenue_growth_qoq = _growth_percent(latest.revenue, qoq_quarter.revenue if qoq_quarter else None)
        revenue_growth_yoy = _growth_percent(latest.revenue, yoy_quarter.revenue if yoy_quarter else None)
        profit_growth_qoq = _growth_percent(latest.profit, qoq_quarter.profit if qoq_quarter else None)
        profit_growth_yoy = _growth_percent(latest.profit, yoy_quarter.profit if yoy_quarter else None)

        ttm_eps = None
        if len(quarters) >= 4 and all(q.eps_basic is not None for q in quarters[:4]):
            ttm_eps = sum((q.eps_basic for q in quarters[:4]), Decimal("0"))

        to_date = date.today()
        from_date = to_date - timedelta(days=_PRICE_LOOKBACK_DAYS)
        bars = await self._price_repository.get_bars(stock.symbol, from_date, to_date)
        current_price = bars[-1].close if bars else None

        pe_ratio = None
        if ttm_eps and ttm_eps > 0 and current_price is not None:
            pe_ratio = Decimal(str(round(float(current_price / ttm_eps), 2)))

        dividend_yield = None
        if current_price is not None and current_price > 0:
            actions = await self._corporate_action_repository.get_for_symbol(stock.symbol)
            cutoff = to_date - timedelta(days=_DIVIDEND_LOOKBACK_DAYS)
            trailing_dividends = sum(
                (
                    sum_dividend_amount(a.purpose)
                    for a in actions
                    if "dividend" in a.purpose.lower() and a.ex_date and a.ex_date >= cutoff
                ),
                Decimal("0"),
            )
            if trailing_dividends > 0:
                dividend_yield = Decimal(str(round(float(trailing_dividends / current_price) * 100, 2)))

        return FundamentalsOut(
            symbol=stock.symbol,
            has_data=True,
            latest_period_end=latest.period_end,
            revenue_growth_yoy=revenue_growth_yoy,
            revenue_growth_qoq=revenue_growth_qoq,
            profit_growth_yoy=profit_growth_yoy,
            profit_growth_qoq=profit_growth_qoq,
            ttm_eps=Decimal(str(round(float(ttm_eps), 4))) if ttm_eps is not None else None,
            pe_ratio=pe_ratio,
            dividend_yield=dividend_yield,
        )
