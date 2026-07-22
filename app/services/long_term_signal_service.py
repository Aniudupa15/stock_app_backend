import logging

from app.core.exceptions import StockNotFoundError
from app.domain.ports import StockRepositoryPort
from app.schemas.fundamentals import FundamentalsOut
from app.schemas.long_term_signal import LongTermSignalOut
from app.services.fundamentals_service import FundamentalsService

logger = logging.getLogger(__name__)

_GROWTH_STRONG_THRESHOLD = 10.0
_DIVIDEND_YIELD_THRESHOLD = 2.0


class LongTermSignalService:
    """Rule-based long-term Buy/Hold/Avoid scoring from `FundamentalsService`
    output. Deterministic, no ML/LLM. Explicitly does not score PE ratio
    directionally - a "good" PE is sector-relative, and no sector-benchmark
    data is available - PE is surfaced as context only, never as a signal.
    """

    def __init__(self, stock_repository: StockRepositoryPort, fundamentals_service: FundamentalsService):
        self._stock_repository = stock_repository
        self._fundamentals_service = fundamentals_service

    async def get_signal(self, symbol: str) -> LongTermSignalOut:
        stock = await self._stock_repository.get_by_symbol(symbol)
        if stock is None:
            raise StockNotFoundError(symbol)

        fundamentals = await self._fundamentals_service.get_fundamentals(stock.symbol)
        if not fundamentals.has_data:
            return LongTermSignalOut(symbol=stock.symbol, has_data=False)

        return self._score(stock.symbol, fundamentals)

    def _score(self, symbol: str, f: FundamentalsOut) -> LongTermSignalOut:
        strengths: list[str] = []
        weaknesses: list[str] = []
        opportunities: list[str] = []
        risks: list[str] = []
        reasoning: list[str] = []

        revenue_component = 0
        if f.revenue_growth_yoy is not None:
            if f.revenue_growth_yoy > _GROWTH_STRONG_THRESHOLD:
                revenue_component = 1
                strengths.append(f"Revenue grew {f.revenue_growth_yoy}% year-over-year.")
            elif f.revenue_growth_yoy < 0:
                revenue_component = -1
                weaknesses.append(f"Revenue declined {f.revenue_growth_yoy}% year-over-year.")
            else:
                reasoning.append(f"Revenue growth of {f.revenue_growth_yoy}% YoY is modest.")

        profit_component = 0
        if f.profit_growth_yoy is not None:
            if f.profit_growth_yoy > _GROWTH_STRONG_THRESHOLD:
                profit_component = 1
                strengths.append(f"Profit grew {f.profit_growth_yoy}% year-over-year.")
            elif f.profit_growth_yoy < 0:
                profit_component = -1
                weaknesses.append(f"Profit declined {f.profit_growth_yoy}% year-over-year.")
            else:
                reasoning.append(f"Profit growth of {f.profit_growth_yoy}% YoY is modest.")

        if (
            f.profit_growth_yoy is not None
            and f.revenue_growth_yoy is not None
            and f.profit_growth_yoy > f.revenue_growth_yoy
            and f.profit_growth_yoy > 0
        ):
            opportunities.append("Profit is growing faster than revenue, suggesting improving margins.")
        if (
            f.profit_growth_yoy is not None
            and f.revenue_growth_yoy is not None
            and f.profit_growth_yoy < 0 < f.revenue_growth_yoy
        ):
            risks.append("Revenue is growing but profit is declining, suggesting margin pressure.")

        dividend_bonus = 0
        if f.dividend_yield is not None and f.dividend_yield > _DIVIDEND_YIELD_THRESHOLD:
            dividend_bonus = 1
            strengths.append(f"Trailing dividend yield of {f.dividend_yield}% provides an income component.")

        if f.pe_ratio is not None:
            reasoning.append(
                f"PE ratio is {f.pe_ratio} - shown for reference only; no sector-relative "
                "benchmark is available to judge whether this is cheap or expensive."
            )

        core_score = revenue_component + profit_component
        total_score = core_score + dividend_bonus

        if total_score >= 2:
            signal = "BUY"
        elif total_score <= -1:
            signal = "AVOID"
        else:
            signal = "HOLD"

        confidence = round(min(abs(total_score) / 3 * 100, 90))

        if core_score >= 2:
            growth_potential = "High"
        elif core_score == 1:
            growth_potential = "Moderate"
        else:
            growth_potential = "Low/Uncertain"

        if core_score <= -1:
            risk_level = "High"
        elif weaknesses or risks:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        if not strengths and not weaknesses:
            reasoning.append(
                "Insufficient growth signal to form a strong view - treat this as a HOLD pending more data."
            )

        return LongTermSignalOut(
            symbol=symbol,
            has_data=True,
            signal=signal,
            confidence=confidence,
            risk_level=risk_level,
            growth_potential=growth_potential,
            strengths=strengths,
            weaknesses=weaknesses,
            opportunities=opportunities,
            risks=risks,
            reasoning=reasoning,
        )
