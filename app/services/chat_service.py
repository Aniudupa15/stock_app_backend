import re
import uuid
from decimal import Decimal

from app.core.exceptions import StockNotFoundError
from app.domain.ports import StockRepositoryPort
from app.schemas.chat import ChatIntent, ChatResponse
from app.services.indicator_service import IndicatorService
from app.services.notification_service import NotificationService
from app.services.portfolio_service import PortfolioService
from app.services.stock_service import StockService
from app.services.watchlist_service import WatchlistService

_TOKEN_RE = re.compile(r"[A-Za-z]+")
_STOPWORDS = {
    "a",
    "am",
    "an",
    "and",
    "any",
    "are",
    "at",
    "do",
    "doing",
    "for",
    "going",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "new",
    "of",
    "on",
    "please",
    "show",
    "tell",
    "the",
    "to",
    "todays",
    "what",
    "whats",
    "whos",
    "with",
    "you",
}
_INDICATOR_KEYWORDS = ("rsi", "macd", "indicator", "technical", "sma", "moving average")
_QUOTE_KEYWORDS = ("price", "quote", "trading at", "worth", "ltp")
_PORTFOLIO_KEYWORDS = ("portfolio", "holding", "invested", "pnl", "p&l", "profit", "loss")
_WATCHLIST_KEYWORDS = ("watchlist", "watching")
_ALERTS_KEYWORDS = ("alert", "notification")

HELP_TEXT = (
    "I can answer questions like: 'what's my portfolio doing', 'RSI for RELIANCE', "
    "'price of TCS', \"what's in my watchlist\", or 'any new alerts'."
)


def _fmt(value: Decimal | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}{suffix}"


class ChatService:
    """Rule-based Q&A over a fixed set of intents - keyword-matches the
    message, then dispatches to the real service that already computes the
    answer. No LLM, no free-form generation: every reply is a template filled
    with real computed values, matching the "templated reasoning over real
    computed values" philosophy already used by IntradaySignalService /
    LongTermSignalService for AI signals.
    """

    def __init__(
        self,
        stock_repository: StockRepositoryPort,
        stock_service: StockService,
        indicator_service: IndicatorService,
        portfolio_service: PortfolioService,
        watchlist_service: WatchlistService,
        notification_service: NotificationService,
    ):
        self._stock_repository = stock_repository
        self._stock_service = stock_service
        self._indicator_service = indicator_service
        self._portfolio_service = portfolio_service
        self._watchlist_service = watchlist_service
        self._notification_service = notification_service

    async def _extract_known_symbol(self, message: str) -> str | None:
        for token in _TOKEN_RE.findall(message):
            if len(token) < 2 or token.lower() in _STOPWORDS:
                continue
            stock = await self._stock_repository.get_by_symbol(token.upper())
            if stock is not None:
                return stock.symbol
        return None

    async def ask(self, user_id: uuid.UUID, message: str) -> ChatResponse:
        lowered = message.lower()
        symbol = await self._extract_known_symbol(message)

        if symbol and any(k in lowered for k in _INDICATOR_KEYWORDS):
            return await self._answer_indicators(symbol)
        if symbol and any(k in lowered for k in _QUOTE_KEYWORDS):
            return await self._answer_quote(symbol)
        if any(k in lowered for k in _PORTFOLIO_KEYWORDS):
            return await self._answer_portfolio(user_id)
        if any(k in lowered for k in _WATCHLIST_KEYWORDS):
            return await self._answer_watchlist(user_id)
        if any(k in lowered for k in _ALERTS_KEYWORDS):
            return await self._answer_notifications(user_id)
        return ChatResponse(intent=ChatIntent.UNKNOWN, answer=HELP_TEXT)

    async def _answer_indicators(self, symbol: str) -> ChatResponse:
        try:
            out = await self._indicator_service.get_indicators(symbol)
        except StockNotFoundError:
            return ChatResponse(intent=ChatIntent.INDICATOR_SUMMARY, answer=f"I couldn't find data for {symbol}.")

        if not out.has_data:
            return ChatResponse(
                intent=ChatIntent.INDICATOR_SUMMARY,
                answer=f"{symbol} doesn't have enough price history yet for indicators.",
            )

        parts = []
        if out.rsi_14 is not None:
            parts.append(f"RSI(14) is {out.rsi_14}")
        if out.sma_50 is not None:
            parts.append(f"SMA(50) is {out.sma_50}")
        if out.macd.macd is not None and out.macd.signal is not None:
            parts.append(f"MACD ({out.macd.macd}) vs signal ({out.macd.signal})")

        answer = (
            f"{symbol} as of {out.as_of}: " + "; ".join(parts) + "."
            if parts
            else f"No indicator data is available yet for {symbol}."
        )
        return ChatResponse(intent=ChatIntent.INDICATOR_SUMMARY, answer=answer)

    async def _answer_quote(self, symbol: str) -> ChatResponse:
        detail = await self._stock_service.get_detail(symbol)
        if detail.quote is None:
            answer = f"{symbol} ({detail.name}) - live quote is currently unavailable."
        else:
            q = detail.quote
            direction = "up" if q.change >= 0 else "down"
            answer = (
                f"{symbol} ({detail.name}) is trading at {q.last_price}, "
                f"{direction} {abs(q.change)} ({q.change_percent}%) today."
            )
        return ChatResponse(intent=ChatIntent.STOCK_QUOTE, answer=answer)

    async def _answer_portfolio(self, user_id: uuid.UUID) -> ChatResponse:
        portfolios = await self._portfolio_service.list(user_id)
        if not portfolios:
            return ChatResponse(intent=ChatIntent.PORTFOLIO_SUMMARY, answer="You don't have any portfolios yet.")

        lines = []
        for p in portfolios:
            perf = await self._portfolio_service.get_performance(user_id, p.id)
            lines.append(
                f"{p.name}: invested {_fmt(perf.total_invested)}, now worth {_fmt(perf.current_value)}, "
                f"P&L {_fmt(perf.total_pnl)} ({_fmt(perf.total_pnl_percent, '%')})"
            )
        return ChatResponse(intent=ChatIntent.PORTFOLIO_SUMMARY, answer=" | ".join(lines))

    async def _answer_watchlist(self, user_id: uuid.UUID) -> ChatResponse:
        watchlists = await self._watchlist_service.list(user_id)
        if not watchlists:
            return ChatResponse(intent=ChatIntent.WATCHLIST_SUMMARY, answer="You don't have any watchlists yet.")

        lines = [f"{w.name} ({w.item_count} stocks)" for w in watchlists]
        return ChatResponse(intent=ChatIntent.WATCHLIST_SUMMARY, answer="Your watchlists: " + ", ".join(lines) + ".")

    async def _answer_notifications(self, user_id: uuid.UUID) -> ChatResponse:
        notifications = await self._notification_service.list(user_id, unread_only=True, limit=5, offset=0)
        if not notifications:
            return ChatResponse(intent=ChatIntent.ALERTS_SUMMARY, answer="You have no unread notifications.")

        titles = "; ".join(n.title for n in notifications)
        answer = f"You have {len(notifications)} unread notification(s): {titles}."
        return ChatResponse(intent=ChatIntent.ALERTS_SUMMARY, answer=answer)
