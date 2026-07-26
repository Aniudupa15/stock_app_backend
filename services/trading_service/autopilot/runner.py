"""Auto-pilot run over REAL stored history: scan -> pick -> trade each candidate
with the default strategy -> persist trades -> return a report.

This is the "flip it on and it trades" experience on the data we actually have
(daily bars). It runs the built-in momentum strategy across today's top scanned
candidates and records the resulting trades to the paper account. Capital is
split across the picks. (v1 runs each pick as an independent backtest over the
lookback window rather than one shared-capital portfolio loop - simpler and
reuses the fully-tested Backtester; a shared-capital portfolio pass is a
follow-up.)

Live intraday execution is a separate path (needs a broker tick feed - NSE's
live API is IP-blocked from servers).
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.historical_price_repository import SqlAlchemyHistoricalPriceRepository
from libs.backtest.runner import Backtester
from libs.trading_domain.enums import Product
from services.trading_service.autopilot.eod_report import portfolio_summary
from services.trading_service.autopilot.scanner import scan_candidates
from services.trading_service.autopilot.strategies import default_momentum_strategy
from services.trading_service.features import IndicatorFeatureBuilder
from services.trading_service.historical import to_backtest_bars
from services.trading_service.persistence.models import TradingAccountModel
from services.trading_service.persistence.repositories import (
    AuditLogRepository,
    TradeRepository,
    TradingAccountRepository,
)

_MIN_BARS = 30
_DEFAULT_CASH = Decimal("1000000")


async def run_autopilot(
    session: AsyncSession,
    account: TradingAccountModel,
    *,
    top_k: int = 10,
    lookback_days: int = 180,
) -> dict:
    """Scan, trade each top candidate with the default strategy over the
    lookback window, persist trades, and return the run report."""
    candidates = await scan_candidates(session, limit=top_k)
    starting = Decimal(account.virtual_balance) if account.virtual_balance is not None else _DEFAULT_CASH
    per_symbol_cash = (starting / max(1, len(candidates))) if candidates else starting

    strategy = default_momentum_strategy(quantity=10_000_000, product=Product.CNC)  # margin caps qty to cash
    price_repo = SqlAlchemyHistoricalPriceRepository(session)
    trade_repo = TradeRepository(session)

    from_date = date.today() - timedelta(days=lookback_days)
    to_date = date.today()

    all_trades = []
    traded_symbols: list[str] = []
    for c in candidates:
        ohlcv = await price_repo.get_bars(c.symbol, from_date, to_date)
        if len(ohlcv) < _MIN_BARS:
            continue
        bars = to_backtest_bars(ohlcv)
        builder = IndicatorFeatureBuilder(bars)
        result = await Backtester(
            symbol=c.symbol, strategy=strategy, starting_cash=per_symbol_cash, account_id=account.id
        ).run(bars, builder)
        for trade in result.trades:
            await trade_repo.add(trade)
            all_trades.append(trade)
        if result.trades:
            traded_symbols.append(c.symbol)

    net = sum((t.pnl_net for t in all_trades), Decimal("0"))
    await TradingAccountRepository(session).update_balance(account.id, starting + net)
    await AuditLogRepository(session).append(
        actor="auto-pilot",
        event_type="AUTOPILOT_RUN",
        account_id=account.id,
        payload={
            "candidates": len(candidates),
            "traded": len(traded_symbols),
            "trades": len(all_trades),
            "net_pnl": str(net),
        },
    )

    return {
        "candidates": [c.as_dict() for c in candidates],
        "traded_symbols": traded_symbols,
        "lookback_days": lookback_days,
        "summary": portfolio_summary(all_trades),
    }
