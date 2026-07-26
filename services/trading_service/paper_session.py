"""Persisted paper trading session.

Runs a saved strategy over a symbol's history through the shared Backtester
(same venue+OMS+risk+strategy as live) and commits the outcome to the paper
account: the trade journal, the equity curve, the updated virtual balance, and
an audit entry. This is what turns "can backtest" into "the account actually
paper-traded" - the mandated default mode, persisted.

The identical persistence path is reused by the live paper loop (which feeds
live ticks instead of historical bars).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.historical_price_repository import SqlAlchemyHistoricalPriceRepository
from libs.backtest.metrics import BacktestMetrics
from libs.backtest.runner import Backtester, BacktestResult
from libs.strategy.engine import Strategy
from services.trading_service.features import IndicatorFeatureBuilder
from services.trading_service.historical import to_backtest_bars
from services.trading_service.persistence.models import TradingAccountModel
from services.trading_service.persistence.repositories import (
    AuditLogRepository,
    EquitySnapshotRepository,
    TradeRepository,
    TradingAccountRepository,
)

_MIN_BARS = 30
_DEFAULT_CASH = Decimal("1000000")


class InsufficientHistoryError(Exception):
    def __init__(self, symbol: str, have: int):
        super().__init__(f"not enough history for {symbol}: {have} bars (need >= {_MIN_BARS})")
        self.symbol = symbol
        self.have = have


def metrics_to_dict(m: BacktestMetrics) -> dict:
    return {
        "total_trades": m.total_trades,
        "wins": m.wins,
        "losses": m.losses,
        "win_rate": round(m.win_rate, 4),
        "net_pnl": float(m.net_pnl),
        "gross_profit": float(m.gross_profit),
        "gross_loss": float(m.gross_loss),
        "profit_factor": m.profit_factor,
        "avg_win": float(m.avg_win),
        "avg_loss": float(m.avg_loss),
        "expectancy": float(m.expectancy),
        "max_consecutive_wins": m.max_consecutive_wins,
        "max_consecutive_losses": m.max_consecutive_losses,
        "max_drawdown_pct": round(m.max_drawdown_pct, 4),
        "sharpe": round(m.sharpe, 3),
        "sortino": round(m.sortino, 3),
        "cagr": round(m.cagr, 4),
    }


async def run_paper_session(
    session: AsyncSession,
    account: TradingAccountModel,
    strategy: Strategy,
    symbol: str,
    from_date: date | None,
    to_date: date | None,
) -> BacktestResult:
    """Run and persist a paper session. Raises InsufficientHistoryError if the
    symbol lacks enough bars. Commits trades, equity curve, balance, audit."""
    symbol = symbol.strip().upper()
    from_date = from_date or date(2000, 1, 1)
    to_date = to_date or date.today()

    ohlcv = await SqlAlchemyHistoricalPriceRepository(session).get_bars(symbol, from_date, to_date)
    if len(ohlcv) < _MIN_BARS:
        raise InsufficientHistoryError(symbol, len(ohlcv))

    bars = to_backtest_bars(ohlcv)
    builder = IndicatorFeatureBuilder(bars)
    starting = Decimal(account.virtual_balance) if account.virtual_balance is not None else _DEFAULT_CASH

    result = await Backtester(symbol=symbol, strategy=strategy, starting_cash=starting, account_id=account.id).run(
        bars, builder
    )

    trade_repo = TradeRepository(session)
    for trade in result.trades:
        await trade_repo.add(trade)

    equity_repo = EquitySnapshotRepository(session)
    for ts, equity in result.equity_curve:
        await equity_repo.append(account.id, equity, equity, ts=ts)

    await TradingAccountRepository(session).update_balance(account.id, result.final_equity)
    await AuditLogRepository(session).append(
        actor="paper-engine",
        event_type="PAPER_RUN",
        account_id=account.id,
        payload={
            "symbol": symbol,
            "strategy_id": str(strategy.strategy_id) if strategy.strategy_id else None,
            "trades": len(result.trades),
            "net_pnl": str(result.metrics.net_pnl),
        },
    )
    return result
