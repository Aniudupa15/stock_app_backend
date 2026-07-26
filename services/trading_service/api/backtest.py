"""Backtest endpoint: load a symbol's history, run the strategy through the
shared Backtester with the real indicator engine, persist, return metrics."""

from __future__ import annotations

import uuid
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.historical_price_repository import SqlAlchemyHistoricalPriceRepository
from libs.backtest.metrics import BacktestMetrics
from libs.backtest.runner import Backtester
from libs.strategy.engine import ExitRule, Strategy
from libs.trading_domain.enums import Product, Side
from services.trading_service.api.deps import get_current_user_id, get_db_session
from services.trading_service.api.schemas import BacktestOut, BacktestRequest
from services.trading_service.features import IndicatorFeatureBuilder
from services.trading_service.historical import to_backtest_bars
from services.trading_service.persistence.repositories import BacktestRepository, StrategyRepository
from services.trading_service.strategy_mapping import strategy_from_model

router = APIRouter(prefix="/trading", tags=["backtest"])

_MIN_BARS = 30  # need enough history for indicator warmup to mean anything


def _metrics_dict(m: BacktestMetrics) -> dict:
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


async def _resolve_strategy(body: BacktestRequest, user_id: uuid.UUID, session: AsyncSession) -> Strategy:
    if body.strategy_id is not None:
        model = await StrategyRepository(session).get(body.strategy_id, user_id)
        if model is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="strategy not found")
        return strategy_from_model(model)
    if body.rule_tree is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="provide strategy_id or an inline rule_tree"
        )
    return Strategy(
        name="adhoc",
        rule=body.rule_tree,
        side=Side(body.side),
        product=Product(body.product),
        exit=ExitRule(stop_loss_pct=body.stop_loss_pct, target_pct=body.target_pct),
        quantity=body.quantity,
    )


@router.post("/backtest", response_model=BacktestOut)
async def run_backtest(
    body: BacktestRequest,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
) -> BacktestOut:
    symbol = body.symbol.strip().upper()
    strategy = await _resolve_strategy(body, user_id, session)

    from_date = body.from_date or date(2000, 1, 1)
    to_date = body.to_date or date.today()
    ohlcv = await SqlAlchemyHistoricalPriceRepository(session).get_bars(symbol, from_date, to_date)
    if len(ohlcv) < _MIN_BARS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"not enough history for {symbol}: {len(ohlcv)} bars (need >= {_MIN_BARS})",
        )

    bars = to_backtest_bars(ohlcv)
    builder = IndicatorFeatureBuilder(bars)
    result = await Backtester(symbol=symbol, strategy=strategy, starting_cash=body.starting_cash).run(bars, builder)

    metrics = _metrics_dict(result.metrics)
    bt_id = await BacktestRepository(session).save(
        symbol=symbol,
        starting_cash=body.starting_cash,
        final_equity=result.final_equity,
        metrics=metrics,
        strategy_id=strategy.strategy_id,
        from_date=from_date,
        to_date=to_date,
    )
    return BacktestOut(
        id=bt_id,
        symbol=symbol,
        starting_cash=body.starting_cash,
        final_equity=result.final_equity,
        bars=len(bars),
        equity_points=len(result.equity_curve),
        metrics=metrics,
    )
