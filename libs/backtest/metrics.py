"""Performance metrics over a backtest's trade log + equity curve (Phase 3 §8,
Phase 7 validation). Money stays Decimal; statistical ratios use numpy floats.

Ratios are annualised assuming daily bars (sqrt(252)). None is returned where a
metric is undefined (e.g. profit factor with zero losing trades) rather than a
misleading 0 or inf."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import numpy as np

from libs.trading_domain.entities import Trade

_TRADING_DAYS = 252


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    net_pnl: Decimal
    gross_profit: Decimal
    gross_loss: Decimal
    profit_factor: float | None
    avg_win: Decimal
    avg_loss: Decimal
    expectancy: Decimal
    max_consecutive_wins: int
    max_consecutive_losses: int
    max_drawdown_pct: float
    sharpe: float
    sortino: float
    cagr: float


def compute_metrics(
    trades: list[Trade], equity_curve: list[tuple[datetime, Decimal]], starting_cash: Decimal
) -> BacktestMetrics:
    wins = [t for t in trades if t.pnl_net > 0]
    losses = [t for t in trades if t.pnl_net < 0]
    n = len(trades)

    gross_profit = sum((t.pnl_net for t in wins), Decimal("0"))
    gross_loss = -sum((t.pnl_net for t in losses), Decimal("0"))  # positive magnitude
    net_pnl = sum((t.pnl_net for t in trades), Decimal("0"))

    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else None
    avg_win = gross_profit / len(wins) if wins else Decimal("0")
    avg_loss = gross_loss / len(losses) if losses else Decimal("0")
    expectancy = net_pnl / n if n else Decimal("0")

    max_cw, max_cl, cw, cl = 0, 0, 0, 0
    for t in trades:
        if t.pnl_net > 0:
            cw, cl = cw + 1, 0
            max_cw = max(max_cw, cw)
        elif t.pnl_net < 0:
            cl, cw = cl + 1, 0
            max_cl = max(max_cl, cl)
        else:
            cw = cl = 0

    equity = [float(v) for _, v in equity_curve]
    return BacktestMetrics(
        total_trades=n,
        wins=len(wins),
        losses=len(losses),
        win_rate=len(wins) / n if n else 0.0,
        net_pnl=net_pnl,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        expectancy=expectancy,
        max_consecutive_wins=max_cw,
        max_consecutive_losses=max_cl,
        max_drawdown_pct=_max_drawdown(equity),
        sharpe=_sharpe(equity),
        sortino=_sortino(equity),
        cagr=_cagr(equity_curve),
    )


def _returns(equity: list[float]) -> np.ndarray:
    if len(equity) < 2:
        return np.array([])
    arr = np.array(equity, dtype=float)
    prev = arr[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev != 0, np.diff(arr) / prev, 0.0)
    return rets


def _max_drawdown(equity: list[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, (v - peak) / peak)
    return abs(mdd)


def _sharpe(equity: list[float]) -> float:
    rets = _returns(equity)
    if rets.size < 2 or rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(_TRADING_DAYS))


def _sortino(equity: list[float]) -> float:
    rets = _returns(equity)
    if rets.size < 2:
        return 0.0
    downside = rets[rets < 0]
    if downside.size == 0 or downside.std() == 0:
        return 0.0
    return float(rets.mean() / downside.std() * np.sqrt(_TRADING_DAYS))


def _cagr(equity_curve: list[tuple[datetime, Decimal]]) -> float:
    if len(equity_curve) < 2:
        return 0.0
    start = float(equity_curve[0][1])
    end = float(equity_curve[-1][1])
    days = (equity_curve[-1][0] - equity_curve[0][0]).days
    years = days / 365.25
    if years <= 0 or start <= 0:
        return 0.0
    return (end / start) ** (1 / years) - 1
