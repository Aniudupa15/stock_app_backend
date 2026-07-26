"""Unit tests for backtest performance metrics. Pure."""

from datetime import datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from libs.backtest.metrics import compute_metrics
from libs.trading_domain.entities import Trade

ACCOUNT = uuid4()


def _trade(pnl_net):
    p = Decimal(str(pnl_net))
    return Trade(
        account_id=ACCOUNT,
        symbol="INFY",
        qty=10,
        entry_price=Decimal("100"),
        exit_price=Decimal("100") + p / 10,
        pnl_gross=p,
        charges_total=Decimal("0"),
        pnl_net=p,
    )


def _curve(values):
    return [(datetime(2026, 1, 5 + i, 10, 0), Decimal(str(v))) for i, v in enumerate(values)]


def test_win_rate_and_profit_factor():
    trades = [_trade(100), _trade(50), _trade(-30)]
    m = compute_metrics(trades, _curve([100000, 100100, 100150, 100120]), Decimal("100000"))
    assert m.total_trades == 3
    assert m.wins == 2 and m.losses == 1
    assert m.win_rate == pytest.approx(2 / 3)
    assert m.gross_profit == Decimal("150")
    assert m.gross_loss == Decimal("30")
    assert m.profit_factor == pytest.approx(5.0)
    assert m.net_pnl == Decimal("120")
    assert m.expectancy == Decimal("40")


def test_profit_factor_none_when_no_losses():
    m = compute_metrics([_trade(10), _trade(20)], _curve([100, 110, 130]), Decimal("100"))
    assert m.profit_factor is None
    assert m.max_consecutive_wins == 2
    assert m.max_consecutive_losses == 0


def test_consecutive_streaks():
    trades = [_trade(10), _trade(10), _trade(-5), _trade(-5), _trade(-5), _trade(10)]
    m = compute_metrics(trades, _curve([100, 110, 120, 115, 110, 105, 115]), Decimal("100"))
    assert m.max_consecutive_wins == 2
    assert m.max_consecutive_losses == 3


def test_max_drawdown():
    # peak 120 then down to 105 -> dd = (105-120)/120 = 12.5%
    m = compute_metrics([], _curve([100, 120, 105, 130]), Decimal("100"))
    assert m.max_drawdown_pct == pytest.approx(0.125)


def test_empty_backtest_is_safe():
    m = compute_metrics([], _curve([100000]), Decimal("100000"))
    assert m.total_trades == 0
    assert m.win_rate == 0.0
    assert m.profit_factor is None
    assert m.sharpe == 0.0
    assert m.max_drawdown_pct == 0.0
