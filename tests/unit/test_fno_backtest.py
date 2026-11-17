"""F&O backtest replay. Candles here are SYNTHETIC (scripted paths) - they
test the mechanics (signals, fills, square-off, charges, option model), not
the strategy's edge."""

import math
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

from libs.derivatives.contracts import InstrumentKind
from libs.trading_calendar.calendar import IST
from libs.trading_domain.enums import ExitReason
from services.trading_service.fno_backtest import (
    Candle,
    FnoBacktestConfig,
    _path,
    black_scholes,
    expiry_for,
    run_fno_backtest,
)


def _day(d: date, price_at) -> list[Candle]:
    """375 one-minute candles 09:15-15:29; price_at(minute) -> close."""
    out, prev = [], Decimal(str(price_at(0)))
    for m in range(375):
        close = Decimal(str(price_at(m)))
        ts = IST.localize(datetime.combine(d, time(9, 15)) + timedelta(minutes=m))
        out.append(Candle(ts, prev, max(prev, close) + 2, min(prev, close) - 2, close))
        prev = close
    return out


def _v_shape(m):  # fall 30 min, rally 60, drift down
    if m < 30:
        return 24000 - 5 * m
    if m < 90:
        return 23850 + 8 * (m - 30)
    return 24330 - 3 * (m - 90)


def _flat(m):
    return 24000 + (m % 2)


async def test_futures_backtest_trades_and_squares_off_daily():
    candles = _day(date(2026, 1, 5), _v_shape) + _day(date(2026, 1, 6), _flat)
    result = await run_fno_backtest(candles, FnoBacktestConfig(lot_size=75))
    assert result.days == 2
    assert result.trades, "V-shaped day should trigger the EMA crossover"
    first = result.trades[0]
    assert first.qty == 75 and first.entry_ts.date() == date(2026, 1, 5)
    assert all(t.exit_ts.time() <= time(15, 15) for t in result.trades)
    # Net P&L reconciles: trades vs equity curve.
    assert result.metrics.net_pnl == sum(t.pnl_net for t in result.trades)
    assert result.equity_curve[-1][1] == Decimal("500000") + result.metrics.net_pnl
    assert sum(result.daily_pnl.values()) == result.metrics.net_pnl
    assert result.total_charges > 0
    assert any("Net P&L" in line for line in result.lines())


async def test_open_position_is_squared_off_at_1515():
    def late_rally(m):  # flat, then a slow rally starting 14:30 - no target hit before 15:15
        return 24000 if m < 315 else 24000 + (m - 315)

    result = await run_fno_backtest(_day(date(2026, 1, 5), late_rally), FnoBacktestConfig(lot_size=75))
    (trade,) = result.trades
    assert trade.exit_reason is ExitReason.SQUARE_OFF
    assert trade.exit_ts == IST.localize(datetime(2026, 1, 5, 15, 15))


async def test_option_backtest_uses_modelled_atm_premiums():
    d = date(2026, 1, 5)  # Monday; NIFTY weekly expiry Tuesday 6th
    candles = _day(d, _v_shape)
    vix = [(c.ts, Decimal("13")) for c in candles]
    result = await run_fno_backtest(candles, FnoBacktestConfig(instrument="CE", lot_size=75), vix=vix)
    assert result.trades
    t = result.trades[0]
    assert 10 < t.entry_price < 400  # a plausible 1-DTE ATM premium, not an index level
    assert any("MODELLED" in line for line in result.lines())


async def test_option_backtest_requires_vix():
    with pytest.raises(ValueError):
        await run_fno_backtest(_day(date(2026, 1, 5), _flat), FnoBacktestConfig(instrument="PE"))


def test_black_scholes_put_call_parity_and_expiry():
    s, k, t, v, r = 24000.0, 24000.0, 7 / 365, 0.13, 0.065
    call = black_scholes(InstrumentKind.CE, s, k, t, v)
    put = black_scholes(InstrumentKind.PE, s, k, t, v)
    assert call - put == pytest.approx(s - k * math.exp(-r * t), abs=0.01)
    assert 150 < call < 250  # ~1-week ATM NIFTY call at 13 VIX
    assert black_scholes(InstrumentKind.CE, 24100, 24000, 0, v) == 100
    assert black_scholes(InstrumentKind.PE, 24100, 24000, 0, v) == 0


def test_expiry_rules():
    assert expiry_for("NIFTY", date(2026, 1, 5)) == date(2026, 1, 6)  # Mon -> Tue
    assert expiry_for("NIFTY", date(2026, 1, 6)) == date(2026, 1, 6)  # expiry day itself
    assert expiry_for("NIFTY", date(2026, 1, 7)) == date(2026, 1, 13)
    assert expiry_for("BANKNIFTY", date(2026, 1, 5)) == date(2026, 1, 27)  # last Tuesday
    assert expiry_for("BANKNIFTY", date(2026, 1, 28)) == date(2026, 2, 24)
    assert expiry_for("RELIANCE", date(2026, 12, 30)) == date(2027, 1, 26)


def test_candle_path_visits_nearer_extreme_first():
    up = Candle(IST.localize(datetime(2026, 1, 5, 10)), Decimal(100), Decimal(110), Decimal(98), Decimal(108))
    assert _path(up) == [Decimal(100), Decimal(98), Decimal(110), Decimal(108)]
    down = Candle(up.ts, Decimal(100), Decimal(101), Decimal(90), Decimal(92))
    assert _path(down) == [Decimal(100), Decimal(101), Decimal(90), Decimal(92)]
