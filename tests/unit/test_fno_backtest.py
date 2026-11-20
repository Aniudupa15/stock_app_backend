"""F&O backtest replay. Candles here are SYNTHETIC (scripted paths) - they
test the mechanics (signals, fills, square-off, charges, option model), not
the strategy's edge."""

import math
from dataclasses import replace
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


# ---------------------------------------------------------------- variants


def test_trend_filter_rules():
    from services.trading_service.fno_backtest import _trend_ok

    D = Decimal
    assert _trend_ok("open", True, D(101), D(100), D(0), D(0), True)
    assert not _trend_ok("open", True, D(99), D(100), D(0), D(0), True)
    assert _trend_ok("open", False, D(99), D(100), D(0), D(0), True)  # PE wants a falling day
    assert not _trend_ok("orb", True, D(120), D(100), D(110), D(90), False)  # range not finished
    assert _trend_ok("orb", True, D(111), D(100), D(110), D(90), True)
    assert _trend_ok("orb", False, D(89), D(100), D(110), D(90), True)


async def test_orb_filter_blocks_entries_on_a_day_that_never_breaks_out():
    def fade(m):  # spikes in the first 15 min, then V-shape below that high
        if m < 15:
            return 24000 + 20 * m
        return _v_shape(m) - 100

    base = FnoBacktestConfig(lot_size=75)
    plain = await run_fno_backtest(_day(date(2026, 1, 5), fade), base)
    orb = await run_fno_backtest(_day(date(2026, 1, 5), fade), FnoBacktestConfig(lot_size=75, trend_filter="orb"))
    assert plain.trades and not orb.trades


async def test_no_target_rides_to_square_off():
    result = await run_fno_backtest(
        _day(date(2026, 1, 5), _v_shape),
        FnoBacktestConfig(lot_size=75, no_target=True, sl_pct=Decimal("0.8"), trail_pct=Decimal("0.8")),
    )
    assert result.trades
    assert all(t.exit_reason is not ExitReason.TARGET for t in result.trades)


async def test_sweep_runs_every_variant_and_next_expiry_differs():
    from services.trading_service.fno_backtest import sweep_table, sweep_variants

    candles = _day(date(2026, 1, 5), _v_shape)
    vix = [(c.ts, Decimal("13")) for c in candles]
    variants = sweep_variants(FnoBacktestConfig(instrument="CE", lot_size=75))
    assert len(variants) == 9 and variants[0].label == "baseline"
    results = [await run_fno_backtest(candles, v, vix=vix) for v in variants]
    table = sweep_table(results)
    assert len(table) == 2 + len(variants)
    near = results[0].trades[0].entry_price
    nxt = next(r for r in results if r.config.expiry == "next" and r.trades).trades[0].entry_price
    assert nxt > near  # 8-day option costs more than a 1-day one


async def test_target_sweep_changes_only_the_exit():
    from services.trading_service.fno_backtest import target_variants

    base = FnoBacktestConfig(instrument="PE", max_trades=1, trend_filter="open", sl_pct=Decimal("30"),
                             trail_pct=Decimal("30"), no_target=True, lot_size=65)  # fmt: skip
    vs = target_variants(base)
    assert [v.label for v in vs] == ["no target (ride winners)", "target 30%", "target 60%", "target 100%"]
    for v in vs:
        assert (v.sl_pct, v.trail_pct, v.trend_filter, v.max_trades) == (Decimal("30"), Decimal("30"), "open", 1)
    assert vs[0].no_target and not vs[1].no_target and vs[3].target_pct == Decimal(100)


async def test_max_risk_skips_expensive_signals_like_the_live_bot():
    candles = _day(date(2026, 1, 5), _v_shape)
    vix = [(c.ts, Decimal("13")) for c in candles]
    base = FnoBacktestConfig(instrument="CE", lot_size=75, max_trades=1, no_target=True,
                             sl_pct=Decimal("30"), trail_pct=Decimal("30"))  # fmt: skip
    free = await run_fno_backtest(candles, base, vix=vix)
    assert free.trades
    entry = free.trades[0].entry_price
    risk = 75 * entry * Decimal("0.30") + 60  # entry includes ~0.2% slippage, hence the +-20 margins
    capped = await run_fno_backtest(candles, replace(base, max_trade_risk=risk - 20), vix=vix)
    assert capped.trades == [] and capped.skipped_signals == 1  # skipped, and no retry that day
    roomy = await run_fno_backtest(candles, replace(base, max_trade_risk=risk + 500), vix=vix)
    assert len(roomy.trades) == 1 and roomy.skipped_signals == 0


async def test_breakeven_lifts_stop_to_entry_and_caps_the_loss():
    # Rally ~+50% after entry, then collapse: without breakeven the trailing
    # stop (30% below the high) exits at a loss; with breakeven at +40% the
    # stop sits at the entry price, so the trade exits around breakeven.
    def spike_and_crash(m):
        if m < 30:
            return 24000 - 5 * m
        if m < 60:
            return 23850 + 3 * (m - 30)
        return 23940 - 20 * (m - 60)

    candles = _day(date(2026, 1, 5), spike_and_crash)
    vix = [(c.ts, Decimal("13")) for c in candles]
    base = FnoBacktestConfig(instrument="CE", lot_size=75, max_trades=1, no_target=True,
                             sl_pct=Decimal("30"), trail_pct=Decimal("30"))  # fmt: skip
    plain = await run_fno_backtest(candles, base, vix=vix)
    be = await run_fno_backtest(candles, replace(base, breakeven_pct=Decimal("20")), vix=vix)
    assert plain.trades and be.trades
    t_plain, t_be = plain.trades[0], be.trades[0]
    assert t_be.exit_price >= t_be.entry_price - Decimal("1")  # stopped at ~entry, not below
    assert t_be.pnl_gross >= t_plain.pnl_gross


def test_breakeven_sweep_changes_only_breakeven():
    from services.trading_service.fno_backtest import breakeven_variants

    base = FnoBacktestConfig(instrument="PE", max_trades=1, trend_filter="open", sl_pct=Decimal("30"),
                             trail_pct=Decimal("30"), no_target=True, max_trade_risk=Decimal("3000"))  # fmt: skip
    vs = breakeven_variants(base)
    assert [v.breakeven_pct for v in vs] == [None, Decimal(20), Decimal(40), Decimal(60)]
    assert all(v.no_target and v.max_trade_risk == Decimal("3000") and v.trend_filter == "open" for v in vs)


def test_retry_sweep_changes_only_the_skip_rule():
    from services.trading_service.fno_backtest import retry_variants

    base = FnoBacktestConfig(instrument="PE", max_trades=1, trend_filter="open", sl_pct=Decimal("30"),
                             trail_pct=Decimal("30"), no_target=True, breakeven_pct=Decimal("20"),
                             max_trade_risk=Decimal("3000"))  # fmt: skip
    a, b = retry_variants(base)
    assert (a.retry_after_skip, b.retry_after_skip) == (False, True)
    assert replace(b, retry_after_skip=False, label=a.label) == a


async def test_max_dte_only_trades_close_to_expiry():
    # Mon 2026-01-05 (NIFTY weekly expiry Tue 06 -> 1 day) and Wed 2026-01-07 (expiry Tue 13 -> 6 days)
    candles = _day(date(2026, 1, 5), _v_shape) + _day(date(2026, 1, 7), _v_shape)
    vix = [(c.ts, Decimal("13")) for c in candles]
    base = FnoBacktestConfig(instrument="CE", lot_size=75, max_trades=1, no_target=True,
                             sl_pct=Decimal("30"), trail_pct=Decimal("30"))  # fmt: skip
    anyday = await run_fno_backtest(candles, base, vix=vix)
    near = await run_fno_backtest(candles, replace(base, max_dte=1), vix=vix)
    expiry_only = await run_fno_backtest(candles, replace(base, max_dte=0), vix=vix)
    assert {t.entry_ts.date() for t in anyday.trades} == {date(2026, 1, 5), date(2026, 1, 7)}
    assert {t.entry_ts.date() for t in near.trades} == {date(2026, 1, 5)}
    assert expiry_only.trades == []
