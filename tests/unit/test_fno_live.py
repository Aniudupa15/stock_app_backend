"""Live F&O executor: order lifecycle, exchange stop-loss, trailing, square-off,
risk skips, guard and pre-flight. Fake Kite + simulated order book; prices
are a SYNTHETIC script - mechanics only, no network, no real orders."""

from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from libs.trading_calendar.calendar import IST
from services.trading_service.fno_live import (
    DryRunBroker,
    GuardError,
    LiveConfig,
    OrderGuard,
    preflight,
    run_live_session,
    tick_down,
)

PE_ROWS = [
    {"instrument_token": 7, "tradingsymbol": "NIFTY2610624000PE", "name": "NIFTY", "expiry": date(2026, 1, 6),
     "strike": 24000.0, "tick_size": 0.05, "lot_size": 65, "instrument_type": "PE", "exchange": "NFO"},
]  # fmt: skip


class MarketKite:
    """Market data only. `premium(minute)` drives the put price; the index is
    `index_vs_open` points from its 24000 open. Any order call fails loudly."""

    def __init__(self, clock, premium, index_vs_open=-60):
        self.clock, self.premium, self.index_vs_open = clock, premium, index_vs_open

    def minute(self):
        start = self.clock().replace(hour=9, minute=15, second=0, microsecond=0)
        return int((self.clock() - start).total_seconds() // 60)

    def instruments(self, exchange):
        return PE_ROWS if exchange == "NFO" else [{"tradingsymbol": "NIFTY 50", "instrument_token": 256265}]

    def ltp(self, keys):
        return {k: {"last_price": 24000 + self.index_vs_open} for k in keys}

    def ohlc(self, keys):
        return {k: {"ohlc": {"open": 24000}} for k in keys}

    def quote(self, keys):
        return {k: {"last_price": self.premium(self.minute()), "volume": 1_000_000} for k in keys}

    def place_order(self, **kw):
        raise AssertionError("dry run must never reach Kite's order API")


class RecordingBroker(DryRunBroker):
    """Simulated exchange that records what a LIVE session sent."""

    def __init__(self, market):
        super().__init__()
        self.market = market
        self.placed = []

    def place_order(self, **p):
        self.placed.append(p)
        return super().place_order(**p)

    def order_history(self, order_id):
        self.mark(Decimal(str(self.market.premium(self.market.minute()))))
        return super().order_history(order_id)


def v_shape(m):  # put premium: dips, rallies hard, then fades (index-like V scaled)
    if m < 30:
        return 150 - 2.5 * m
    if m < 90:
        return 75 + 4 * (m - 30)
    return max(5, 315 - 1.5 * (m - 90))


async def _run(premium, *, live=False, index_vs_open=-60, broker_cls=None, **cfg):
    now = {"t": IST.localize(datetime(2026, 1, 5, 9, 10))}

    async def sleep(s):
        now["t"] += timedelta(seconds=s)

    kite = MarketKite(lambda: now["t"], premium, index_vs_open)
    broker = broker_cls(kite) if broker_cls else None
    notes = []
    report = await run_live_session(
        kite,
        LiveConfig(live=live, poll_seconds=15, **cfg),
        clock=lambda: now["t"],
        sleep=sleep,
        log=lambda m: None,
        notify=lambda s, b: notes.append(s),
        broker=broker,
    )
    return report, broker, notes


async def test_dry_run_sends_nothing_to_kite_and_trails_out_in_profit():
    report, _, _ = await _run(v_shape)
    assert not report.live
    (trade,) = report.trades
    assert trade.qty == 65
    assert trade.reason == "STOP_LOSS"  # the trailing stop, ratcheted up during the rally
    assert trade.exit > trade.entry and trade.net > 0
    kinds = [(o["action"], o.get("order_type")) for o in report.orders]
    assert kinds[0] == ("place", "LIMIT") and kinds[1] == ("place", "SL")
    assert any(a == "modify" for a, _ in kinds)  # trailing modifications
    # Trailing must keep going past Zerodha's per-order modification cap (the
    # stop is re-placed): exit is 30% below the 315 peak, as in the backtest.
    assert trade.exit == Decimal("220.50")
    assert sum(1 for o in report.orders if o.get("order_type") == "SL") > 1


async def test_live_entry_puts_stop_at_exchange_immediately():
    report, broker, notes = await _run(v_shape, live=True, broker_cls=RecordingBroker)
    entry, stop = broker.placed[0], broker.placed[1]
    assert (entry["transaction_type"], entry["order_type"], entry["product"], entry["exchange"]) == (
        "BUY", "LIMIT", "MIS", "NFO",
    )  # fmt: skip
    assert entry["quantity"] == 65 and entry["tag"] == "fnobot"
    assert stop["order_type"] == "SL" and stop["transaction_type"] == "SELL"
    fill = report.trades[0].entry
    assert Decimal(str(stop["trigger_price"])) == tick_down(fill * Decimal("0.70"))
    assert Decimal(str(stop["price"])) < Decimal(str(stop["trigger_price"]))
    assert any("ENTRY" in n for n in notes) and any("EXIT" in n for n in notes)


async def test_square_off_at_1515_when_stop_never_hits():
    def rally_all_day(m):
        return 75 if m < 30 else 75 + 0.5 * (m - 30)

    def slow_v(m):
        return 120 - 1.5 * m if m < 30 else rally_all_day(m)

    report, broker, _ = await _run(slow_v, live=True, broker_cls=RecordingBroker)
    (trade,) = report.trades
    assert trade.reason == "SQUARE_OFF"
    assert trade.exit_ts.time() >= datetime(2026, 1, 5, 15, 15).time()
    assert broker.placed[-1]["transaction_type"] == "SELL" and broker.placed[-1]["order_type"] == "LIMIT"


async def test_signal_skipped_when_stop_risk_exceeds_daily_limit():
    def expensive(m):  # same shape, premium ~3x: 30% stop on 65 qty > Rs3000
        return v_shape(m) * 3

    report, broker, _ = await _run(expensive, live=True, broker_cls=RecordingBroker)
    assert report.trades == [] and broker.placed == []
    assert report.skipped and "remaining limit" in report.skipped[0]


async def test_no_entry_when_index_is_above_its_open():
    report, _, _ = await _run(v_shape, index_vs_open=+60)
    assert report.trades == [] and report.orders == []


async def test_gap_through_stop_is_chased_with_lower_limit():
    def crash(m):  # rally then a vertical drop far below the SL-limit price
        if m < 90:
            return v_shape(m)
        return 20

    report, broker, _ = await _run(crash, live=True, broker_cls=RecordingBroker)
    (trade,) = report.trades
    assert trade.reason == "STOP_LOSS"
    assert trade.exit <= Decimal("20")  # filled after the limit was lowered


def test_guard_refuses_other_contracts_and_oversize():
    guard = OrderGuard(DryRunBroker(), symbol="NIFTY2610624000PE", max_qty=65, log=lambda m: None)
    with pytest.raises(GuardError):
        guard._check("NIFTY2610624000CE", 65)
    with pytest.raises(GuardError):
        guard.place("BUY", 130, "LIMIT", Decimal("100"))
    assert guard.place("BUY", 65, "LIMIT", Decimal("100"))


class PreflightKite:
    def __init__(self, user="YUP337", positions=(), orders=()):
        self.user, self._pos, self._orders = user, list(positions), list(orders)

    def profile(self):
        return {"user_id": self.user}

    def positions(self):
        return {"net": self._pos}

    def orders(self):
        return self._orders


def test_preflight_blocks_wrong_account_ip_or_open_positions():
    ok = preflight(PreflightKite(), expected_user="YUP337", allowed_ips={"1.2.3.4"}, public_ip="1.2.3.4")
    assert ok.ok
    assert not preflight(
        PreflightKite(user="AB1234"), expected_user="YUP337", allowed_ips={"1.2.3.4"}, public_ip="1.2.3.4"
    ).ok
    assert not preflight(PreflightKite(), expected_user="YUP337", allowed_ips={"1.2.3.4"}, public_ip="9.9.9.9").ok
    assert not preflight(PreflightKite(), expected_user="YUP337", allowed_ips=set(), public_ip="1.2.3.4").ok
    held = PreflightKite(positions=[{"tradingsymbol": "NIFTYX", "exchange": "NFO", "quantity": 65}])
    assert not preflight(held, expected_user="YUP337", allowed_ips={"1.2.3.4"}, public_ip="1.2.3.4").ok
    working = PreflightKite(orders=[{"order_id": "1", "tag": "fnobot", "status": "TRIGGER PENDING"}])
    assert not preflight(working, expected_user="YUP337", allowed_ips={"1.2.3.4"}, public_ip="1.2.3.4").ok


async def test_strike_is_chosen_at_the_0915_open_not_at_startup():
    seen = []

    class GapKite(MarketKite):
        def ltp(self, keys):  # yesterday's close before 09:15, gap-down open after
            price = 24000 if self.clock().time() < datetime(2026, 1, 5, 9, 15).time() else 23700
            seen.append(self.clock().time())
            return {k: {"last_price": price} for k in keys}

    now = {"t": IST.localize(datetime(2026, 1, 5, 9, 0))}

    async def sleep(s):
        now["t"] += timedelta(seconds=s)

    await run_live_session(
        GapKite(lambda: now["t"], v_shape), LiveConfig(poll_seconds=15),
        clock=lambda: now["t"], sleep=sleep, log=lambda m: None,
    )  # fmt: skip
    assert seen[0] >= datetime(2026, 1, 5, 9, 15).time()


async def test_breakeven_raises_exchange_stop_to_entry_then_exits_flat():
    def pop_and_drop(m):  # dip, +25% pop after the entry, then a collapse
        if m < 30:
            return 150 - 2.5 * m
        if m < 45:
            return 75 + 3 * (m - 30)
        if m < 52:
            return 120 + 2 * (m - 45)
        return max(5, 134 - 6 * (m - 52))

    plain, _, _ = await _run(pop_and_drop, live=True, broker_cls=RecordingBroker, breakeven_pct=None)
    be, broker, _ = await _run(pop_and_drop, live=True, broker_cls=RecordingBroker)  # default +20%
    t_plain, t_be = plain.trades[0], be.trades[0]
    # Stop sat at the entry price; in a fast fall it fills within the SL-limit's 5% buffer
    # (here 104 vs 108), far above the 30%-trail level the plain run sold at.
    assert t_be.exit >= t_be.entry * Decimal("0.95")
    assert t_plain.exit < t_be.exit
    assert t_be.net > t_plain.net


def two_signals(m):  # 1st upturn on an expensive put (~Rs230), 2nd much later when it's ~Rs95
    if m < 30:
        return 300 - 3 * m
    if m < 50:
        return 210 + 4 * (m - 30)
    if m < 110:
        return 290 - 3.5 * (m - 50)
    return min(200, 80 + 3 * (m - 110))


async def test_skip_ends_the_day_by_default():
    report, broker, _ = await _run(two_signals, live=True, broker_cls=RecordingBroker)
    assert report.trades == [] and broker.placed == []
    assert len(report.skipped) == 1


async def test_retry_after_skip_takes_the_later_cheaper_signal():
    report, broker, _ = await _run(two_signals, live=True, broker_cls=RecordingBroker, retry_after_skip=True)
    assert len(report.skipped) >= 1
    (trade,) = report.trades  # still max 1 trade a day
    assert trade.entry < Decimal("150")
    assert trade.entry_ts.time() > datetime(2026, 1, 5, 10, 30).time()
