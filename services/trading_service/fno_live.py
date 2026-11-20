"""LIVE intraday F&O execution of a backtested preset (default: pe-trend).

Same DECISIONS as the backtest / paper session - EMA9/21 cross on the
option's own 1-minute bars with RSI > 50, the index-vs-open trend filter, one
entry a day, 30% stop, 30% trailing stop (₹2 step), no target, out by 15:15.

Different from paper in how it EXECUTES, deliberately:
  * The stop-loss is a real SL order resting AT THE EXCHANGE as soon as the
    entry fills, so it protects the position even if this PC, its internet
    or this script dies. (Zerodha blocks SL-M for options, so it is an SL
    limit order with a price buffer; if the market gaps through it, the
    script replaces it with an aggressive limit sell.)
  * MIS product: if everything else fails Zerodha auto-squares-off ~15:20.
  * Risk caps: max lots (default 1), a hard daily loss limit, and a trade is
    SKIPPED if its stop-loss risk would exceed what's left of that limit.

Safety rails - every order goes through `OrderGuard`, which only allows:
NFO, MIS, the one contract chosen today, quantity <= max lots, our tag.
Dry run (the default) sends NO orders: it simulates fills against live
prices with `DryRunBroker` and logs exactly what it would have sent.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from libs.broker.kite_quotes import KiteQuoteSource, load_fno_contracts
from libs.charges import nse_fno
from libs.charges.models import Side
from libs.derivatives.contracts import FnoContract, InstrumentKind
from libs.strategy.engine import StrategyEvaluator
from libs.trading_calendar.calendar import IST, TradingCalendar
from libs.trading_domain.entities import MarketQuote
from services.trading_service.fno_paper import (
    ReadOnlyKite,
    SessionConfig,
    StreamingIndicators,
    _TrendState,
    default_fno_strategy,
    pick_contract,
)

TAG = "fnobot"
_TICK = Decimal("0.05")


class GuardError(Exception):
    """An order the guard refuses to send (bug or misconfiguration)."""


def tick_down(x: Decimal, tick: Decimal = _TICK) -> Decimal:
    return (x / tick).to_integral_value(rounding=ROUND_FLOOR) * tick


def tick_up(x: Decimal, tick: Decimal = _TICK) -> Decimal:
    return (x / tick).to_integral_value(rounding=ROUND_CEILING) * tick


@dataclass
class LiveConfig:
    underlying: str = "NIFTY"
    instrument: str = "PE"
    expiry: str = "near"
    trend_filter: str = "open"
    lots: int = 1
    max_lots: int = 1
    max_trades: int = 1
    sl_pct: Decimal = Decimal("30")
    trail_pct: Decimal = Decimal("30")
    trail_step: Decimal = Decimal("2")  # rupees, as in the backtest
    # Lift the stop to the entry price once the put is up this % (backtest: better in
    # both halves of Nov 2025 - Nov 2026). None disables it.
    breakeven_pct: Decimal | None = Decimal("20")
    # A signal skipped for stop risk > limit ends the day (as backtested) unless this is set.
    retry_after_skip: bool = False
    daily_loss_limit: Decimal = Decimal("3000")
    bar_seconds: int = 60
    poll_seconds: int = 5
    square_off: time = time(15, 15)
    last_entry: time = time(15, 0)
    entry_buffer_pct: Decimal = Decimal("1")  # marketable-limit buy: up to 1% above ask
    sl_limit_buffer_pct: Decimal = Decimal("5")  # SL-limit sell price this far below trigger
    fill_timeout_s: int = 30
    max_sl_modifications: int = 20  # Zerodha caps modifications per order
    live: bool = False
    holidays: frozenset[date] = frozenset()


# ---------------------------------------------------------------- brokers


class OrderGuard:
    """The ONLY path to Zerodha's order API. Refuses anything outside the
    day's single contract/size, and records every order it sends."""

    def __init__(self, kite, *, symbol: str, max_qty: int, log: Callable[[str], None]) -> None:
        self._kite = kite
        self.symbol = symbol
        self.max_qty = max_qty
        self._log = log
        self.sent: list[dict] = []

    def _check(self, tradingsymbol: str, quantity: int) -> None:
        if tradingsymbol != self.symbol:
            raise GuardError(f"refusing order for {tradingsymbol} (today's contract is {self.symbol})")
        if not 0 < quantity <= self.max_qty:
            raise GuardError(f"refusing quantity {quantity} (max {self.max_qty})")

    def place(self, side: str, quantity: int, order_type: str, price: Decimal, trigger: Decimal | None = None) -> str:
        self._check(self.symbol, quantity)
        params = {
            "variety": "regular",
            "exchange": "NFO",
            "tradingsymbol": self.symbol,
            "transaction_type": side,
            "quantity": quantity,
            "product": "MIS",
            "order_type": order_type,
            "price": float(price),
            "validity": "DAY",
            "tag": TAG,
        }
        if trigger is not None:
            params["trigger_price"] = float(trigger)
        self._log(
            f"ORDER {side} {quantity} {self.symbol} {order_type} price {price}"
            + (f" trigger {trigger}" if trigger else "")
        )
        order_id = str(self._kite.place_order(**params))
        self.sent.append({"action": "place", "order_id": order_id, **params})
        return order_id

    def modify(self, order_id: str, *, price: Decimal, trigger: Decimal | None = None) -> None:
        kw = {"price": float(price)}
        if trigger is not None:
            kw["trigger_price"] = float(trigger)
        self._log(f"MODIFY {order_id} price {price}" + (f" trigger {trigger}" if trigger else ""))
        self._kite.modify_order("regular", order_id, **kw)
        self.sent.append({"action": "modify", "order_id": order_id, **kw})

    def cancel(self, order_id: str) -> None:
        self._log(f"CANCEL {order_id}")
        self._kite.cancel_order("regular", order_id)
        self.sent.append({"action": "cancel", "order_id": order_id})

    def status(self, order_id: str) -> dict:
        history = self._kite.order_history(order_id)
        return history[-1] if history else {}


class DryRunBroker:
    """Stands in for Kite's order API in dry run: fills against the live
    quote the session feeds it (`mark`). No network calls, no orders."""

    def __init__(self) -> None:
        self._orders: dict[str, dict] = {}
        self._n = 0
        self._ltp: Decimal | None = None

    def mark(self, ltp: Decimal) -> None:
        self._ltp = ltp
        for o in self._orders.values():
            if o["status"] == "TRIGGER PENDING" and ltp <= o["trigger"]:
                o["status"] = "OPEN"
            if o["status"] == "OPEN":
                self._try_fill(o)

    def _try_fill(self, o: dict) -> None:
        ltp = self._ltp
        if ltp is None:
            return
        if o["side"] == "BUY" and ltp <= o["price"]:
            o.update(status="COMPLETE", filled_quantity=o["qty"], average_price=float(ltp))
        elif o["side"] == "SELL" and ltp >= o["price"]:
            o.update(status="COMPLETE", filled_quantity=o["qty"], average_price=float(ltp))

    def place_order(self, **p) -> str:
        self._n += 1
        oid = f"DRY{self._n:04d}"
        status = "TRIGGER PENDING" if p["order_type"] == "SL" else "OPEN"
        self._orders[oid] = {
            "side": p["transaction_type"],
            "qty": p["quantity"],
            "price": Decimal(str(p["price"])),
            "trigger": Decimal(str(p.get("trigger_price") or 0)),
            "status": status,
            "filled_quantity": 0,
            "average_price": 0.0,
        }
        if status == "OPEN":
            self._try_fill(self._orders[oid])
        return oid

    def modify_order(self, variety, order_id, **kw) -> str:
        o = self._orders[order_id]
        if "price" in kw:
            o["price"] = Decimal(str(kw["price"]))
        if "trigger_price" in kw:
            o["trigger"] = Decimal(str(kw["trigger_price"]))
        if o["status"] == "OPEN":
            self._try_fill(o)
        return order_id

    def cancel_order(self, variety, order_id) -> str:
        if self._orders[order_id]["status"] in ("OPEN", "TRIGGER PENDING"):
            self._orders[order_id]["status"] = "CANCELLED"
        return order_id

    def order_history(self, order_id) -> list[dict]:
        o = self._orders[order_id]
        return [
            {
                "status": o["status"],
                "filled_quantity": o["filled_quantity"],
                "average_price": o["average_price"],
                "pending_quantity": o["qty"] - o["filled_quantity"],
            }
        ]


# ---------------------------------------------------------------- session


@dataclass
class LiveTrade:
    qty: int
    entry: Decimal
    exit: Decimal | None = None
    reason: str = ""
    charges: Decimal = Decimal(0)
    entry_ts: datetime | None = None
    exit_ts: datetime | None = None

    @property
    def net(self) -> Decimal:
        if self.exit is None:
            return Decimal(0)
        return (self.exit - self.entry) * self.qty - self.charges


@dataclass
class LiveReport:
    contract: FnoContract
    live: bool
    trades: list[LiveTrade] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    orders: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def net_pnl(self) -> Decimal:
        return sum((t.net for t in self.trades), Decimal(0))

    def lines(self) -> list[str]:
        mode = "LIVE - REAL ORDERS" if self.live else "DRY RUN - no orders were sent"
        out = [f"[{mode}] {self.contract.tradingsymbol} (lot {self.contract.lot_size}, expiry {self.contract.expiry})"]
        for t in self.trades:
            exit_txt = f"{t.exit:.2f}" if t.exit is not None else "OPEN"
            out.append(
                f"  {t.reason or 'open':10} qty {t.qty} {t.entry:.2f} -> {exit_txt}  "
                f"charges {t.charges:.2f}  net {t.net:+.2f}"
            )
        out += [f"  skipped: {s}" for s in self.skipped]
        out.append(f"Net P&L today (after charges): Rs{self.net_pnl:+,.2f}")
        out += self.notes
        return out


class _Ledger:
    """Today's realised P&L for the loss limit."""

    def __init__(self) -> None:
        self.realised = Decimal(0)


def _charges(kind: InstrumentKind, qty: int, entry: Decimal, exit_: Decimal) -> Decimal:
    return nse_fno.compute(Side.BUY, kind, qty, entry).total + nse_fno.compute(Side.SELL, kind, qty, exit_).total


async def run_live_session(
    kite,
    config: LiveConfig,
    *,
    clock: Callable[[], datetime] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    log: Callable[[str], None] = print,
    notify: Callable[[str, str], None] | None = None,
    broker=None,
) -> LiveReport:
    """One trading day. `kite` is the logged-in KiteConnect (market data is
    read through a read-only view). Orders go to `broker` - Kite itself when
    `config.live`, a DryRunBroker otherwise."""
    clock = clock or (lambda: datetime.now(IST))
    notify = notify or (lambda subject, body: None)
    data = ReadOnlyKite(kite)
    calendar = TradingCalendar(holidays=set(config.holidays))
    today = clock().date()

    # Pick the ATM strike from the 09:15 open, as the backtest does - before
    # that, NIFTY's LTP is still yesterday's close (wrong strike on gap days).
    if calendar.is_trading_day(today):
        open_at = calendar.continuous_window(today)[0]
        while clock().astimezone(IST).time() < open_at:
            await sleep(config.poll_seconds)

    contracts = await load_fno_contracts(data, underlyings={config.underlying})
    quotes = KiteQuoteSource(data, contracts)
    spot = (await quotes.spot_prices([config.underlying]))[config.underlying]
    contract = pick_contract(contracts, config.underlying, config.instrument, spot, today, config.expiry)
    qty = min(config.lots, config.max_lots) * contract.lot_size
    dry = None
    if broker is None:
        if config.live:
            broker = kite
        else:
            broker = dry = DryRunBroker()
    elif isinstance(broker, DryRunBroker):
        dry = broker
    guard = OrderGuard(broker, symbol=contract.tradingsymbol, max_qty=config.max_lots * contract.lot_size, log=log)
    report = LiveReport(contract, config.live)
    mode = "LIVE" if config.live else "DRY RUN"
    log(f"[{mode}] spot {config.underlying} {spot}; trading {contract.tradingsymbol} qty {qty}")
    notify(f"[{mode}] F&O bot started: {contract.tradingsymbol}", "\n".join(report.lines()[:1]))

    session = SessionConfig(underlying=config.underlying, trend_filter=config.trend_filter)
    trend = _TrendState(data, quotes, session, contract.kind is not InstrumentKind.PE, log)
    strategy = default_fno_strategy(config.lots)
    evaluator = StrategyEvaluator()
    indicators = StreamingIndicators()
    ledger = _Ledger()
    prev_features: dict | None = None
    next_bar_at = clock() + timedelta(seconds=config.bar_seconds)
    entries = 0

    state = "FLAT"  # FLAT -> ENTRY_PENDING -> IN_POSITION -> EXITING -> FLAT/DONE
    entry_id = sl_id = exit_id = None
    entry_placed_at = exit_placed_at = None
    sl_trigger = Decimal(0)
    sl_mods = exit_retries = 0
    triggered_since: datetime | None = None
    trade: LiveTrade | None = None

    def finish(exit_price: Decimal, reason: str, at: datetime) -> None:
        nonlocal trade, state
        assert trade is not None
        trade.exit, trade.reason, trade.exit_ts = exit_price, reason, at
        trade.charges = _charges(contract.kind, trade.qty, trade.entry, exit_price)
        ledger.realised += trade.net
        log(f"EXIT {reason} @ {exit_price}  net {trade.net:+.2f}  (today {ledger.realised:+.2f})")
        notify(f"[{mode}] EXIT {reason} {contract.tradingsymbol}: Rs{trade.net:+,.0f}", "\n".join(report.lines()))
        trade = None
        state = "FLAT"

    def place_exit(price: Decimal, at: datetime) -> None:
        nonlocal exit_id, exit_placed_at, state
        exit_id = guard.place("SELL", trade.qty, "LIMIT", tick_down(price))
        exit_placed_at, state = at, "EXITING"

    while True:
        now = clock()
        t = now.astimezone(IST).time()
        if not calendar.is_trading_day(now.date()):
            report.notes.append("Not a trading day - nothing done.")
            break
        if t < calendar.continuous_window(now.date())[0]:
            await sleep(config.poll_seconds)
            continue
        if state in ("FLAT", "DONE") and (t >= config.square_off or not calendar.is_open(now)):
            break
        if not calendar.is_open(now):  # 15:30 close with an exit still working
            report.notes.append("Market closed while an order was still working - check Kite.")
            break

        try:
            quote: MarketQuote = await quotes.get_quote(contract.tradingsymbol)
        except Exception as exc:
            log(f"quote error: {exc}")
            await sleep(config.poll_seconds)
            continue
        ltp = quote.ltp
        if dry is not None:
            dry.mark(ltp)

        try:
            # ---------------- order management
            if state == "ENTRY_PENDING":
                st = guard.status(entry_id)
                filled = int(st.get("filled_quantity") or 0)
                if st.get("status") == "COMPLETE" or (
                    now - entry_placed_at >= timedelta(seconds=config.fill_timeout_s) and filled > 0
                ):
                    if st.get("status") != "COMPLETE":
                        guard.cancel(entry_id)  # keep the partial fill only
                    avg = Decimal(str(st.get("average_price") or ltp))
                    trade = LiveTrade(filled, avg, entry_ts=now)
                    report.trades.append(trade)
                    sl_trigger = tick_down(avg * (1 - config.sl_pct / 100))
                    sl_price = tick_down(sl_trigger * (1 - config.sl_limit_buffer_pct / 100))
                    sl_id = guard.place("SELL", filled, "SL", max(sl_price, _TICK), sl_trigger)
                    sl_mods, triggered_since, state = 0, None, "IN_POSITION"
                    log(f"FILLED {filled} @ {avg}; stop-loss at exchange: trigger {sl_trigger}")
                    notify(
                        f"[{mode}] ENTRY {contract.tradingsymbol} {filled} @ {avg}",
                        f"Stop-loss resting at the exchange: trigger {sl_trigger}.",
                    )
                elif st.get("status") in ("REJECTED", "CANCELLED"):
                    msg = f"entry {st.get('status')}: {st.get('status_message', '')}"
                    log(msg)
                    report.notes.append(msg)
                    notify(f"[{mode}] Entry {st.get('status')} - stopping for today", msg)
                    state = "DONE"
                elif now - entry_placed_at >= timedelta(seconds=config.fill_timeout_s):
                    guard.cancel(entry_id)
                    report.skipped.append(f"{now:%H:%M} entry not filled in {config.fill_timeout_s}s - cancelled")
                    state = "FLAT"

            elif state == "IN_POSITION":
                st = guard.status(sl_id)
                if st.get("status") == "COMPLETE":
                    finish(Decimal(str(st.get("average_price"))), "STOP_LOSS", now)
                elif st.get("status") in ("REJECTED", "CANCELLED"):
                    # Stop vanished (rejected/cancelled outside the bot): never sit unprotected.
                    log(f"stop-loss order {st.get('status')} - exiting at market price")
                    place_exit(ltp * Decimal("0.97"), now)
                elif t >= config.square_off:
                    guard.cancel(sl_id)
                    place_exit((quote.bid or ltp) * Decimal("0.98"), now)
                else:
                    if st.get("status") == "OPEN":  # triggered, limit not filled (gap)
                        triggered_since = triggered_since or now
                        if now - triggered_since >= timedelta(seconds=10):
                            guard.modify(sl_id, price=tick_down(ltp * Decimal("0.95")))
                            triggered_since = now
                    else:
                        trail = tick_down(ltp * (1 - config.trail_pct / 100))
                        candidate = trail if trail - sl_trigger >= config.trail_step else None
                        if config.breakeven_pct is not None and ltp >= trade.entry * (1 + config.breakeven_pct / 100):
                            be_level = tick_down(trade.entry)
                            if be_level > sl_trigger and (candidate is None or be_level > candidate):
                                candidate = be_level
                                log(f"BREAKEVEN: put up {config.breakeven_pct}% - stop raised to entry {be_level}")
                        if candidate is not None:
                            sl_trigger = candidate
                            new_price = max(tick_down(candidate * (1 - config.sl_limit_buffer_pct / 100)), _TICK)
                            if sl_mods < config.max_sl_modifications:
                                guard.modify(sl_id, price=new_price, trigger=candidate)
                                sl_mods += 1
                            else:
                                # Zerodha caps modifications per order: replace the stop with a
                                # fresh one (cancel first - never two sell orders at once).
                                guard.cancel(sl_id)
                                sl_id = guard.place("SELL", trade.qty, "SL", new_price, candidate)
                                sl_mods = 0

            elif state == "EXITING":
                st = guard.status(exit_id)
                if st.get("status") == "COMPLETE":
                    finish(
                        Decimal(str(st.get("average_price"))), "SQUARE_OFF" if t >= config.square_off else "EXIT", now
                    )
                    if t >= config.square_off:
                        state = "DONE"
                elif now - exit_placed_at >= timedelta(seconds=15):
                    exit_retries += 1
                    guard.modify(exit_id, price=tick_down(ltp * (Decimal(1) - Decimal("0.02") * (exit_retries + 1))))
                    exit_placed_at = now
                    if exit_retries == 5:
                        notify(f"[{mode}] EXIT NOT FILLING {contract.tradingsymbol}", "Check Kite now.")

            # ---------------- signals (bar close)
            features = None
            if now >= next_bar_at:
                features = indicators.update(ltp)
                next_bar_at = now + timedelta(seconds=config.bar_seconds)
            signal = (
                features is not None
                and state == "FLAT"
                and entries < config.max_trades
                and t < config.last_entry
                and ledger.realised > -config.daily_loss_limit
                and evaluator.evaluate(strategy, contract.tradingsymbol, features, prev_features, ltp, now) is not None
            )
            if signal and not await trend.allows(now):
                side = "above" if contract.kind is InstrumentKind.PE else "below"
                msg = (
                    f"{now:%H:%M} signal @ {ltp} ignored: trend filter "
                    f"(NIFTY {side} its open {trend.day_open if trend.day_open is not None else '?'})"
                )
                log(msg)
                report.skipped.append(msg)
                signal = False
            if signal:
                risk = qty * ltp * config.sl_pct / 100 + Decimal(60)  # stop distance + ~charges
                budget = config.daily_loss_limit + min(ledger.realised, Decimal(0))
                if risk <= budget or not config.retry_after_skip:
                    entries += 1  # a skip uses the day's entry unless retry_after_skip
                if risk > budget:
                    msg = f"{now:%H:%M} signal @ {ltp} skipped: stop-loss risk Rs{risk:,.0f} > remaining limit Rs{budget:,.0f}"
                    log(msg)
                    report.skipped.append(msg)
                else:
                    limit = tick_up((quote.ask or ltp) * (1 + config.entry_buffer_pct / 100))
                    entry_id = guard.place("BUY", qty, "LIMIT", limit)
                    entry_placed_at, state = now, "ENTRY_PENDING"
                    if dry is not None:
                        dry.mark(ltp)
            if features is not None:
                prev_features = features
        except GuardError:
            raise
        except Exception as exc:  # broker hiccup: alert, keep the exchange stop, retry next poll
            log(f"broker error: {type(exc).__name__}: {exc}")
            notify(
                f"[{mode}] F&O bot error",
                f"{type(exc).__name__}: {exc}\nThe exchange stop-loss (if any) is still active.",
            )

        if state == "DONE" and t >= config.square_off:
            break
        await sleep(config.poll_seconds)

    if not report.trades and not report.skipped and "Not a trading day" not in " ".join(report.notes):
        report.notes.append("No entry signal today (the put's EMA 9/21 + RSI conditions never lined up).")
    if trade is not None and trade.exit is None:
        report.notes.append(
            "Position still open at the end - the exchange stop-loss is active and Zerodha auto-squares MIS ~15:20."
        )
        notify(f"[{mode}] POSITION LEFT OPEN {contract.tradingsymbol}", "Check Kite now.")
    report.orders = guard.sent
    notify(f"[{mode}] F&O day summary: Rs{report.net_pnl:+,.0f}", "\n".join(report.lines()))
    return report


# ---------------------------------------------------------------- pre-flight


@dataclass
class Preflight:
    ok: bool
    lines: list[str]


def preflight(kite, *, expected_user: str | None, allowed_ips: set[str], public_ip: str | None) -> Preflight:
    """Checks that must ALL pass before a live session may send an order."""
    lines: list[str] = []
    ok = True

    def check(passed: bool, text: str) -> None:
        nonlocal ok
        ok = ok and passed
        lines.append(f"[{'PASS' if passed else 'FAIL'}] {text}")

    user = kite.profile().get("user_id", "?")
    check(bool(expected_user) and user.upper() == (expected_user or "").upper(),
          f"account: logged in as {user}, KITE_USER_ID={expected_user or 'NOT SET'}")  # fmt: skip
    check(bool(allowed_ips) and public_ip in allowed_ips,
          f"static IP: this PC is {public_ip or 'unknown'}, KITE_ALLOWED_IPS={','.join(sorted(allowed_ips)) or 'NOT SET'}")  # fmt: skip
    open_fno = [
        p["tradingsymbol"]
        for p in kite.positions().get("net", [])
        if p.get("exchange") in ("NFO", "BFO") and p.get("quantity")
    ]
    check(not open_fno, f"no open F&O positions ({', '.join(open_fno) or 'none'})")
    pending = [
        o["order_id"] for o in kite.orders() if o.get("tag") == TAG and o.get("status") in ("OPEN", "TRIGGER PENDING")
    ]
    check(not pending, f"no working bot orders ({', '.join(pending) or 'none'})")
    return Preflight(ok, lines)
