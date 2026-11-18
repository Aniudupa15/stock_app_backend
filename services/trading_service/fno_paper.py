"""Intraday F&O PAPER trading session against live Zerodha quotes.

What it does, per session (one trading day):
  1. picks a contract on the chosen underlying - nearest future, or the ATM
     call/put of the nearest expiry - from Zerodha's instrument master;
  2. polls `kite.quote()` every `poll_seconds`, feeding each quote into the
     paper venue (so resting SL/target orders fill on the real price path);
  3. every `bar_seconds` closes a bar, updates EMA/RSI features and lets the
     strategy decide on an entry (risk-gated, lot-sized, margin-checked);
  4. squares everything off at the MIS cut-off (15:15 IST by default) and
     returns the journal: fills, round-trip trades, P&L after charges.

NO REAL ORDERS: Kite is wrapped in `ReadOnlyKite`, which exposes only
market-data/auth methods - `place_order` etc. do not exist on it, so even a
bug can't reach Zerodha's order API. Execution is `PaperExecutionVenue`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from uuid import NAMESPACE_URL, uuid5

from libs.broker.kite_quotes import KiteQuoteSource, load_fno_contracts, spot_key
from libs.derivatives.contracts import ContractRegistry, FnoContract, InstrumentKind
from libs.derivatives.margin import FnoMarginModel
from libs.engine.paper_engine import PaperTradingEngine
from libs.execution.paper import PaperExecutionVenue
from libs.oms.oms import OrderManagementSystem
from libs.risk.gate import RiskGate, RiskProfile
from libs.strategy.engine import ExitRule, Strategy
from libs.trading_calendar.calendar import IST, TradingCalendar
from libs.trading_domain.entities import Fill, Trade, TrailingSpec
from libs.trading_domain.enums import Product, Side
from services.trading_service.fno_rules import OPENING_RANGE_MINUTES, override_exits, trend_ok

PAPER_ACCOUNT_ID = uuid5(NAMESPACE_URL, "stock-app/fno-paper")


class ReadOnlyKite:
    """Whitelist proxy around KiteConnect: market data + login only."""

    _ALLOWED = frozenset(
        {
            "quote",
            "ltp",
            "ohlc",
            "historical_data",
            "instruments",
            "profile",
            "margins",
            "login_url",
            "generate_session",
            "set_access_token",
        }
    )

    def __init__(self, kite) -> None:
        object.__setattr__(self, "_kite", kite)

    def __getattr__(self, name: str):
        if name not in self._ALLOWED:
            raise AttributeError(f"ReadOnlyKite blocks '{name}' - paper mode never touches the order API")
        return getattr(self._kite, name)


# ---------------------------------------------------------------- features


class StreamingIndicators:
    """EMA(fast), EMA(slow), RSI(14, Wilder) over closed bars."""

    def __init__(self, fast: int = 9, slow: int = 21, rsi_period: int = 14) -> None:
        self.fast, self.slow, self.rsi_period = fast, slow, rsi_period
        self._ema: dict[int, Decimal | None] = {fast: None, slow: None}
        self._prev_close: Decimal | None = None
        self._seed: list[Decimal] = []  # first `rsi_period` price changes
        self._avg_gain: Decimal | None = None
        self._avg_loss: Decimal | None = None
        self.bars = 0

    def update(self, close: Decimal) -> dict:
        self.bars += 1
        for period in (self.fast, self.slow):
            prev = self._ema[period]
            k = Decimal(2) / Decimal(period + 1)
            self._ema[period] = close if prev is None else prev + k * (close - prev)

        if self._prev_close is not None:
            change = close - self._prev_close
            gain, loss = max(change, Decimal(0)), max(-change, Decimal(0))
            n = Decimal(self.rsi_period)
            if self._avg_gain is None or self._avg_loss is None:
                self._seed.append(change)
                if len(self._seed) == self.rsi_period:  # simple-average seed
                    self._avg_gain = sum((max(d, Decimal(0)) for d in self._seed), Decimal(0)) / n
                    self._avg_loss = sum((max(-d, Decimal(0)) for d in self._seed), Decimal(0)) / n
            else:  # Wilder smoothing
                self._avg_gain = (self._avg_gain * (n - 1) + gain) / n
                self._avg_loss = (self._avg_loss * (n - 1) + loss) / n
        self._prev_close = close

        features: dict = {"close": close}
        # Only publish an EMA once it has seen `period` bars (warm-up).
        if self.bars >= self.fast:
            features[f"EMA_{self.fast}"] = self._ema[self.fast]
        if self.bars >= self.slow:
            features[f"EMA_{self.slow}"] = self._ema[self.slow]
        if self._avg_gain is not None and self._avg_loss is not None:
            if self._avg_loss == 0:
                features["RSI_14"] = Decimal(100)
            else:
                rs = self._avg_gain / self._avg_loss
                features["RSI_14"] = Decimal(100) - Decimal(100) / (1 + rs)
        # 2 dp keeps signal reasoning readable; crossovers are decided on these.
        return {k: v.quantize(Decimal("0.01")) for k, v in features.items()}


def default_fno_strategy(lots: int, product: Product = Product.MIS) -> Strategy:
    """Long-only trend entry on the TRADED contract's own price:
    EMA9 crosses above EMA21 with RSI14 > 50. For a PE this means "put
    premium trending up" i.e. a bearish view on the underlying.
    Exits: 1% SL / 2% target on futures-scale moves are too tight for option
    premiums, so the caller passes contract-appropriate exits."""
    return Strategy(
        name="F&O EMA 9/21 crossover",
        rule={
            "op": "AND",
            "nodes": [
                {"op": "CROSS_ABOVE", "left": {"feature": "EMA_9"}, "right": {"feature": "EMA_21"}},
                {"op": "GT", "left": {"feature": "RSI_14"}, "right": {"const": 50}},
            ],
        },
        side=Side.BUY,
        product=product,
        quantity=lots,
        timeframe="intraday",
    )


def exits_for(contract: FnoContract) -> ExitRule:
    if contract.kind is InstrumentKind.FUT:
        return ExitRule(
            stop_loss_pct=Decimal("0.5"),
            target_pct=Decimal("1"),
            trailing=TrailingSpec(by="PCT", value=Decimal("0.5"), step=Decimal("0.1")),
        )
    # Option premiums move ~5-10x the underlying in % terms.
    return ExitRule(
        stop_loss_pct=Decimal("15"),
        target_pct=Decimal("30"),
        trailing=TrailingSpec(by="PCT", value=Decimal("15"), step=Decimal("2")),
    )


# ---------------------------------------------------------------- session


@dataclass
class SessionConfig:
    underlying: str = "NIFTY"
    instrument: str = "FUT"  # FUT | CE | PE
    lots: int = 1
    capital: Decimal = Decimal("500000")
    bar_seconds: int = 60
    poll_seconds: int = 5
    square_off_buffer_minutes: int = 15
    max_daily_loss: Decimal | None = None  # default: 2% of capital
    max_trades: int = 5
    holidays: frozenset[date] = frozenset()
    # Strategy variant (same meaning as in the backtest; see fno_rules.PRESETS)
    sl_pct: Decimal | None = None
    target_pct: Decimal | None = None
    no_target: bool = False
    trail_pct: Decimal | None = None
    trend_filter: str = "none"  # none | open | orb  (judged on the live index price)
    expiry: str = "near"  # near | next  (options)
    breakeven_pct: Decimal | None = None


@dataclass
class SessionReport:
    contract: FnoContract
    starting_cash: Decimal
    final_cash: Decimal
    fills: list[Fill] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    bars: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def net_pnl(self) -> Decimal:
        return self.final_cash - self.starting_cash

    @property
    def total_charges(self) -> Decimal:
        return sum((f.charges.total for f in self.fills), Decimal(0))

    def lines(self) -> list[str]:
        c = self.contract
        out = [
            f"Contract: {c.tradingsymbol} ({c.kind.value}, lot {c.lot_size}, expiry {c.expiry})",
            f"Bars: {self.bars}   Fills: {len(self.fills)}   Round trips: {len(self.trades)}",
        ]
        for t in self.trades:
            out.append(
                f"  {t.exit_reason.value if t.exit_reason else '?':10} qty {t.qty} "
                f"{t.entry_price:.2f} -> {t.exit_price:.2f}  gross {t.pnl_gross:+.2f}  "
                f"charges {t.charges_total:.2f}  net {t.pnl_net:+.2f}"
            )
        out += [
            f"Charges paid: Rs{self.total_charges:,.2f}",
            f"Net P&L (after charges): Rs{self.net_pnl:+,.2f}",
            f"Paper cash: Rs{self.starting_cash:,.2f} -> Rs{self.final_cash:,.2f}",
            *self.notes,
            "PAPER TRADING - no real orders were placed.",
        ]
        return out


def pick_contract(
    contracts: ContractRegistry,
    underlying: str,
    instrument: str,
    spot: Decimal,
    today: date,
    expiry: str = "near",
) -> FnoContract:
    kind = InstrumentKind(instrument.upper())
    if kind is InstrumentKind.FUT:
        contract = contracts.nearest_future(underlying, today=today)
    else:
        chosen = None
        if expiry == "next":
            expiries = contracts.expiries(underlying, kind, on_or_after=today)
            if len(expiries) < 2:
                raise ValueError(f"no next expiry listed for {underlying} {kind.value}")
            chosen = expiries[1]
        contract = contracts.atm_option(underlying, kind, spot, today=today, expiry=chosen)
    if contract is None:
        raise ValueError(f"no live {kind.value} contract found for {underlying}")
    return contract


async def run_session(
    kite,
    config: SessionConfig,
    *,
    clock: Callable[[], datetime] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    log: Callable[[str], None] = print,
) -> SessionReport:
    """Run one intraday paper session until the square-off cut-off."""
    kite = kite if isinstance(kite, ReadOnlyKite) else ReadOnlyKite(kite)
    clock = clock or (lambda: datetime.now(IST))
    calendar = TradingCalendar(holidays=set(config.holidays))
    today = clock().date()

    contracts = await load_fno_contracts(kite, underlyings={config.underlying})
    quotes = KiteQuoteSource(kite, contracts)
    spot = (await quotes.spot_prices([config.underlying])).get(config.underlying)
    if spot is None:
        raise ValueError(f"no spot price for {config.underlying}")
    contract = pick_contract(contracts, config.underlying, config.instrument, spot, today, config.expiry)
    log(f"Spot {config.underlying} = {spot}; paper-trading {contract.tradingsymbol} (lot {contract.lot_size})")

    margin = FnoMarginModel()
    venue = PaperExecutionVenue(
        PAPER_ACCOUNT_ID,
        quotes,
        calendar,
        clock,
        starting_cash=config.capital,
        contracts=contracts,
        margin_model=margin,
    )
    oms = OrderManagementSystem(venue, clock)
    profile = RiskProfile(
        max_daily_loss=config.max_daily_loss or config.capital * Decimal("0.02"),
        max_open_positions=1,
        square_off_buffer_minutes=config.square_off_buffer_minutes,
    )
    exit_rule = override_exits(
        exits_for(contract),
        sl_pct=config.sl_pct,
        target_pct=config.target_pct,
        no_target=config.no_target,
        trail_pct=config.trail_pct,
        breakeven_pct=config.breakeven_pct,
    )
    strategy = replace(default_fno_strategy(config.lots), exit=exit_rule)
    trend = _TrendState(kite, quotes, config, contract.kind is not InstrumentKind.PE, log)
    engine = PaperTradingEngine(
        PAPER_ACCOUNT_ID,
        venue,
        oms,
        RiskGate(),
        profile,
        calendar,
        clock,
        [strategy],
        contracts=contracts,
        margin_model=margin,
    )

    indicators = StreamingIndicators()
    prev_features: dict | None = None
    bar_close: Decimal | None = None
    next_bar_at = clock() + timedelta(seconds=config.bar_seconds)
    symbol = contract.tradingsymbol
    report = SessionReport(contract, config.capital, config.capital)
    entries = 0

    while True:
        now = clock()
        if not calendar.is_trading_day(now.date()):
            report.notes.append("Not a trading day - nothing done.")
            break
        if now.astimezone(IST).time() < calendar.continuous_window(now.date())[0]:
            await sleep(config.poll_seconds)  # before the open: wait for it
            continue
        if not calendar.is_open(now) or calendar.in_square_off_window(now, config.square_off_buffer_minutes):
            break  # MIS cut-off reached (or market closed)

        try:
            quote = await quotes.get_quote(symbol)
        except Exception as exc:  # transient data hiccup: skip this poll
            log(f"quote error: {exc}")
            await sleep(config.poll_seconds)
            continue
        bar_close = quote.ltp

        features = None
        if now >= next_bar_at and bar_close is not None:
            features = indicators.update(bar_close)
            report.bars += 1
            next_bar_at = now + timedelta(seconds=config.bar_seconds)

        allow_entry = features is not None and entries < config.max_trades
        if allow_entry and config.trend_filter != "none":
            allow_entry = await trend.allows(now)
        signal = await engine.process_tick(
            symbol, quote, features if allow_entry else None, prev_features if allow_entry else None
        )
        if features is not None:
            prev_features = features
        if signal is not None:
            entries += 1
            log(f"ENTRY {signal.reasoning}  SL {signal.stop_loss}  target {signal.targets}")
        for trade in oms.trades[len(report.trades) :]:
            report.trades.append(trade)
            log(f"EXIT {trade.exit_reason.value if trade.exit_reason else ''} net {trade.pnl_net:+.2f}")
        await sleep(config.poll_seconds)

    # Square off anything still open (MIS cut-off / end of session).
    try:
        quote = await quotes.get_quote(symbol)
        await venue.on_tick(quote)
    except Exception as exc:
        log(f"final quote error: {exc}")
    if oms.open_brackets(symbol):
        if calendar.is_open(clock()):
            await engine.square_off(symbol)
        else:
            report.notes.append("Market already closed with a position open - left unsquared in paper book.")
    report.trades = list(oms.trades)
    report.fills = list(venue.fills)
    report.final_cash = venue.equity()
    return report


class _TrendState:
    """Live inputs for the trend filter: today's index open (from Kite's OHLC)
    and the 15-minute opening range (from Kite's 1-minute history once the
    range is complete - so starting the script late still gets it right)."""

    def __init__(self, kite, quotes: KiteQuoteSource, config: SessionConfig, bullish: bool, log) -> None:
        self._kite, self._quotes, self._config, self._bullish, self._log = kite, quotes, config, bullish, log
        self._key = spot_key(config.underlying)
        self.day_open: Decimal | None = None
        self.or_high: Decimal | None = None
        self.or_low: Decimal | None = None

    async def _load_open(self) -> None:
        raw = await asyncio.to_thread(self._kite.ohlc, [self._key])
        value = (raw.get(self._key) or {}).get("ohlc", {}).get("open")
        if value:
            self.day_open = Decimal(str(value))
            self._log(f"{self._config.underlying} opened at {self.day_open}")

    async def _load_opening_range(self, now: datetime) -> None:
        start = IST.localize(datetime.combine(now.date(), time(9, 15)))
        end = start + timedelta(minutes=OPENING_RANGE_MINUTES)
        token = next(
            int(r["instrument_token"])
            for r in await asyncio.to_thread(self._kite.instruments, self._key.split(":")[0])
            if r["tradingsymbol"] == self._key.split(":", 1)[1]
        )
        rows = await asyncio.to_thread(self._kite.historical_data, token, start, end - timedelta(seconds=1), "minute")
        if rows:
            self.or_high = max(Decimal(str(r["high"])) for r in rows)
            self.or_low = min(Decimal(str(r["low"])) for r in rows)
            self._log(f"Opening range {self.or_low} - {self.or_high}")

    async def allows(self, now: datetime) -> bool:
        mode = self._config.trend_filter
        or_done = (
            now.astimezone(IST).time()
            >= (datetime.combine(now.date(), time(9, 15)) + timedelta(minutes=OPENING_RANGE_MINUTES)).time()
        )
        try:
            if self.day_open is None:
                await self._load_open()
            if mode == "orb" and or_done and self.or_high is None:
                await self._load_opening_range(now)
            spot = (await self._quotes.spot_prices([self._config.underlying]))[self._config.underlying]
        except Exception as exc:  # no data -> no entry (never trade blind)
            self._log(f"trend filter data error: {exc}")
            return False
        if self.day_open is None or (mode == "orb" and (self.or_high is None or self.or_low is None)):
            return False
        return trend_ok(
            mode, self._bullish, spot, self.day_open, self.or_high or Decimal(0), self.or_low or Decimal(0), or_done
        )
