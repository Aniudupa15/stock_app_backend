"""Backtest the intraday F&O strategy on 1-minute candles.

Replays history through the SAME pieces the live paper session uses -
`default_fno_strategy` + `exits_for`, `StreamingIndicators`, the paper venue
(F&O charges, margin check, slippage), the OMS (SL / target / trailing) and
the risk gate (daily loss limit, lot sizing) - so the numbers describe what
`scripts/fno_paper_trade.py` would have done, not a simplified twin.

Per day: indicators start fresh at 09:15 (as the live script does), at most
`max_trades` entries, everything squared off at 15:15.

Price data and its limits (stated, not hidden):
  * FUT  - the underlying's 1-minute candles stand in for the near-month
           future. The futures basis (usually a small premium) is ignored; it
           barely moves intraday so P&L in points is very close.
  * CE/PE - Zerodha has no history for expired option contracts, so option
           prices are MODELLED with Black-Scholes from the underlying candle
           and India VIX as implied volatility, for the ATM strike of the
           nearest expiry picked at each day's open. Real premiums carry skew
           and wider spreads, so treat option results as indicative.
  * Each candle is replayed as open -> low/high -> close (the extreme nearer
    the open first), so a stop and a target inside the same minute resolve
    in the more pessimistic order when the candle moved against the trade.
  * Today's lot size is applied to the whole history.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal

from libs.backtest.metrics import BacktestMetrics, compute_metrics
from libs.derivatives.contracts import ContractRegistry, FnoContract, InstrumentKind
from libs.derivatives.margin import FnoMarginModel
from libs.engine.paper_engine import PaperTradingEngine
from libs.execution.paper import PaperExecutionVenue
from libs.execution.slippage import SlippageModel
from libs.oms.oms import OrderManagementSystem
from libs.risk.gate import RiskGate, RiskProfile
from libs.strategy.engine import StrategyEvaluator
from libs.trading_calendar.calendar import IST, TradingCalendar
from libs.trading_domain.entities import MarketQuote, Trade
from services.trading_service import fno_rules
from services.trading_service.fno_paper import PAPER_ACCOUNT_ID, StreamingIndicators, default_fno_strategy, exits_for

STRIKE_STEPS = {"NIFTY": 50, "BANKNIFTY": 100, "FINNIFTY": 50, "MIDCPNIFTY": 25, "NIFTYNXT50": 100, "SENSEX": 100}
# Weekly-expiry weekday (Mon=0). NSE moved NIFTY weekly expiry to Tuesday from
# Sep 2025; BSE SENSEX weeklies expire Thursday. Everything else: monthly.
WEEKLY_EXPIRY_WEEKDAY = {"NIFTY": 1, "SENSEX": 3}
MONTHLY_EXPIRY_WEEKDAY = 1  # last Tuesday of the month (NSE, from Sep 2025)
RISK_FREE_RATE = 0.065
_FAR_EXPIRY = date(2099, 12, 31)
_LARGE_VOLUME = 10**9  # index candles carry no volume; don't apply the illiquid penalty


@dataclass(frozen=True, slots=True)
class Candle:
    ts: datetime  # bar START, IST-aware
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal


@dataclass
class FnoBacktestConfig:
    underlying: str = "NIFTY"
    instrument: str = "FUT"  # FUT | CE | PE
    lots: int = 1
    lot_size: int = 75
    capital: Decimal = Decimal("500000")
    max_trades: int = 5
    max_daily_loss: Decimal | None = None  # default: 2% of capital
    square_off: time = time(15, 15)
    slippage_bps: Decimal | None = None  # default: 1 bp futures, 20 bp option premium
    strike_step: int | None = None
    candle_interval: str = "1m"  # bar size the strategy runs on (1m live default)
    # --- strategy variants (None = the live script's default for the contract) ---
    sl_pct: Decimal | None = None
    target_pct: Decimal | None = None
    no_target: bool = False  # ride winners: stop + trailing stop + 15:15 exit only
    trail_pct: Decimal | None = None
    trend_filter: str = "none"  # none | open | orb  (see _trend_ok)
    expiry: str = "near"  # near | next  (options: skip the nearest expiry)
    # Live bot rule: a signal whose stop-loss risk (qty x price x SL% + ~Rs60 charges)
    # exceeds this is SKIPPED and counts as the day's entry. None = no cap.
    max_trade_risk: Decimal | None = None
    breakeven_pct: Decimal | None = None  # stop -> entry once the option is up this %
    retry_after_skip: bool = False  # a too-expensive signal doesn't end the day; keep watching
    label: str = ""

    def describe(self) -> str:
        parts = [f"max {self.max_trades}/day"]
        if self.sl_pct is not None:
            parts.append(f"SL {self.sl_pct}%")
        parts.append(
            "no target" if self.no_target else (f"target {self.target_pct}%" if self.target_pct else "default target")
        )
        if self.trail_pct is not None:
            parts.append(f"trail {self.trail_pct}%")
        if self.retry_after_skip:
            parts.append("retry after skip")
        if self.breakeven_pct is not None:
            parts.append(f"breakeven at +{self.breakeven_pct}%")
        if self.trend_filter != "none":
            parts.append(f"trend={self.trend_filter}")
        if self.max_trade_risk is not None:
            parts.append(f"skip if risk > Rs{self.max_trade_risk:,.0f}")
        if self.instrument != "FUT" and self.expiry != "near":
            parts.append(f"expiry={self.expiry}")
        return ", ".join(parts)


@dataclass
class FnoBacktestResult:
    config: FnoBacktestConfig
    trades: list[Trade]
    equity_curve: list[tuple[datetime, Decimal]]
    metrics: BacktestMetrics
    total_charges: Decimal
    days: int
    first_day: date | None
    last_day: date | None
    daily_pnl: dict[date, Decimal] = field(default_factory=dict)
    skipped_signals: int = 0

    def by_exit_reason(self) -> dict[str, tuple[int, Decimal]]:
        out: dict[str, list] = defaultdict(lambda: [0, Decimal(0)])
        for t in self.trades:
            key = t.exit_reason.value if t.exit_reason else "?"
            out[key][0] += 1
            out[key][1] += t.pnl_net
        return {k: (v[0], v[1]) for k, v in out.items()}

    def extreme_days(self, n: int = 5) -> list[str]:
        """The n worst and n best days with dates, how many trades and the
        monthly P&L - to see what bad stretches actually look like."""
        days = sorted(self.daily_pnl.items(), key=lambda kv: kv[1])
        traded = [(d, p) for d, p in days if p != 0]
        if not traded:
            return []
        worst = [(d, p) for d, p in traded if p < 0][:n]
        best = [(d, p) for d, p in traded[::-1] if p > 0][:n]
        out = [f"Worst {n} days: " + (", ".join(f"{d} Rs{p:+,.0f}" for d, p in worst) or "none")]
        out.append(f"Best {n} days:  " + (", ".join(f"{d} Rs{p:+,.0f}" for d, p in best) or "none"))
        months: dict[str, Decimal] = defaultdict(Decimal)
        for d, p in self.daily_pnl.items():
            months[d.strftime("%Y-%m")] += p
        out.append("Months: " + "  ".join(f"{m} {v:+,.0f}" for m, v in sorted(months.items())))
        losing = sum(1 for v in months.values() if v < 0)
        out.append(f"Losing months: {losing} of {len(months)}   Worst month: Rs{min(months.values()):+,.0f}")
        return out

    def lines(self) -> list[str]:
        m, c = self.metrics, self.config
        green = sum(1 for v in self.daily_pnl.values() if v > 0)
        red = sum(1 for v in self.daily_pnl.values() if v < 0)
        worst = min(self.daily_pnl.values(), default=Decimal(0))
        best = max(self.daily_pnl.values(), default=Decimal(0))
        pf = f"{m.profit_factor:.2f}" if m.profit_factor is not None else "n/a"
        out = [
            f"Backtest: {c.underlying} {c.instrument}, {c.lots} lot(s) x {c.lot_size}, "
            f"capital Rs{c.capital:,.0f}, {c.candle_interval} bars",
            f"Variant: {c.label + ' - ' if c.label else ''}{c.describe()}",
            f"Period: {self.first_day} -> {self.last_day} ({self.days} trading days)",
            f"Trades: {m.total_trades}   Win rate: {m.win_rate:.1%}   Profit factor: {pf}",
            f"Net P&L (after charges): Rs{m.net_pnl:+,.2f}   Charges paid: Rs{self.total_charges:,.2f}",
            f"Avg win: Rs{m.avg_win:,.2f}   Avg loss: Rs{m.avg_loss:,.2f}   Expectancy/trade: Rs{m.expectancy:+,.2f}",
            f"Max drawdown: {m.max_drawdown_pct:.2%}   Sharpe: {m.sharpe:.2f}   Max losing streak: {m.max_consecutive_losses}",
            f"Days: {green} green / {red} red / {self.days - green - red} flat   Best day: Rs{best:+,.2f}   Worst day: Rs{worst:+,.2f}",
            *(
                [f"Signals skipped (stop risk > Rs{c.max_trade_risk:,.0f}): {self.skipped_signals}"]
                if c.max_trade_risk is not None
                else []
            ),
            "Exits: " + ", ".join(f"{k} {n} (Rs{p:+,.0f})" for k, (n, p) in sorted(self.by_exit_reason().items())),
        ]
        if c.instrument != "FUT":
            out.append("Option prices are Black-Scholes MODELLED from the index + India VIX - indicative only.")
        else:
            out.append("Futures prices are proxied by the index (basis ignored).")
        out.append("Past results do not guarantee future returns. Not investment advice.")
        out += self.extreme_days()
        return out


# ---------------------------------------------------------------- option model


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes(kind: InstrumentKind, spot: float, strike: float, years: float, vol: float) -> float:
    """European option value. At/after expiry -> intrinsic."""
    intrinsic = max(0.0, spot - strike) if kind is InstrumentKind.CE else max(0.0, strike - spot)
    if years <= 0 or vol <= 0:
        return intrinsic
    sqrt_t = math.sqrt(years)
    d1 = (math.log(spot / strike) + (RISK_FREE_RATE + 0.5 * vol * vol) * years) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    disc = math.exp(-RISK_FREE_RATE * years)
    if kind is InstrumentKind.CE:
        value = spot * _norm_cdf(d1) - strike * disc * _norm_cdf(d2)
    else:
        value = strike * disc * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
    return max(value, intrinsic * disc, 0.0)


def _tick(value: float, tick: Decimal = Decimal("0.05")) -> Decimal:
    """Round to the exchange tick, never below one tick (a quoted option
    always has a price)."""
    d = (Decimal(str(value)) / tick).quantize(Decimal(1), rounding=ROUND_HALF_UP) * tick
    return max(d, tick)


def expiry_for(underlying: str, d: date) -> date:
    """Nearest expiry on/after `d` (holiday shifts ignored)."""
    weekday = WEEKLY_EXPIRY_WEEKDAY.get(underlying)
    if weekday is not None:
        return d + timedelta(days=(weekday - d.weekday()) % 7)
    # Monthly: last `MONTHLY_EXPIRY_WEEKDAY` of this month, else next month's.
    for month_offset in (0, 1):
        y, m = d.year + (d.month - 1 + month_offset) // 12, (d.month - 1 + month_offset) % 12 + 1
        last = (date(y + (m // 12), m % 12 + 1, 1)) - timedelta(days=1)
        exp = last - timedelta(days=(last.weekday() - MONTHLY_EXPIRY_WEEKDAY) % 7)
        if exp >= d:
            return exp
    raise AssertionError("unreachable")


def _years_to_expiry(now: datetime, expiry: date) -> float:
    close = IST.localize(datetime.combine(expiry, time(15, 30)))
    return max(0.0, (close - now).total_seconds()) / (365.0 * 24 * 3600)


# ---------------------------------------------------------------- replay


class _ReplayQuotes:
    def __init__(self) -> None:
        self.prices: dict[str, Decimal] = {}

    async def get_quote(self, symbol: str) -> MarketQuote:
        return MarketQuote(symbol=symbol, ltp=self.prices[symbol], day_volume=_LARGE_VOLUME)


def _path(c: Candle) -> list[Decimal]:
    """Intra-candle price path: open, nearer extreme, farther extreme, close."""
    first, second = (c.low, c.high) if c.open - c.low <= c.high - c.open else (c.high, c.low)
    return [c.open, first, second, c.close]


def _vol_at(vix: list[tuple[datetime, Decimal]], ts: datetime, cursor: list[int]) -> float:
    """VIX as of `ts` (last value at or before it), walking a shared cursor."""
    i = cursor[0]
    while i + 1 < len(vix) and vix[i + 1][0] <= ts:
        i += 1
    cursor[0] = i
    return float(vix[i][1]) / 100.0


async def run_fno_backtest(
    candles: list[Candle],
    config: FnoBacktestConfig,
    *,
    vix: list[tuple[datetime, Decimal]] | None = None,
) -> FnoBacktestResult:
    kind = InstrumentKind(config.instrument.upper())
    if kind.is_option and not vix:
        raise ValueError("option backtests need India VIX candles for the volatility model")
    candles = sorted(candles, key=lambda c: c.ts)
    vix = sorted(vix or [], key=lambda v: v[0])
    vix_cursor = [0]

    clock_now = {"t": candles[0].ts if candles else datetime.now(IST)}

    def clock() -> datetime:
        return clock_now["t"]

    registry = ContractRegistry()
    quotes = _ReplayQuotes()
    bps = (
        config.slippage_bps
        if config.slippage_bps is not None
        else (Decimal(1) if kind is InstrumentKind.FUT else Decimal(20))
    )
    venue = PaperExecutionVenue(
        PAPER_ACCOUNT_ID,
        quotes,
        TradingCalendar(),
        clock,
        starting_cash=config.capital,
        slippage=SlippageModel(base_bps=bps, impact_coeff_bps=Decimal(0), illiquid_penalty_bps=Decimal(0)),
        contracts=registry,
    )
    oms = OrderManagementSystem(venue, clock)
    margin = FnoMarginModel()
    profile = RiskProfile(
        max_daily_loss=config.max_daily_loss or config.capital * Decimal("0.02"),
        max_open_positions=1,
        square_off_buffer_minutes=int(
            (datetime.combine(date.min, time(15, 30)) - datetime.combine(date.min, config.square_off)).seconds // 60
        ),
    )

    fut_contract = FnoContract(
        f"{config.underlying}-FUT", config.underlying, InstrumentKind.FUT, _FAR_EXPIRY, config.lot_size
    )
    if kind is InstrumentKind.FUT:
        registry.add(fut_contract)
    step = config.strike_step or STRIKE_STEPS.get(config.underlying, 50)

    by_day: dict[date, list[Candle]] = defaultdict(list)
    for c in candles:
        by_day[c.ts.astimezone(IST).date()].append(c)

    equity_curve: list[tuple[datetime, Decimal]] = []
    daily_pnl: dict[date, Decimal] = {}
    equity_before = config.capital
    skipped = 0

    for day in sorted(by_day):
        day_candles = by_day[day]
        if kind is InstrumentKind.FUT:
            contract = fut_contract
        else:
            spot0 = day_candles[0].open
            strike = (spot0 / step).quantize(Decimal(1), rounding=ROUND_HALF_UP) * step
            expiry = expiry_for(config.underlying, day)
            if config.expiry == "next":
                expiry = expiry_for(config.underlying, expiry + timedelta(days=1))
            contract = FnoContract(
                f"{config.underlying}{expiry:%y%m%d}{strike}{kind.value}",
                config.underlying,
                kind,
                expiry,
                config.lot_size,
                strike=strike,
            )
            registry.add(contract)

        strategy = replace(default_fno_strategy(config.lots), exit=_exit_rule(contract, config))
        evaluator = StrategyEvaluator()
        engine = PaperTradingEngine(
            PAPER_ACCOUNT_ID, venue, oms, RiskGate(), profile, TradingCalendar(), clock, [strategy],
            contracts=registry, margin_model=margin,
        )  # fmt: skip
        symbol = contract.tradingsymbol
        indicators = StreamingIndicators()
        prev_features: dict | None = None
        entries = 0
        day_open = day_candles[0].open
        or_end = (datetime.combine(day, time(9, 15)) + timedelta(minutes=OPENING_RANGE_MINUTES)).time()
        or_high, or_low = Decimal("-Infinity"), Decimal("Infinity")
        bullish = kind is not InstrumentKind.PE  # FUT/CE profit from a rise, PE from a fall

        def price(underlying_price: Decimal, at: datetime, contract=contract) -> Decimal:
            if contract.kind is InstrumentKind.FUT:
                return underlying_price
            vol = _vol_at(vix, at, vix_cursor)
            years = _years_to_expiry(at, contract.expiry)
            return _tick(black_scholes(contract.kind, float(underlying_price), float(contract.strike), years, vol))

        last_price: Decimal | None = None
        for c in day_candles:
            start = c.ts.astimezone(IST)
            if start.time() >= config.square_off:
                break
            if start.time() < or_end:
                or_high, or_low = max(or_high, c.high), min(or_low, c.low)
            path = _path(c)
            for i, underlying_px in enumerate(path[:-1]):
                clock_now["t"] = start + timedelta(seconds=15 * i)
                last_price = price(underlying_px, clock_now["t"])
                quotes.prices[symbol] = last_price
                await engine.process_tick(symbol, MarketQuote(symbol, last_price, day_volume=_LARGE_VOLUME))

            clock_now["t"] = start + timedelta(seconds=59)
            last_price = price(path[-1], clock_now["t"])
            quotes.prices[symbol] = last_price
            features = indicators.update(last_price)
            allow = entries < config.max_trades and _trend_ok(
                config.trend_filter, bullish, c.close, day_open, or_high, or_low, start.time() >= or_end
            )
            if (
                allow
                and config.max_trade_risk is not None
                and not oms.open_brackets(symbol)
                and evaluator.evaluate(strategy, symbol, features, prev_features, last_price, clock_now["t"])
            ):
                qty = config.lots * config.lot_size
                sl = strategy.exit.stop_loss_pct or Decimal(0)
                if qty * last_price * sl / 100 + Decimal(60) > config.max_trade_risk:
                    allow = False
                    if not config.retry_after_skip:
                        entries = config.max_trades  # as live: the skipped signal uses the day's entry
                    skipped += 1
            signal = await engine.process_tick(
                symbol,
                MarketQuote(symbol, last_price, day_volume=_LARGE_VOLUME),
                features if allow else None,
                prev_features if allow else None,
            )
            prev_features = features
            if signal is not None:
                entries += 1

        # 15:15 square-off at the last seen price.
        if oms.open_brackets(symbol) and last_price is not None:
            sq_time = IST.localize(datetime.combine(day, config.square_off))
            clock_now["t"] = max(clock_now["t"], sq_time)
            await engine.square_off(symbol)

        equity = venue.equity()
        eod = IST.localize(datetime.combine(day, time(15, 30)))
        equity_curve.append((eod, equity))
        daily_pnl[day] = equity - equity_before
        equity_before = equity

    trades = list(oms.trades)
    days = sorted(by_day)
    return FnoBacktestResult(
        config=config,
        trades=trades,
        equity_curve=equity_curve,
        metrics=compute_metrics(trades, equity_curve, config.capital),
        total_charges=sum((f.charges.total for f in venue.fills), Decimal(0)),
        days=len(days),
        first_day=days[0] if days else None,
        last_day=days[-1] if days else None,
        daily_pnl=daily_pnl,
        skipped_signals=skipped,
    )


OPENING_RANGE_MINUTES = fno_rules.OPENING_RANGE_MINUTES
_trend_ok = fno_rules.trend_ok


def _exit_rule(contract: FnoContract, config: FnoBacktestConfig):
    return fno_rules.override_exits(
        exits_for(contract),
        sl_pct=config.sl_pct,
        target_pct=config.target_pct,
        no_target=config.no_target,
        trail_pct=config.trail_pct,
        breakeven_pct=config.breakeven_pct,
    )


def sweep_variants(base: FnoBacktestConfig) -> list[FnoBacktestConfig]:
    """The comparison grid: baseline, then one idea at a time, then combos."""
    fut = base.instrument == "FUT"
    wide_sl, wide_trail = (Decimal("0.8"), Decimal("0.8")) if fut else (Decimal("30"), Decimal("30"))
    v = [
        ("baseline", {}),
        ("fewer trades", {"max_trades": 2}),
        ("with day trend", {"max_trades": 2, "trend_filter": "open"}),
        ("opening-range breakout", {"max_trades": 2, "trend_filter": "orb"}),
        ("ride winners", {"no_target": True, "sl_pct": wide_sl, "trail_pct": wide_trail}),
        ("ORB + ride winners, 1/day",
         {"max_trades": 1, "trend_filter": "orb", "no_target": True, "sl_pct": wide_sl, "trail_pct": wide_trail}),
        ("trend + ride winners, 1/day",
         {"max_trades": 1, "trend_filter": "open", "no_target": True, "sl_pct": wide_sl, "trail_pct": wide_trail}),
    ]  # fmt: skip
    if not fut:
        v += [
            ("next expiry, wide SL", {"expiry": "next", "sl_pct": wide_sl, "trail_pct": wide_trail}),
            ("next expiry + ORB + ride, 1/day",
             {"expiry": "next", "max_trades": 1, "trend_filter": "orb", "no_target": True,
              "sl_pct": wide_sl, "trail_pct": wide_trail}),
        ]  # fmt: skip
    return [replace(base, label=name, **kw) for name, kw in v]


def sweep_table(results: list[FnoBacktestResult]) -> list[str]:
    head = f"{'Variant':34} {'Trades':>6} {'Win%':>6} {'PF':>5} {'Net P&L':>12} {'Charges':>10} {'MaxDD':>6}"
    out = [head, "-" * len(head)]
    for r in results:
        m = r.metrics
        pf = f"{m.profit_factor:.2f}" if m.profit_factor is not None else "n/a"
        out.append(
            f"{r.config.label[:34]:34} {m.total_trades:>6} {m.win_rate:>6.1%} {pf:>5} "
            f"{m.net_pnl:>+12,.0f} {r.total_charges:>10,.0f} {m.max_drawdown_pct:>6.1%}"
        )
    return out


def target_variants(base: FnoBacktestConfig) -> list[FnoBacktestConfig]:
    """A/B of the exit only: everything else exactly as `base`."""
    out = [replace(base, label="no target (ride winners)", no_target=True, target_pct=None)]
    for pct in (30, 60, 100):
        out.append(replace(base, label=f"target {pct}%", no_target=False, target_pct=Decimal(pct)))
    return out


def breakeven_variants(base: FnoBacktestConfig) -> list[FnoBacktestConfig]:
    """A/B of the breakeven rule only: everything else exactly as `base`."""
    out = [replace(base, label="no breakeven (current)", breakeven_pct=None)]
    for pct in (20, 40, 60):
        out.append(replace(base, label=f"breakeven at +{pct}%", breakeven_pct=Decimal(pct)))
    return out


def retry_variants(base: FnoBacktestConfig) -> list[FnoBacktestConfig]:
    """A/B of what a too-expensive signal does: end the day vs keep watching."""
    return [
        replace(base, label="skip ends the day (current)", retry_after_skip=False),
        replace(base, label="keep watching after a skip", retry_after_skip=True),
    ]
