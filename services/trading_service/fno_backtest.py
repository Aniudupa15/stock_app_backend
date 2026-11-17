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
from libs.trading_calendar.calendar import IST, TradingCalendar
from libs.trading_domain.entities import MarketQuote, Trade
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

    def by_exit_reason(self) -> dict[str, tuple[int, Decimal]]:
        out: dict[str, list] = defaultdict(lambda: [0, Decimal(0)])
        for t in self.trades:
            key = t.exit_reason.value if t.exit_reason else "?"
            out[key][0] += 1
            out[key][1] += t.pnl_net
        return {k: (v[0], v[1]) for k, v in out.items()}

    def lines(self) -> list[str]:
        m, c = self.metrics, self.config
        green = sum(1 for v in self.daily_pnl.values() if v > 0)
        red = sum(1 for v in self.daily_pnl.values() if v < 0)
        worst = min(self.daily_pnl.values(), default=Decimal(0))
        best = max(self.daily_pnl.values(), default=Decimal(0))
        pf = f"{m.profit_factor:.2f}" if m.profit_factor is not None else "n/a"
        out = [
            f"Backtest: {c.underlying} {c.instrument}, {c.lots} lot(s) x {c.lot_size}, capital Rs{c.capital:,.0f}",
            f"Period: {self.first_day} -> {self.last_day} ({self.days} trading days)",
            f"Trades: {m.total_trades}   Win rate: {m.win_rate:.1%}   Profit factor: {pf}",
            f"Net P&L (after charges): Rs{m.net_pnl:+,.2f}   Charges paid: Rs{self.total_charges:,.2f}",
            f"Avg win: Rs{m.avg_win:,.2f}   Avg loss: Rs{m.avg_loss:,.2f}   Expectancy/trade: Rs{m.expectancy:+,.2f}",
            f"Max drawdown: {m.max_drawdown_pct:.2%}   Sharpe: {m.sharpe:.2f}   Max losing streak: {m.max_consecutive_losses}",
            f"Days: {green} green / {red} red / {self.days - green - red} flat   Best day: Rs{best:+,.2f}   Worst day: Rs{worst:+,.2f}",
            "Exits: " + ", ".join(f"{k} {n} (Rs{p:+,.0f})" for k, (n, p) in sorted(self.by_exit_reason().items())),
        ]
        if c.instrument != "FUT":
            out.append("Option prices are Black-Scholes MODELLED from the index + India VIX - indicative only.")
        else:
            out.append("Futures prices are proxied by the index (basis ignored).")
        out.append("Past results do not guarantee future returns. Not investment advice.")
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

    for day in sorted(by_day):
        day_candles = by_day[day]
        if kind is InstrumentKind.FUT:
            contract = fut_contract
        else:
            spot0 = day_candles[0].open
            strike = (spot0 / step).quantize(Decimal(1), rounding=ROUND_HALF_UP) * step
            expiry = expiry_for(config.underlying, day)
            contract = FnoContract(
                f"{config.underlying}{expiry:%y%m%d}{strike}{kind.value}",
                config.underlying,
                kind,
                expiry,
                config.lot_size,
                strike=strike,
            )
            registry.add(contract)

        strategy = replace(default_fno_strategy(config.lots), exit=exits_for(contract))
        engine = PaperTradingEngine(
            PAPER_ACCOUNT_ID, venue, oms, RiskGate(), profile, TradingCalendar(), clock, [strategy],
            contracts=registry, margin_model=margin,
        )  # fmt: skip
        symbol = contract.tradingsymbol
        indicators = StreamingIndicators()
        prev_features: dict | None = None
        entries = 0

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
            allow = entries < config.max_trades
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
    )
