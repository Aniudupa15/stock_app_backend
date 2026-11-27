"""The four end-of-day NIFTY option strategies (rules as specified; every
decision is taken on the day's closing prices).

1. MonthlyIronCondor   - sell ~delta strangle 30-35 DTE in the monthly, wings beyond
2. TrendCreditSpread   - daily EMA trend; sell the put (bull) / call (bear) spread
3. VixSpikeCondor      - after an India VIX spike turns down, sell a wide condor
4. TrendDebitSpread    - fresh daily EMA cross -> buy a call / put spread
"""

from __future__ import annotations

from collections import deque
from datetime import date

from libs.options_eod.chain import DayChain
from libs.options_eod.engine import Ctx, Leg, Strategy


class Ema:
    def __init__(self, period: int) -> None:
        self.period, self.value, self.n = period, None, 0

    def update(self, x: float) -> float:
        k = 2.0 / (self.period + 1)
        self.value = x if self.value is None else self.value + k * (x - self.value)
        self.n += 1
        return self.value

    @property
    def ready(self) -> bool:
        return self.n >= self.period


def _monthly_in_window(day: DayChain, lo: int, hi: int) -> date | None:
    for e in day.monthly_expiries():
        if lo <= day.dte(e) <= hi:
            return e
    return None


def _first_monthly_with(day: DayChain, min_dte: int) -> date | None:
    return next((e for e in day.monthly_expiries() if day.dte(e) >= min_dte), None)


def _condor(day: DayChain, expiry: date, delta: float, wing: float) -> list[Leg] | None:
    sc = day.strike_by_delta(expiry, "CE", delta)
    sp = day.strike_by_delta(expiry, "PE", delta)
    if sc is None or sp is None:
        return None
    lc = day.strike_near(expiry, "CE", sc + wing, "above")
    lp = day.strike_near(expiry, "PE", sp - wing, "below")
    if lc is None or lp is None or lc <= sc or lp >= sp:
        return None
    return [Leg(expiry, sc, "CE", -1), Leg(expiry, lc, "CE", +1), Leg(expiry, sp, "PE", -1), Leg(expiry, lp, "PE", +1)]


def _short_strikes(legs: list[Leg]) -> tuple[float | None, float | None]:
    call = next((leg.strike for leg in legs if leg.qty < 0 and leg.kind == "CE"), None)
    put = next((leg.strike for leg in legs if leg.qty < 0 and leg.kind == "PE"), None)
    return call, put


class MonthlyIronCondor(Strategy):
    label = "iron condor (monthly)"

    def __init__(
        self,
        delta: float = 0.15,
        wing: float = 200,
        take: float | None = 0.5,  # close at this fraction of the credit captured; None = no target
        exit_dte: int | None = 14,  # close at this many days to expiry; None = hold to expiry
        stop_mult: float = 2.0,
        vix_lo: float = 12,
        vix_hi: float = 22,
        entry_dte: tuple[int, int] = (30, 35),
    ) -> None:
        super().__init__(delta=delta, wing=wing, take=take, exit_dte=exit_dte, stop_mult=stop_mult)
        self.delta, self.wing, self.take, self.exit_dte, self.stop_mult = delta, wing, take, exit_dte, stop_mult
        self.vix_lo, self.vix_hi, self.entry_dte = vix_lo, vix_hi, entry_dte
        self.done: set[date] = set()

    def on_day(self, day: DayChain, ctx: Ctx) -> None:
        pos = ctx.position
        if pos is None:
            e = _monthly_in_window(day, *self.entry_dte)
            if e is None or e in self.done or day.vix is None or not (self.vix_lo <= day.vix <= self.vix_hi):
                return
            legs = _condor(day, e, self.delta, self.wing)
            if legs and ctx.open(legs, tag=f"VIX {day.vix:.1f}") and ctx.position.entry_cash > 0:
                self.done.add(e)
            elif ctx.position is not None:  # no credit (shouldn't happen) - don't hold a debit condor
                ctx.close("NO_CREDIT")
            return
        credit, cost = pos.entry_cash, ctx.cost_to_close()
        sc, sp = _short_strikes(pos.legs)
        if cost >= self.stop_mult * credit:
            ctx.close("STOP")
        elif (sc is not None and day.spot > sc) or (sp is not None and day.spot < sp):
            ctx.close("BREACH")
        elif self.take is not None and cost <= (1 - self.take) * credit:
            ctx.close("TARGET")
        elif self.exit_dte is not None and day.dte(pos.expiry) <= self.exit_dte:
            ctx.close("TIME")


class TrendCreditSpread(Strategy):
    label = "trend credit spread"

    def __init__(
        self,
        fast: int = 20,
        slow: int = 50,
        delta: float = 0.25,
        width: float = 100,
        cycle: str = "weekly",  # weekly | monthly
        stop_mult: float = 2.0,
    ) -> None:
        super().__init__(ema=f"{fast}/{slow}", delta=delta, width=width, cycle=cycle, stop_mult=stop_mult)
        self.f, self.s = Ema(fast), Ema(slow)
        self.delta, self.width, self.cycle, self.stop_mult = delta, width, cycle, stop_mult
        self.done: set[object] = set()

    def on_day(self, day: DayChain, ctx: Ctx) -> None:
        fast, slow = self.f.update(day.spot), self.s.update(day.spot)
        pos = ctx.position
        if pos is not None:
            credit, cost = pos.entry_cash, ctx.cost_to_close()
            sc, sp = _short_strikes(pos.legs)
            if cost >= self.stop_mult * credit:
                ctx.close("STOP")
            elif (sc is not None and day.spot > sc) or (sp is not None and day.spot < sp):
                ctx.close("BREACH")
            return
        if not self.s.ready:
            return
        if self.cycle == "weekly":
            week = day.date.isocalendar()[:2]
            if day.date.weekday() < 2 or week in self.done:  # first trading day from Wednesday
                return
            expiry = next((e for e in day.expiries() if day.dte(e) >= 4), None)
            key = week
        else:
            expiry = _monthly_in_window(day, 20, 35)
            key = expiry
            if expiry is None or key in self.done:
                return
        if expiry is None:
            return
        bullish = fast > slow
        kind = "PE" if bullish else "CE"
        short = day.strike_by_delta(expiry, kind, self.delta)
        if short is None:
            return
        long_ = (
            day.strike_near(expiry, kind, short - self.width, "below")
            if bullish
            else day.strike_near(expiry, kind, short + self.width, "above")
        )
        if long_ is None or long_ == short:
            return
        if ctx.open([Leg(expiry, short, kind, -1), Leg(expiry, long_, kind, +1)], tag="bull" if bullish else "bear"):
            self.done.add(key)


class VixSpikeCondor(Strategy):
    label = "VIX spike condor"

    def __init__(
        self,
        pct: float = 0.8,
        wait_turn: bool = True,
        delta: float = 0.10,
        wing: float = 250,
        take: float = 0.5,
        stop_mult: float = 2.0,
        lookback: int = 252,
    ) -> None:
        super().__init__(pct=pct, wait_turn=wait_turn, delta=delta, wing=wing)
        self.pct, self.wait_turn, self.delta, self.wing, self.take, self.stop_mult = (
            pct, wait_turn, delta, wing, take, stop_mult,
        )  # fmt: skip
        self.hist: deque[float] = deque(maxlen=lookback)
        self.spiked = False
        self.enter_next = False
        self.armed = True  # one trade per spike: re-arms once VIX is back below the threshold
        self.prev_vix: float | None = None

    def _quantile(self, q: float) -> float:
        s = sorted(self.hist)
        return s[min(len(s) - 1, int(q * (len(s) - 1)))]

    def on_day(self, day: DayChain, ctx: Ctx) -> None:
        vix = day.vix
        if vix is None:
            return
        full = len(self.hist) == self.hist.maxlen
        threshold = self._quantile(self.pct) if full else None
        median = self._quantile(0.5) if full else None
        pos = ctx.position
        if pos is not None:
            credit, cost = pos.entry_cash, ctx.cost_to_close()
            if cost >= self.stop_mult * credit:
                ctx.close("STOP")
            elif cost <= (1 - self.take) * credit:
                ctx.close("TARGET")
            elif median is not None and vix <= median:
                ctx.close("VIX_NORMAL")
        elif self.enter_next:
            e = _first_monthly_with(day, 15)
            legs = _condor(day, e, self.delta, self.wing) if e else None
            if legs and ctx.open(legs, tag=f"VIX {vix:.1f}"):
                self.enter_next, self.armed = False, False
            elif threshold is not None and vix <= threshold:
                self.enter_next = False  # couldn't build it while the spike lasted: give up
        elif self.spiked and self.prev_vix is not None and vix < self.prev_vix:
            # First lower close after the spike - even if VIX is still elevated: enter tomorrow.
            self.enter_next, self.spiked = True, False
        elif threshold is not None and vix > threshold and self.armed:
            if self.wait_turn:
                self.spiked = True
            else:
                self.enter_next = True
        if threshold is not None and vix <= threshold:
            self.armed = True
        self.hist.append(vix)
        self.prev_vix = vix


class TrendDebitSpread(Strategy):
    label = "trend debit spread"

    def __init__(
        self,
        fast: int = 20,
        slow: int = 50,
        width: float = 200,
        target: float = 0.8,  # close when the spread is worth this fraction of its width
        exit_dte: int = 7,
        roll: bool = True,
    ) -> None:
        super().__init__(ema=f"{fast}/{slow}", width=width, target=target)
        self.f, self.s = Ema(fast), Ema(slow)
        self.width, self.target, self.exit_dte, self.roll = width, target, exit_dte, roll
        self.prev_bull: bool | None = None
        self.pending: bool | None = None  # direction to enter tomorrow
        self.rolled = False

    def _open(self, day: DayChain, ctx: Ctx, bull: bool) -> bool:
        e = _first_monthly_with(day, 25)
        if e is None:
            return False
        kind = "CE" if bull else "PE"
        atm = day.strike_near(e, kind, day.spot)
        if atm is None:
            return False
        far = day.strike_near(e, kind, atm + self.width if bull else atm - self.width, "above" if bull else "below")
        if far is None or far == atm:
            return False
        return ctx.open([Leg(e, atm, kind, +1), Leg(e, far, kind, -1)], tag="bull" if bull else "bear")

    def on_day(self, day: DayChain, ctx: Ctx) -> None:
        fast, slow = self.f.update(day.spot), self.s.update(day.spot)
        if not self.s.ready:
            return
        bull = fast > slow
        cross = self.prev_bull is not None and bull != self.prev_bull
        self.prev_bull = bull
        pos = ctx.position
        if pos is not None:
            is_bull = pos.tag == "bull"
            value = -ctx.cost_to_close()  # what the spread is worth now
            if cross and bull != is_bull:
                ctx.close("OPPOSITE_CROSS")
                self.pending, self.rolled = bull, False
            elif value >= self.target * self.width:
                ctx.close("TARGET")
            elif day.dte(pos.expiry) <= self.exit_dte:
                ctx.close("TIME")
                if self.roll and not self.rolled and bull == is_bull:
                    self.rolled = True
                    self.pending = bull
            return
        if self.pending is not None:
            if self._open(day, ctx, self.pending):
                self.pending = None
            return
        if cross:
            self.pending, self.rolled = bull, False  # enter tomorrow
