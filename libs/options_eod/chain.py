"""One trading day's NIFTY option chain at the close: real prices, implied
volatility and delta, expiry/strike lookups."""

from __future__ import annotations

from datetime import date

from libs.derivatives import bs
from libs.options_eod.bhav import OptRow


class DayChain:
    def __init__(self, d: date, spot: float, vix: float | None, rows: list[OptRow]) -> None:
        self.date = d
        self.spot = spot
        self.vix = vix
        self._rows: dict[tuple[date, float, str], OptRow] = {(r.expiry, r.strike, r.kind): r for r in rows}
        self._strikes: dict[tuple[date, str], list[float]] = {}
        for r in rows:
            self._strikes.setdefault((r.expiry, r.kind), []).append(r.strike)
        for v in self._strikes.values():
            v.sort()
        self._expiries = sorted({r.expiry for r in rows if r.expiry >= d})
        self._iv: dict[tuple[date, float, str], float | None] = {}

    # --- lookups -------------------------------------------------------

    def expiries(self) -> list[date]:
        return self._expiries

    def monthly_expiries(self) -> list[date]:
        """Last listed expiry of each calendar month (works across NSE's
        Thursday->Tuesday expiry change and holiday shifts)."""
        last: dict[tuple[int, int], date] = {}
        for e in self._expiries:
            key = (e.year, e.month)
            if key not in last or e > last[key]:
                last[key] = e
        return sorted(last.values())

    def strikes(self, expiry: date, kind: str) -> list[float]:
        return self._strikes.get((expiry, kind), [])

    def dte(self, expiry: date) -> int:
        return (expiry - self.date).days

    def years(self, expiry: date) -> float:
        return max(self.dte(expiry), 0) / 365.0

    def price(self, expiry: date, strike: float, kind: str) -> float | None:
        """Closing price if it traded, else NSE's settlement price, else None."""
        r = self._rows.get((expiry, strike, kind))
        if r is None:
            return None
        if r.volume > 0 and r.close > 0:
            return r.close
        if r.settle > 0:
            return r.settle
        return r.close if r.close > 0 else None

    def model_price(self, expiry: date, strike: float, kind: str) -> float:
        """Black-Scholes price at India VIX - only for legs with no quote."""
        vol = (self.vix or 15.0) / 100.0
        return bs.price(kind == "CE", self.spot, strike, self.years(expiry), vol)

    def intrinsic(self, strike: float, kind: str) -> float:
        return max(0.0, self.spot - strike) if kind == "CE" else max(0.0, strike - self.spot)

    # --- greeks ----------------------------------------------------------

    def iv(self, expiry: date, strike: float, kind: str) -> float | None:
        key = (expiry, strike, kind)
        if key not in self._iv:
            p = self.price(expiry, strike, kind)
            self._iv[key] = (
                bs.implied_vol(kind == "CE", p, self.spot, strike, self.years(expiry)) if p is not None else None
            )
        return self._iv[key]

    def delta(self, expiry: date, strike: float, kind: str) -> float | None:
        vol = self.iv(expiry, strike, kind)
        if vol is None:
            return None
        return bs.delta(kind == "CE", self.spot, strike, self.years(expiry), vol)

    def strike_by_delta(self, expiry: date, kind: str, target: float) -> float | None:
        """OTM strike whose |delta| is closest to `target` (e.g. 0.15)."""
        best: tuple[float, float] | None = None
        for k in self.strikes(expiry, kind):
            if (kind == "CE" and k < self.spot) or (kind == "PE" and k > self.spot):
                continue
            d = self.delta(expiry, k, kind)
            if d is None:
                continue
            gap = abs(abs(d) - target)
            if best is None or gap < best[0]:
                best = (gap, k)
        return best[1] if best else None

    def strike_near(self, expiry: date, kind: str, level: float, side: str = "nearest") -> float | None:
        """Listed strike nearest `level`; side='above' -> >= level, 'below' -> <= level."""
        ks = [k for k in self.strikes(expiry, kind) if self.price(expiry, k, kind) is not None]
        if side == "above":
            ks = [k for k in ks if k >= level]
            return min(ks) if ks else None
        if side == "below":
            ks = [k for k in ks if k <= level]
            return max(ks) if ks else None
        return min(ks, key=lambda k: (abs(k - level), k)) if ks else None
