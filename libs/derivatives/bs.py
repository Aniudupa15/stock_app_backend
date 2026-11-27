"""Black-Scholes price, delta and implied volatility for European index options.

Used by the end-of-day options backtester to turn REAL option closing prices
into implied volatility and delta (for "sell the 0.15-delta strike" rules),
and as a last-resort price when a contract has no quote that day.
"""

from __future__ import annotations

import math

RISK_FREE = 0.065


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1d2(spot: float, strike: float, years: float, vol: float, r: float) -> tuple[float, float]:
    sq = vol * math.sqrt(years)
    d1 = (math.log(spot / strike) + (r + 0.5 * vol * vol) * years) / sq
    return d1, d1 - sq


def price(is_call: bool, spot: float, strike: float, years: float, vol: float, r: float = RISK_FREE) -> float:
    if years <= 0 or vol <= 0:
        return max(0.0, spot - strike) if is_call else max(0.0, strike - spot)
    d1, d2 = _d1d2(spot, strike, years, vol, r)
    disc = math.exp(-r * years)
    if is_call:
        return spot * _ncdf(d1) - strike * disc * _ncdf(d2)
    return strike * disc * _ncdf(-d2) - spot * _ncdf(-d1)


def delta(is_call: bool, spot: float, strike: float, years: float, vol: float, r: float = RISK_FREE) -> float:
    """Call delta in [0, 1], put delta in [-1, 0]."""
    if years <= 0 or vol <= 0:
        if is_call:
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0
    d1, _ = _d1d2(spot, strike, years, vol, r)
    return _ncdf(d1) if is_call else _ncdf(d1) - 1.0


def implied_vol(
    is_call: bool, premium: float, spot: float, strike: float, years: float, r: float = RISK_FREE
) -> float | None:
    """Volatility that reproduces `premium` (bisection). None when it can't be
    solved: expired, or a premium at/below intrinsic (stale deep-ITM quotes)."""
    if years <= 0 or premium <= 0:
        return None
    lo, hi = 1e-3, 5.0
    if premium <= price(is_call, spot, strike, years, lo, r) + 1e-9:
        return None
    if premium >= price(is_call, spot, strike, years, hi, r):
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if price(is_call, spot, strike, years, mid, r) > premium:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
