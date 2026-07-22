"""XIRR (extended internal rate of return) over irregularly-dated cash flows,
via Newton's method. Pure function, no dependency beyond stdlib - the only
other option (numpy-financial) doesn't even ship an XIRR, only IRR for
evenly-spaced periods, which doesn't fit real buy/sell transaction dates.
"""

from datetime import date

_MAX_ITERATIONS = 100
_TOLERANCE = 1e-6
_DERIVATIVE_STEP = 1e-6


def _xnpv(rate: float, cash_flows: list[tuple[date, float]]) -> float:
    t0 = cash_flows[0][0]
    return sum(amount / (1 + rate) ** ((when - t0).days / 365) for when, amount in cash_flows)


def xirr(cash_flows: list[tuple[date, float]], guess: float = 0.1) -> float | None:
    """Returns the annualized rate as a fraction (0.12 = 12%), or None if the
    cash flows don't admit a solution (fewer than 2 flows, all one sign - no
    money ever came back - or Newton's method fails to converge).
    """
    if len(cash_flows) < 2:
        return None
    if all(amount >= 0 for _, amount in cash_flows) or all(amount <= 0 for _, amount in cash_flows):
        return None

    rate = guess
    for _ in range(_MAX_ITERATIONS):
        try:
            npv = _xnpv(rate, cash_flows)
            npv_shifted = _xnpv(rate + _DERIVATIVE_STEP, cash_flows)
        except (OverflowError, ZeroDivisionError):
            return None

        derivative = (npv_shifted - npv) / _DERIVATIVE_STEP
        if derivative == 0:
            return None

        new_rate = rate - npv / derivative
        if new_rate <= -1:
            return None
        if abs(new_rate - rate) < _TOLERANCE:
            return new_rate
        rate = new_rate

    return None
