from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PivotLevels:
    pivot: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float


def pivot_points(high: float, low: float, close: float) -> PivotLevels:
    """Classic/standard floor-trader pivot points, computed from a single
    completed bar (typically the prior day's H/L/C for the next session's levels).
    """
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return PivotLevels(pivot=pivot, r1=r1, r2=r2, r3=r3, s1=s1, s2=s2, s3=s3)
