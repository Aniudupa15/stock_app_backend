"""Adapt the data-service's persisted OHLCV bars into the backtester's Bar.

Integration layer: may depend on both `app` (data-service) and `libs`, which
`libs` itself must not. This is where the two halves of the monorepo meet
until the physical restructure lands.
"""

from __future__ import annotations

from app.domain.entities import OhlcvBar
from libs.backtest.bar import Bar


def to_backtest_bar(bar: OhlcvBar) -> Bar:
    return Bar(
        day=bar.trade_date,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        volume=bar.volume,
    )


def to_backtest_bars(bars: list[OhlcvBar]) -> list[Bar]:
    """Map a symbol's daily history (ascending by date) to backtester bars."""
    return [to_backtest_bar(b) for b in bars]
