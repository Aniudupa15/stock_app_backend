"""Strategy-variant rules shared by the F&O backtest and the live paper
session, so a variant means exactly the same thing in both.

Presets are the variants that held up in the full-year NIFTY put backtest
(profitable in both halves of Nov 2025 - Nov 2026, with modelled premiums).
Paper-trade them on real premiums before trusting them.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

from libs.strategy.engine import ExitRule

OPENING_RANGE_MINUTES = 15

PRESETS: dict[str, dict] = {
    # Buy the ATM put (nearest expiry) only while NIFTY is below today's open;
    # 1 entry a day, 30% stop + 30% trailing stop, no target, out by 15:15.
    "pe-trend": {
        "instrument": "PE",
        "max_trades": 1,
        "trend_filter": "open",
        "no_target": True,
        "sl_pct": Decimal("30"),
        "trail_pct": Decimal("30"),
    },
    # Same exits, next-week expiry, entry only after NIFTY breaks below the
    # first 15 minutes' low.
    "pe-orb-next": {
        "instrument": "PE",
        "max_trades": 1,
        "trend_filter": "orb",
        "expiry": "next",
        "no_target": True,
        "sl_pct": Decimal("30"),
        "trail_pct": Decimal("30"),
    },
}


def trend_ok(
    mode: str, bullish: bool, price: Decimal, day_open: Decimal, or_high: Decimal, or_low: Decimal, or_done: bool
) -> bool:
    """Entry filter on the UNDERLYING (index) price:
    open - only trade with the day: index above its open (FUT/CE) or below it (PE)
    orb  - opening-range breakout: after the first 15 minutes, index beyond
           that range's high (FUT/CE) or low (PE)."""
    if mode == "none":
        return True
    if mode == "open":
        return price > day_open if bullish else price < day_open
    if mode == "orb":
        return or_done and (price > or_high if bullish else price < or_low)
    raise ValueError(f"unknown trend filter {mode!r}")


def override_exits(
    base: ExitRule,
    *,
    sl_pct: Decimal | None = None,
    target_pct: Decimal | None = None,
    no_target: bool = False,
    trail_pct: Decimal | None = None,
) -> ExitRule:
    trailing = base.trailing
    if trail_pct is not None and trailing is not None:
        trailing = replace(trailing, value=trail_pct)
    return replace(
        base,
        stop_loss_pct=sl_pct if sl_pct is not None else base.stop_loss_pct,
        target_pct=None if no_target else (target_pct or base.target_pct),
        trailing=trailing,
    )


def parse_with_preset(parser, argv=None):
    """argparse with `--preset NAME`: the preset supplies defaults, and any
    flag given explicitly on the command line still wins."""
    pre, _ = parser.parse_known_args(argv)
    if getattr(pre, "preset", None):
        parser.set_defaults(**PRESETS[pre.preset])
    return parser.parse_args(argv)
