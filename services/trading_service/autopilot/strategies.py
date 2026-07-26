"""Built-in auto-pilot strategy presets (white-box, no user configuration).

The point of auto-pilot: the user flips it ON and the app trades - no strategy
authoring required. These presets are the default brains. They're the same
white-box rule DSL as user strategies, so they're fully inspectable and
SEBI-aligned (deterministic, disclosed).
"""

from __future__ import annotations

from decimal import Decimal

from libs.strategy.engine import ExitRule, Strategy
from libs.trading_domain.entities import TrailingSpec
from libs.trading_domain.enums import Product, Side

# Momentum-with-trend, not overbought:
#   close > EMA20  AND  EMA20 > EMA50  AND  55 < RSI14 < 72
MOMENTUM_RULE = {
    "op": "AND",
    "nodes": [
        {"op": "GT", "left": {"feature": "close"}, "right": {"feature": "EMA_20"}},
        {"op": "GT", "left": {"feature": "EMA_20"}, "right": {"feature": "EMA_50"}},
        {"op": "GT", "left": {"feature": "RSI_14"}, "right": {"const": 55}},
        {"op": "LT", "left": {"feature": "RSI_14"}, "right": {"const": 72}},
    ],
}


def default_momentum_strategy(quantity: int = 10, product: Product = Product.MIS) -> Strategy:
    """Default intraday auto-pilot strategy: momentum entries, 1.5% stop, 3%
    target, 1% trailing stop. MIS so positions square off same day."""
    return Strategy(
        name="Auto-Pilot Momentum",
        rule=MOMENTUM_RULE,
        side=Side.BUY,
        product=product,
        exit=ExitRule(
            stop_loss_pct=Decimal("1.5"),
            target_pct=Decimal("3"),
            trailing=TrailingSpec(by="PCT", value=Decimal("1"), step=Decimal("0.2")),
        ),
        quantity=quantity,
    )
