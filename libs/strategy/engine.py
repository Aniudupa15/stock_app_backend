"""Strategy definition + evaluator: a white-box rule fires -> a Signal with
entry/SL/target and templated reasoning citing the actual values.

Pure: given features and an LTP it produces a Signal (or None). Sizing, risk,
and order placement happen downstream (risk gate + OMS)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from uuid import UUID

from libs.strategy.dsl import evaluate_rule, explain_rule
from libs.trading_domain.entities import Signal, TrailingSpec
from libs.trading_domain.enums import Product, Side

_TICK = Decimal("0.01")


def _round(price: Decimal) -> Decimal:
    return price.quantize(_TICK, rounding=ROUND_HALF_UP)


@dataclass(frozen=True, slots=True)
class ExitRule:
    stop_loss_pct: Decimal | None = None
    target_pct: Decimal | None = None
    trailing: TrailingSpec | None = None


@dataclass(frozen=True, slots=True)
class Strategy:
    name: str
    rule: dict  # white-box entry rule tree
    side: Side = Side.BUY
    product: Product = Product.MIS
    exit: ExitRule = field(default_factory=ExitRule)
    quantity: int = 1  # requested size; the risk gate resizes down
    strategy_id: UUID | None = None
    timeframe: str = "1D"


class StrategyEvaluator:
    def evaluate(
        self,
        strategy: Strategy,
        symbol: str,
        features: dict,
        prev_features: dict | None,
        ltp: Decimal,
        now: datetime | None = None,
    ) -> Signal | None:
        if not evaluate_rule(strategy.rule, features, prev_features):
            return None

        entry = ltp
        stop_loss, targets = self._exit_levels(strategy, entry)
        reasoning = f"{strategy.name}: entry matched [{explain_rule(strategy.rule, features)}] @ LTP={entry}"
        return Signal(
            strategy_id=strategy.strategy_id,
            symbol=symbol,
            side=strategy.side,
            entry=entry,
            stop_loss=stop_loss,
            targets=targets,
            reasoning=reasoning,
            ts=now,
        )

    @staticmethod
    def _exit_levels(strategy: Strategy, entry: Decimal) -> tuple[Decimal | None, list[Decimal]]:
        sl_pct = strategy.exit.stop_loss_pct
        tgt_pct = strategy.exit.target_pct
        stop_loss: Decimal | None = None
        targets: list[Decimal] = []
        # Long: SL below, target above. Short: mirror.
        direction = 1 if strategy.side is Side.BUY else -1
        if sl_pct is not None:
            stop_loss = _round(entry * (Decimal("1") - direction * sl_pct / Decimal("100")))
        if tgt_pct is not None:
            targets = [_round(entry * (Decimal("1") + direction * tgt_pct / Decimal("100")))]
        return stop_loss, targets
