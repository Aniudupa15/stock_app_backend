"""Map a persisted StrategyModel to the libs `Strategy` the engine runs."""

from __future__ import annotations

from decimal import Decimal

from libs.strategy.engine import ExitRule, Strategy
from libs.trading_domain.enums import Product, Side
from services.trading_service.persistence.models import StrategyModel


def _dec(value) -> Decimal | None:
    return Decimal(str(value)) if value is not None else None


def strategy_from_model(model: StrategyModel) -> Strategy:
    exit_rule = model.exit_rule or {}
    return Strategy(
        name=model.name,
        rule=model.rule_tree,
        side=Side(model.side),
        product=Product(model.product),
        exit=ExitRule(
            stop_loss_pct=_dec(exit_rule.get("stop_loss_pct")),
            target_pct=_dec(exit_rule.get("target_pct")),
        ),
        quantity=model.quantity,
        strategy_id=model.id,
    )
