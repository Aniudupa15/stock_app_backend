"""Unit tests for the white-box strategy DSL + evaluator. Pure."""

from decimal import Decimal

from libs.strategy.dsl import evaluate_rule, explain_rule
from libs.strategy.engine import ExitRule, Strategy, StrategyEvaluator
from libs.trading_domain.enums import Product, Side


def test_gt_leaf():
    rule = {"op": "GT", "left": {"feature": "close"}, "right": {"feature": "EMA_20"}}
    assert evaluate_rule(rule, {"close": 1010, "EMA_20": 1000}) is True
    assert evaluate_rule(rule, {"close": 990, "EMA_20": 1000}) is False


def test_const_operand():
    rule = {"op": "GTE", "left": {"feature": "RSI_14"}, "right": {"const": 55}}
    assert evaluate_rule(rule, {"RSI_14": 55}) is True
    assert evaluate_rule(rule, {"RSI_14": 54.9}) is False


def test_and_or_not():
    a = {"op": "GT", "left": {"feature": "x"}, "right": {"const": 1}}
    b = {"op": "GT", "left": {"feature": "y"}, "right": {"const": 1}}
    assert evaluate_rule({"op": "AND", "nodes": [a, b]}, {"x": 2, "y": 2}) is True
    assert evaluate_rule({"op": "AND", "nodes": [a, b]}, {"x": 2, "y": 0}) is False
    assert evaluate_rule({"op": "OR", "nodes": [a, b]}, {"x": 2, "y": 0}) is True
    assert evaluate_rule({"op": "NOT", "node": a}, {"x": 0}) is True


def test_cross_above_needs_prev_and_transition():
    rule = {"op": "CROSS_ABOVE", "left": {"feature": "EMA_20"}, "right": {"feature": "EMA_50"}}
    prev = {"EMA_20": 99, "EMA_50": 100}  # below
    now = {"EMA_20": 101, "EMA_50": 100}  # above -> crossed
    assert evaluate_rule(rule, now, prev) is True
    # already above in both -> not a fresh cross
    assert evaluate_rule(rule, now, {"EMA_20": 101, "EMA_50": 100}) is False
    # no history -> cannot cross
    assert evaluate_rule(rule, now, None) is False


def test_missing_feature_is_false_not_error():
    rule = {"op": "GT", "left": {"feature": "missing"}, "right": {"const": 1}}
    assert evaluate_rule(rule, {"other": 5}) is False


def test_explain_rule_substitutes_values():
    rule = {
        "op": "AND",
        "nodes": [
            {"op": "GT", "left": {"feature": "close"}, "right": {"feature": "EMA_20"}},
            {"op": "GTE", "left": {"feature": "RSI_14"}, "right": {"const": 55}},
        ],
    }
    text = explain_rule(rule, {"close": 1010, "EMA_20": 1000, "RSI_14": 58})
    assert "close(1010)" in text and "EMA_20(1000)" in text and "RSI_14(58)" in text
    assert "AND" in text


def test_evaluator_emits_signal_with_levels_and_reasoning():
    rule = {"op": "GT", "left": {"feature": "close"}, "right": {"feature": "EMA_20"}}
    strategy = Strategy(
        name="EMA breakout",
        rule=rule,
        side=Side.BUY,
        product=Product.MIS,
        exit=ExitRule(stop_loss_pct=Decimal("2"), target_pct=Decimal("3")),
        quantity=10,
    )
    signal = StrategyEvaluator().evaluate(strategy, "INFY", {"close": 1010, "EMA_20": 1000}, None, Decimal("1000"))
    assert signal is not None
    assert signal.side is Side.BUY
    assert signal.entry == Decimal("1000")
    assert signal.stop_loss == Decimal("980.00")  # 1000 - 2%
    assert signal.targets == [Decimal("1030.00")]  # 1000 + 3%
    assert "EMA breakout" in signal.reasoning


def test_evaluator_returns_none_when_rule_false():
    rule = {"op": "GT", "left": {"feature": "close"}, "right": {"feature": "EMA_20"}}
    strategy = Strategy(name="s", rule=rule)
    assert StrategyEvaluator().evaluate(strategy, "INFY", {"close": 990, "EMA_20": 1000}, None, Decimal("990")) is None
