"""White-box strategy rule DSL (Phase 3 §9).

A rule is a JSON-serialisable boolean tree of AND/OR/NOT over comparison
leaves (GT/LT/GTE/LTE/EQ/CROSS_ABOVE/CROSS_BELOW) whose operands are either a
named feature (`{"feature": "EMA_20"}`) or a constant (`{"const": 55}`).
Evaluation is against an injected feature dict (indicator-name -> value), so
this stays pure and independent of how indicators are computed - the same
tree drives backtest, paper, and live (SEBI white-box: fully disclosed,
deterministic, and explainable).

A missing feature makes its leaf False (rather than crashing) - early in a
session an indicator may not have enough history yet, and "not enough data to
say yes" is the correct, conservative reading.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

_BOOL_OPS = {"AND", "OR", "NOT"}
_CMP_SYMBOLS = {
    "GT": ">",
    "LT": "<",
    "GTE": ">=",
    "LTE": "<=",
    "EQ": "==",
    "CROSS_ABOVE": "crosses above",
    "CROSS_BELOW": "crosses below",
}


class RuleError(ValueError):
    """Malformed rule tree (structural, not a missing runtime value)."""


def _to_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _resolve(ref: dict, features: dict) -> Decimal | None:
    if "const" in ref:
        return _to_decimal(ref["const"])
    if "feature" in ref:
        return _to_decimal(features.get(ref["feature"]))
    raise RuleError(f"operand must have 'feature' or 'const': {ref!r}")


def evaluate_rule(node: dict, features: dict, prev_features: dict | None = None) -> bool:
    op = node.get("op")
    if op is None:
        raise RuleError(f"node missing 'op': {node!r}")

    if op == "AND":
        return all(evaluate_rule(n, features, prev_features) for n in node["nodes"])
    if op == "OR":
        return any(evaluate_rule(n, features, prev_features) for n in node["nodes"])
    if op == "NOT":
        return not evaluate_rule(node["node"], features, prev_features)

    if op not in _CMP_SYMBOLS:
        raise RuleError(f"unknown op: {op!r}")

    left = _resolve(node["left"], features)
    right = _resolve(node["right"], features)

    if op in ("CROSS_ABOVE", "CROSS_BELOW"):
        if prev_features is None:
            return False
        prev_left = _resolve(node["left"], prev_features)
        prev_right = _resolve(node["right"], prev_features)
        if None in (left, right, prev_left, prev_right):
            return False
        if op == "CROSS_ABOVE":
            return prev_left <= prev_right and left > right
        return prev_left >= prev_right and left < right

    if left is None or right is None:
        return False
    if op == "GT":
        return left > right
    if op == "LT":
        return left < right
    if op == "GTE":
        return left >= right
    if op == "LTE":
        return left <= right
    return left == right  # EQ


def explain_rule(node: dict, features: dict) -> str:
    """Human-readable rendering with actual feature values substituted -
    the templated reasoning attached to every signal."""
    op = node.get("op")
    if op == "AND":
        return " AND ".join(f"({explain_rule(n, features)})" for n in node["nodes"])
    if op == "OR":
        return " OR ".join(f"({explain_rule(n, features)})" for n in node["nodes"])
    if op == "NOT":
        return f"NOT ({explain_rule(node['node'], features)})"
    return f"{_label(node['left'], features)} {_CMP_SYMBOLS.get(op, op)} {_label(node['right'], features)}"


def _label(ref: dict, features: dict) -> str:
    if "const" in ref:
        return str(ref["const"])
    key = ref["feature"]
    value = features.get(key)
    return f"{key}({value})" if value is not None else f"{key}(n/a)"
