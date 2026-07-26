"""Unit tests for the pre-trade risk gate. Pure - no DB, no network."""

from datetime import datetime
from decimal import Decimal
from uuid import uuid4

from libs.risk.gate import RiskDecision, RiskGate, RiskProfile, RiskState
from libs.trading_calendar.calendar import TradingCalendar
from libs.trading_domain.entities import OrderIntent
from libs.trading_domain.enums import OrderType, Product, Side

MON_OPEN = datetime(2026, 1, 5, 10, 0)
CAL = TradingCalendar()
GATE = RiskGate()
ACCOUNT = uuid4()


def _intent(side=Side.BUY, qty=100, product=Product.MIS):
    return OrderIntent(uuid4(), ACCOUNT, "INFY", side, OrderType.MARKET, product, qty)


def _state(**kw):
    base = dict(equity=Decimal("1000000"), available_cash=Decimal("1000000"))
    base.update(kw)
    return RiskState(**base)


def test_allows_clean_entry():
    v = GATE.evaluate(_intent(), state=_state(), profile=RiskProfile(), now=MON_OPEN, calendar=CAL, entry_price=Decimal("1000"))
    assert v.decision is RiskDecision.ALLOW
    assert v.quantity == 100


def test_kill_switch_rejects_entry():
    v = GATE.evaluate(
        _intent(), state=_state(), profile=RiskProfile(kill_switch=True), now=MON_OPEN, calendar=CAL, entry_price=Decimal("1000")
    )
    assert v.decision is RiskDecision.REJECT
    assert v.reason == "kill switch engaged"


def test_exit_allowed_even_with_kill_switch():
    # Holding 100 long; a SELL reduces -> must be allowed despite kill switch.
    state = _state(current_net_qty=100)
    v = GATE.evaluate(
        _intent(side=Side.SELL),
        state=state,
        profile=RiskProfile(kill_switch=True),
        now=MON_OPEN,
        calendar=CAL,
        entry_price=Decimal("1000"),
    )
    assert v.decision is RiskDecision.ALLOW


def test_market_closed_rejects():
    closed = datetime(2026, 1, 5, 8, 0)
    v = GATE.evaluate(_intent(), state=_state(), profile=RiskProfile(), now=closed, calendar=CAL, entry_price=Decimal("1000"))
    assert v.decision is RiskDecision.REJECT
    assert v.reason == "market closed"


def test_mis_rejected_in_square_off_window():
    sq = datetime(2026, 1, 5, 15, 20)  # inside default 15:15-15:30 window
    v = GATE.evaluate(_intent(product=Product.MIS), state=_state(), profile=RiskProfile(), now=sq, calendar=CAL, entry_price=Decimal("1000"))
    assert v.decision is RiskDecision.REJECT
    assert v.reason == "within square-off window"


def test_daily_loss_limit_rejects():
    state = _state(realized_pnl_today=Decimal("-5000"))
    v = GATE.evaluate(
        _intent(), state=state, profile=RiskProfile(max_daily_loss=Decimal("5000")), now=MON_OPEN, calendar=CAL, entry_price=Decimal("1000")
    )
    assert v.decision is RiskDecision.REJECT
    assert v.reason == "daily loss limit reached"


def test_cooldown_after_consecutive_losses():
    state = _state(consecutive_losses=3, last_loss_at=datetime(2026, 1, 5, 9, 55))
    profile = RiskProfile(cooldown_losses=3, cooldown_minutes=30)
    v = GATE.evaluate(_intent(), state=state, profile=profile, now=MON_OPEN, calendar=CAL, entry_price=Decimal("1000"))
    assert v.decision is RiskDecision.REJECT
    assert v.reason == "cooldown after consecutive losses"


def test_max_open_positions_blocks_new_symbol():
    state = _state(open_positions_count=5, current_net_qty=0)
    v = GATE.evaluate(_intent(), state=state, profile=RiskProfile(max_open_positions=5), now=MON_OPEN, calendar=CAL, entry_price=Decimal("1000"))
    assert v.decision is RiskDecision.REJECT
    assert v.reason == "max open positions reached"


def test_position_sizing_resizes_down_by_per_trade_risk():
    # equity 1,000,000; risk 1% = 10,000; SL distance = 1000-980 = 20 -> max 500 qty
    state = _state(equity=Decimal("1000000"))
    profile = RiskProfile(per_trade_risk_pct=Decimal("1"))
    v = GATE.evaluate(
        _intent(qty=1000), state=state, profile=profile, now=MON_OPEN, calendar=CAL, entry_price=Decimal("1000"), stop_loss=Decimal("980")
    )
    assert v.decision is RiskDecision.RESIZE
    assert v.quantity == 500
    assert v.reason == "per-trade risk"


def test_insufficient_margin_resizes_down():
    # cash only 50,000 at 1000/share -> max 50 shares
    state = _state(available_cash=Decimal("50000"))
    v = GATE.evaluate(_intent(qty=100), state=state, profile=RiskProfile(), now=MON_OPEN, calendar=CAL, entry_price=Decimal("1000"))
    assert v.decision is RiskDecision.RESIZE
    assert v.quantity == 50
    assert v.reason == "insufficient margin"


def test_zero_affordable_rejects():
    state = _state(available_cash=Decimal("500"))
    v = GATE.evaluate(_intent(qty=100), state=state, profile=RiskProfile(), now=MON_OPEN, calendar=CAL, entry_price=Decimal("1000"))
    assert v.decision is RiskDecision.REJECT
