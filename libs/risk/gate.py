"""Pre-trade risk gate (Phase 3 §7).

A pure decision function: given an order intent plus a snapshot of account
state and the user's risk profile, return ALLOW / RESIZE(qty) / REJECT(reason).
Stateless - the caller computes `RiskState` from positions/trades and passes it
in, so the gate is trivially testable and has no I/O.

Two hard rules encoded here:
  * **Exits are never blocked.** An intent that reduces an existing position
    (an SL hit, a square-off, a manual close) is always allowed at full size,
    regardless of kill switch or loss limits - you must never trap a position.
  * **Capital preservation over frequency.** Every quantity cap resizes DOWN;
    the gate never increases size, and rejects rather than exceed a limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum

from libs.trading_calendar.calendar import TradingCalendar
from libs.trading_domain.entities import OrderIntent
from libs.trading_domain.enums import Product, Side


@dataclass(frozen=True, slots=True)
class RiskProfile:
    max_daily_loss: Decimal | None = None
    max_weekly_loss: Decimal | None = None
    max_monthly_loss: Decimal | None = None
    per_trade_risk_pct: Decimal | None = None  # e.g. 1.0 = 1% of equity per trade
    max_open_positions: int | None = None
    max_exposure: Decimal | None = None  # total ₹ notional across positions
    cooldown_losses: int | None = None  # consecutive losses that trip a cooldown
    cooldown_minutes: int = 0
    square_off_buffer_minutes: int = 15
    kill_switch: bool = False


@dataclass(frozen=True, slots=True)
class RiskState:
    equity: Decimal
    available_cash: Decimal
    realized_pnl_today: Decimal = Decimal("0")
    realized_pnl_week: Decimal = Decimal("0")
    realized_pnl_month: Decimal = Decimal("0")
    open_positions_count: int = 0
    current_exposure: Decimal = Decimal("0")
    current_net_qty: int = 0  # signed net qty already held in the intent's symbol
    consecutive_losses: int = 0
    last_loss_at: datetime | None = None


class RiskDecision(str, Enum):
    ALLOW = "ALLOW"
    RESIZE = "RESIZE"
    REJECT = "REJECT"


@dataclass(frozen=True, slots=True)
class RiskVerdict:
    decision: RiskDecision
    quantity: int
    reason: str | None = None

    @property
    def allowed(self) -> bool:
        return self.decision is not RiskDecision.REJECT


class RiskGate:
    def evaluate(
        self,
        intent: OrderIntent,
        *,
        state: RiskState,
        profile: RiskProfile,
        now: datetime,
        calendar: TradingCalendar,
        entry_price: Decimal,
        stop_loss: Decimal | None = None,
    ) -> RiskVerdict:
        # --- Exits are always allowed (never trap a position) ---
        if self._is_reducing(intent, state.current_net_qty):
            return RiskVerdict(RiskDecision.ALLOW, intent.quantity)

        # --- Hard rejects (entries only) ---
        if profile.kill_switch:
            return self._reject("kill switch engaged")
        if not calendar.is_open(now):
            return self._reject("market closed")
        if intent.product is Product.MIS and calendar.in_square_off_window(now, profile.square_off_buffer_minutes):
            return self._reject("within square-off window")
        if self._in_cooldown(state, profile, now):
            return self._reject("cooldown after consecutive losses")
        loss_reason = self._loss_limit_breached(state, profile)
        if loss_reason:
            return self._reject(loss_reason)
        if (
            profile.max_open_positions is not None
            and state.current_net_qty == 0
            and state.open_positions_count >= profile.max_open_positions
        ):
            return self._reject("max open positions reached")

        # --- Quantity caps: take the most binding, resize down ---
        caps: list[tuple[int, str]] = []
        if profile.per_trade_risk_pct is not None and stop_loss is not None:
            sl_distance = abs(entry_price - stop_loss)
            if sl_distance > 0:
                risk_amount = state.equity * (profile.per_trade_risk_pct / Decimal("100"))
                caps.append((int(risk_amount / sl_distance), "per-trade risk"))
        if profile.max_exposure is not None and entry_price > 0:
            remaining = profile.max_exposure - state.current_exposure
            caps.append((max(0, int(remaining / entry_price)), "max exposure"))
        if entry_price > 0:
            caps.append((int(state.available_cash / entry_price), "insufficient margin"))

        final_qty = intent.quantity
        binding_reason = None
        for cap_qty, reason in caps:
            if cap_qty < final_qty:
                final_qty = cap_qty
                binding_reason = reason

        if final_qty <= 0:
            return self._reject(binding_reason or "risk limits allow zero quantity")
        if final_qty < intent.quantity:
            return RiskVerdict(RiskDecision.RESIZE, final_qty, binding_reason)
        return RiskVerdict(RiskDecision.ALLOW, intent.quantity)

    # --- helpers ---

    @staticmethod
    def _reject(reason: str) -> RiskVerdict:
        return RiskVerdict(RiskDecision.REJECT, 0, reason)

    @staticmethod
    def _is_reducing(intent: OrderIntent, net_qty: int) -> bool:
        if net_qty > 0 and intent.side is Side.SELL:
            return True
        if net_qty < 0 and intent.side is Side.BUY:
            return True
        return False

    @staticmethod
    def _in_cooldown(state: RiskState, profile: RiskProfile, now: datetime) -> bool:
        if profile.cooldown_losses is None or state.last_loss_at is None:
            return False
        if state.consecutive_losses < profile.cooldown_losses:
            return False
        return now < state.last_loss_at + timedelta(minutes=profile.cooldown_minutes)

    @staticmethod
    def _loss_limit_breached(state: RiskState, profile: RiskProfile) -> str | None:
        if profile.max_daily_loss is not None and state.realized_pnl_today <= -profile.max_daily_loss:
            return "daily loss limit reached"
        if profile.max_weekly_loss is not None and state.realized_pnl_week <= -profile.max_weekly_loss:
            return "weekly loss limit reached"
        if profile.max_monthly_loss is not None and state.realized_pnl_month <= -profile.max_monthly_loss:
            return "monthly loss limit reached"
        return None
