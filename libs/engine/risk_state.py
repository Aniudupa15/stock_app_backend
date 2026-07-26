"""Builds a RiskState snapshot from live venue positions + the OMS trade log,
so the pure risk gate has the numbers it needs without doing any I/O itself.
"""

from __future__ import annotations

from decimal import Decimal

from libs.oms.oms import OrderManagementSystem
from libs.risk.gate import RiskState
from libs.trading_domain.ports import ExecutionVenuePort


class RiskStateBuilder:
    def __init__(self, venue: ExecutionVenuePort, oms: OrderManagementSystem, clock):
        self._venue = venue
        self._oms = oms
        self._clock = clock

    async def build(self, symbol: str) -> RiskState:
        positions = await self._venue.positions()
        cash = self._venue.cash()  # paper venue exposes cash()/equity()
        equity = self._venue.equity()
        today = self._clock().date()

        open_count = sum(1 for p in positions if p.net_qty != 0)
        exposure = sum((abs(p.net_qty) * p.avg_price for p in positions), Decimal("0"))
        net_qty = next((p.net_qty for p in positions if p.symbol == symbol), 0)

        realized_today = Decimal("0")
        for trade in self._oms.trades:
            if trade.exit_ts is not None and trade.exit_ts.date() == today:
                realized_today += trade.pnl_net

        consecutive_losses = 0
        last_loss_at = None
        for trade in reversed(self._oms.trades):
            if trade.pnl_net < 0:
                consecutive_losses += 1
                if last_loss_at is None:
                    last_loss_at = trade.exit_ts
            else:
                break

        return RiskState(
            equity=equity,
            available_cash=cash,
            realized_pnl_today=realized_today,
            realized_pnl_week=realized_today,  # v1: session-scoped; date windowing later
            realized_pnl_month=realized_today,
            open_positions_count=open_count,
            current_exposure=exposure,
            current_net_qty=net_qty,
            consecutive_losses=consecutive_losses,
            last_loss_at=last_loss_at,
        )
