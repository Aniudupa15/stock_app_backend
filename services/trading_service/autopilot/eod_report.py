"""End-of-day trading report - the summary the user gets after each session."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from libs.trading_domain.entities import Trade


@dataclass(frozen=True, slots=True)
class EodReport:
    day: date
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    gross_pnl: Decimal
    charges_total: Decimal
    net_pnl: Decimal
    best_symbol: str | None
    best_pnl: Decimal
    worst_symbol: str | None
    worst_pnl: Decimal

    def as_dict(self) -> dict:
        return {
            "day": self.day.isoformat(),
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "gross_pnl": str(self.gross_pnl),
            "charges_total": str(self.charges_total),
            "net_pnl": str(self.net_pnl),
            "best": {"symbol": self.best_symbol, "pnl": str(self.best_pnl)},
            "worst": {"symbol": self.worst_symbol, "pnl": str(self.worst_pnl)},
        }


def build_eod_report(trades: list[Trade], day: date) -> EodReport:
    """Aggregate the trades closed on `day` into a report."""
    todays = [t for t in trades if t.exit_ts is not None and t.exit_ts.date() == day]
    n = len(todays)
    wins = [t for t in todays if t.pnl_net > 0]
    losses = [t for t in todays if t.pnl_net < 0]
    net = sum((t.pnl_net for t in todays), Decimal("0"))
    gross = sum((t.pnl_gross for t in todays), Decimal("0"))
    charges = sum((t.charges_total for t in todays), Decimal("0"))

    best = max(todays, key=lambda t: t.pnl_net, default=None)
    worst = min(todays, key=lambda t: t.pnl_net, default=None)

    return EodReport(
        day=day,
        total_trades=n,
        wins=len(wins),
        losses=len(losses),
        win_rate=len(wins) / n if n else 0.0,
        gross_pnl=gross,
        charges_total=charges,
        net_pnl=net,
        best_symbol=best.symbol if best else None,
        best_pnl=best.pnl_net if best else Decimal("0"),
        worst_symbol=worst.symbol if worst else None,
        worst_pnl=worst.pnl_net if worst else Decimal("0"),
    )
