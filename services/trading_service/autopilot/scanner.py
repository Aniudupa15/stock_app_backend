"""Market scanner - "the app finds the stocks."

Reuses the data-service's daily-materialised intraday signal snapshots
(computed for all ~2,400 stocks by the scheduler), so scanning is a fast
indexed query, not a live recompute. Returns ranked BUY candidates for the
auto-pilot to consider.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.intraday_signal_snapshot_repository import SqlAlchemyIntradaySignalSnapshotRepository


@dataclass(frozen=True, slots=True)
class Candidate:
    symbol: str
    name: str
    signal: str
    confidence: Decimal
    entry: Decimal | None
    target: Decimal | None
    stop_loss: Decimal | None
    reasoning: list[str]

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "signal": self.signal,
            "confidence": float(self.confidence),
            "entry": None if self.entry is None else float(self.entry),
            "target": None if self.target is None else float(self.target),
            "stop_loss": None if self.stop_loss is None else float(self.stop_loss),
            "reasoning": self.reasoning,
        }


async def scan_candidates(
    session: AsyncSession,
    *,
    min_confidence: Decimal = Decimal("60"),
    limit: int = 20,
    buy_only: bool = True,
) -> list[Candidate]:
    """Top-ranked candidates from the materialised intraday signals."""
    repo = SqlAlchemyIntradaySignalSnapshotRepository(session)
    snapshots = await repo.list_top(min_confidence, limit if not buy_only else limit * 2)
    candidates = [
        Candidate(
            symbol=s.symbol,
            name=s.name,
            signal=s.signal,
            confidence=s.confidence,
            entry=s.entry_price,
            target=s.target_price,
            stop_loss=s.stop_loss,
            reasoning=s.reasoning,
        )
        for s in snapshots
        if not buy_only or s.signal == "BUY"
    ]
    return candidates[:limit]
