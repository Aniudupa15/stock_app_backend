"""Auto-pilot orchestrator: trades a basket of symbols autonomously.

Wraps the PaperTradingEngine and drives it across many symbols each cycle. The
engine already enforces per-account risk (max open positions, per-trade sizing,
kill switch, square-off window), so feeding it the day's candidates + whatever
is currently held yields autonomous multi-stock trading with no per-symbol
configuration.

A "cycle" is one pass over the market feed (called on each tick/bar by the live
loop during market hours, or by the backtester/simulator). This class is pure
orchestration over injected components, so it's testable without a live market.
"""

from __future__ import annotations

from libs.engine.paper_engine import PaperTradingEngine
from libs.trading_domain.entities import MarketQuote, Signal


class AutoPilot:
    def __init__(self, engine: PaperTradingEngine) -> None:
        self._engine = engine

    async def run_cycle(
        self,
        feed: dict[str, tuple[MarketQuote, dict | None, dict | None]],
    ) -> list[Signal]:
        """Advance every symbol in the feed by one tick.

        `feed`: symbol -> (quote, features, prev_features). Symbols currently
        held should always be included so their exits (SL/target/trailing/
        square-off) are managed; candidate symbols are included so new entries
        can fire. Returns the signals that opened a new position this cycle.
        """
        fired: list[Signal] = []
        for symbol, (quote, features, prev_features) in feed.items():
            signal = await self._engine.process_tick(symbol, quote, features, prev_features)
            if signal is not None:
                fired.append(signal)
        return fired
