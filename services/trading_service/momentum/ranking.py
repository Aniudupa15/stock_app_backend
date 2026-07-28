"""Cross-sectional momentum ranking - the validated edge (see research).

Ranks the liquid NSE universe by trailing return and returns the top N. Runs on
the daily `historical_prices` already in the DB (one bulk query + numpy), so no
real-time feed is needed. Default params (30-day lookback, top 30, ~300-name
liquid universe) are the config that validated across 6.5 years.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True, slots=True)
class MomentumPick:
    symbol: str
    name: str
    trailing_return_pct: float
    last_close: float
    avg_turnover: float

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "trailing_return_pct": round(self.trailing_return_pct, 2),
            "last_close": round(self.last_close, 2),
        }


async def compute_ranking(
    session: AsyncSession,
    *,
    lookback: int = 30,
    top: int = 30,
    universe_size: int = 300,
) -> list[MomentumPick]:
    """Top `top` momentum stocks: liquid universe (by turnover) ranked by
    `lookback`-day trailing return."""
    latest = (await session.execute(text("select max(trade_date) from historical_prices"))).scalar()
    if latest is None:
        return []
    # ~100 calendar days covers the ~65 trading days we need for turnover + lookback.
    cutoff = latest - timedelta(days=max(100, lookback * 3))
    rows = (
        await session.execute(
            text(
                "select s.symbol, s.name, hp.close, hp.volume "
                "from historical_prices hp join stocks s on s.id = hp.stock_id "
                "where hp.trade_date >= :cutoff and s.is_active = true "
                "order by s.symbol, hp.trade_date"
            ),
            {"cutoff": cutoff},
        )
    ).all()

    series: dict[str, dict] = {}
    for symbol, name, close, volume in rows:
        s = series.setdefault(symbol, {"name": name, "closes": [], "turnover": []})
        s["closes"].append(float(close))
        s["turnover"].append(float(close) * float(volume))

    ranked: list[MomentumPick] = []
    for symbol, s in series.items():
        closes = s["closes"]
        if len(closes) <= lookback or closes[-1 - lookback] <= 0:
            continue
        ret = (closes[-1] / closes[-1 - lookback] - 1) * 100
        turnover = sum(s["turnover"]) / len(s["turnover"])
        ranked.append(MomentumPick(symbol, s["name"], ret, closes[-1], turnover))

    # Liquid universe first (top by turnover), then rank those by momentum.
    liquid = sorted(ranked, key=lambda p: p.avg_turnover, reverse=True)[:universe_size]
    liquid.sort(key=lambda p: p.trailing_return_pct, reverse=True)
    return liquid[:top]
