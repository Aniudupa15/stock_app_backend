from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.domain.entities import MarketMover
from app.domain.ports import MarketMoverRepositoryPort
from app.models.historical_price import HistoricalPriceModel
from app.models.stock import StockModel

# Rows the leaderboard ranks over are drawn from this many trailing sessions
# (partitioned per stock, most-recent-first) so a single query can serve
# "latest close", "close N sessions back", and "252-session high/low" at once.
_WINDOW_SESSIONS = 252


def _rows_to_movers(result) -> list[MarketMover]:
    return [
        MarketMover(
            symbol=row.symbol,
            name=row.name,
            last_price=row.last_price,
            change=row.change,
            change_percent=row.change_percent,
            volume=row.volume,
        )
        for row in result
    ]


class SqlAlchemyMarketMoverRepository(MarketMoverRepositoryPort):
    def __init__(self, session: AsyncSession):
        self._session = session

    def _ranked_subquery(self):
        return (
            select(
                HistoricalPriceModel.stock_id,
                HistoricalPriceModel.close,
                HistoricalPriceModel.volume,
                func.row_number()
                .over(
                    partition_by=HistoricalPriceModel.stock_id,
                    order_by=HistoricalPriceModel.trade_date.desc(),
                )
                .label("rn"),
            )
            .where(HistoricalPriceModel.close.is_not(None))
            .subquery()
        )

    def _latest_with_day_change_stmt(self, ranked):
        """`latest` (rn=1) left-joined to `previous` (rn=2) for a 1-day change,
        joined to `stocks` - the shared shape behind most-active, 52-week
        extremes, and ad-hoc latest-price lookups (watchlists/portfolios).
        """
        latest = aliased(ranked)
        previous = aliased(ranked)
        change = latest.c.close - previous.c.close
        change_percent = (change / previous.c.close) * 100

        stmt = (
            select(
                StockModel.symbol,
                StockModel.name,
                latest.c.close.label("last_price"),
                latest.c.volume.label("volume"),
                change.label("change"),
                change_percent.label("change_percent"),
            )
            .select_from(latest)
            .outerjoin(previous, (previous.c.stock_id == latest.c.stock_id) & (previous.c.rn == 2))
            .join(StockModel, StockModel.id == latest.c.stock_id)
            .where(latest.c.rn == 1)
        )
        return stmt, latest, change_percent

    async def get_top_movers(self, direction: str, lookback_sessions: int, limit: int) -> list[MarketMover]:
        ranked = self._ranked_subquery()
        latest = aliased(ranked)
        past = aliased(ranked)

        change = latest.c.close - past.c.close
        change_percent = (change / past.c.close) * 100

        stmt = (
            select(
                StockModel.symbol,
                StockModel.name,
                latest.c.close.label("last_price"),
                latest.c.volume.label("volume"),
                change.label("change"),
                change_percent.label("change_percent"),
            )
            .select_from(latest)
            .join(past, past.c.stock_id == latest.c.stock_id)
            .join(StockModel, StockModel.id == latest.c.stock_id)
            .where(
                latest.c.rn == 1,
                past.c.rn == lookback_sessions + 1,
                StockModel.is_active.is_(True),
            )
        )
        stmt = stmt.order_by(change_percent.desc() if direction == "gainers" else change_percent.asc()).limit(limit)

        result = await self._session.execute(stmt)
        return _rows_to_movers(result)

    async def get_most_active(self, limit: int) -> list[MarketMover]:
        ranked = self._ranked_subquery()
        stmt, latest, _ = self._latest_with_day_change_stmt(ranked)
        stmt = stmt.where(StockModel.is_active.is_(True)).order_by(latest.c.volume.desc()).limit(limit)

        result = await self._session.execute(stmt)
        return _rows_to_movers(result)

    async def get_52_week_extremes(self, direction: str, limit: int) -> list[MarketMover]:
        ranked = self._ranked_subquery()
        stmt, latest, change_percent = self._latest_with_day_change_stmt(ranked)

        window_agg = (
            select(
                ranked.c.stock_id,
                func.max(ranked.c.close).label("period_high"),
                func.min(ranked.c.close).label("period_low"),
            )
            .where(ranked.c.rn <= _WINDOW_SESSIONS)
            .group_by(ranked.c.stock_id)
            .subquery()
        )

        stmt = stmt.join(window_agg, window_agg.c.stock_id == latest.c.stock_id).where(StockModel.is_active.is_(True))
        if direction == "high":
            stmt = stmt.where(latest.c.close >= window_agg.c.period_high).order_by(change_percent.desc().nulls_last())
        else:
            stmt = stmt.where(latest.c.close <= window_agg.c.period_low).order_by(change_percent.asc().nulls_last())
        stmt = stmt.limit(limit)

        result = await self._session.execute(stmt)
        return _rows_to_movers(result)

    async def get_latest_prices(self, symbols: list[str]) -> list[MarketMover]:
        if not symbols:
            return []

        normalized = [s.strip().upper() for s in symbols]
        ranked = self._ranked_subquery()
        stmt, _, _ = self._latest_with_day_change_stmt(ranked)
        stmt = stmt.where(StockModel.symbol.in_(normalized))

        result = await self._session.execute(stmt)
        return _rows_to_movers(result)
