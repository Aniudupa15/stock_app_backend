from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities import NewsArticle, NewsCategory
from app.domain.ports import NewsRepositoryPort
from app.models.news_article import NewsArticleModel

_UPSERT_BATCH_SIZE = 200


def _to_entity(row: NewsArticleModel) -> NewsArticle:
    return NewsArticle(
        headline=row.headline,
        summary=row.summary,
        source=row.source,
        url=row.url,
        category=row.category,
        related_symbols=list(row.related_symbols),
        published_at=row.published_at,
    )


class SqlAlchemyNewsRepository(NewsRepositoryPort):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def bulk_upsert(self, articles: list[NewsArticle]) -> int:
        if not articles:
            return 0

        rows = [
            {
                "headline": a.headline,
                "summary": a.summary,
                "source": a.source,
                "url": a.url,
                "category": a.category,
                "related_symbols": a.related_symbols,
                "published_at": a.published_at,
            }
            for a in articles
        ]

        upserted = 0
        for i in range(0, len(rows), _UPSERT_BATCH_SIZE):
            batch = rows[i : i + _UPSERT_BATCH_SIZE]
            stmt = pg_insert(NewsArticleModel).values(batch)
            stmt = stmt.on_conflict_do_update(
                index_elements=["url"],
                set_={
                    "headline": stmt.excluded.headline,
                    "summary": stmt.excluded.summary,
                    "category": stmt.excluded.category,
                    "related_symbols": stmt.excluded.related_symbols,
                },
            )
            await self._session.execute(stmt)
            upserted += len(batch)

        await self._session.commit()
        return upserted

    async def list_latest(
        self, category: NewsCategory | None, symbol: str | None, limit: int, offset: int
    ) -> list[NewsArticle]:
        stmt = select(NewsArticleModel).order_by(NewsArticleModel.published_at.desc())
        if category is not None:
            stmt = stmt.where(NewsArticleModel.category == category)
        if symbol is not None:
            stmt = stmt.where(NewsArticleModel.related_symbols.any(symbol.strip().upper()))
        stmt = stmt.offset(offset).limit(limit)

        result = await self._session.execute(stmt)
        return [_to_entity(row) for row in result.scalars().all()]
