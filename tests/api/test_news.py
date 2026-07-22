from datetime import UTC, datetime

from app.domain.entities import NewsArticle, NewsCategory, StockMasterRecord
from app.repositories.news_repository import SqlAlchemyNewsRepository
from app.repositories.stock_repository import SqlAlchemyStockRepository


def _article(url: str, symbols: list[str] | None = None, category=NewsCategory.MARKET) -> NewsArticle:
    return NewsArticle(
        headline="Headline",
        summary="Summary",
        source="example.com",
        url=url,
        category=category,
        related_symbols=symbols or [],
        published_at=datetime(2026, 7, 22, tzinfo=UTC),
    )


async def test_list_news_filters_by_category(app_client, db_session):
    client, _ = app_client
    repo = SqlAlchemyNewsRepository(db_session)
    await repo.bulk_upsert(
        [
            _article("https://example.com/market", category=NewsCategory.MARKET),
            _article("https://example.com/company", category=NewsCategory.COMPANY, symbols=["RELIANCE"]),
        ]
    )

    resp = await client.get("/api/v1/news", params={"category": "COMPANY"})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["url"] == "https://example.com/company"


async def test_stock_news_endpoint_returns_matching_articles(app_client, db_session):
    client, _ = app_client
    stock_repo = SqlAlchemyStockRepository(db_session)
    await stock_repo.upsert_universe(
        [
            StockMasterRecord(
                symbol="RELIANCE", isin=None, name="Reliance", series="EQ", listing_date=None, face_value=None
            )
        ]
    )
    await db_session.commit()

    news_repo = SqlAlchemyNewsRepository(db_session)
    await news_repo.bulk_upsert(
        [
            _article("https://example.com/reliance", category=NewsCategory.COMPANY, symbols=["RELIANCE"]),
            _article("https://example.com/other", category=NewsCategory.COMPANY, symbols=["TCS"]),
        ]
    )

    resp = await client.get("/api/v1/stocks/RELIANCE/news")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["url"] == "https://example.com/reliance"


async def test_stock_news_unknown_symbol_returns_404(app_client):
    client, _ = app_client
    resp = await client.get("/api/v1/stocks/DOESNOTEXIST/news")
    assert resp.status_code == 404
