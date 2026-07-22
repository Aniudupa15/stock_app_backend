from datetime import UTC, datetime

from app.domain.entities import NewsArticle, NewsCategory
from app.repositories.news_repository import SqlAlchemyNewsRepository
from tests.conftest import requires_docker

pytestmark = requires_docker


def _article(
    url: str,
    headline: str = "Headline",
    symbols: list[str] | None = None,
    category=NewsCategory.MARKET,
    published_at=None,
) -> NewsArticle:
    return NewsArticle(
        headline=headline,
        summary="Summary",
        source="example.com",
        url=url,
        category=category,
        related_symbols=symbols or [],
        published_at=published_at or datetime(2026, 7, 22, tzinfo=UTC),
    )


async def test_bulk_upsert_is_idempotent_on_url(db_session):
    repo = SqlAlchemyNewsRepository(db_session)
    article = _article("https://example.com/1", headline="Original headline")

    first = await repo.bulk_upsert([article])
    second = await repo.bulk_upsert([_article("https://example.com/1", headline="Updated headline")])

    assert first == 1
    assert second == 1

    articles = await repo.list_latest(None, None, limit=10, offset=0)
    assert len(articles) == 1
    assert articles[0].headline == "Updated headline"


async def test_list_latest_orders_by_published_at_descending(db_session):
    repo = SqlAlchemyNewsRepository(db_session)
    await repo.bulk_upsert(
        [
            _article("https://example.com/old", published_at=datetime(2026, 7, 1, tzinfo=UTC)),
            _article("https://example.com/new", published_at=datetime(2026, 7, 22, tzinfo=UTC)),
        ]
    )

    articles = await repo.list_latest(None, None, limit=10, offset=0)

    assert [a.url for a in articles] == ["https://example.com/new", "https://example.com/old"]


async def test_list_latest_filters_by_category_and_symbol(db_session):
    repo = SqlAlchemyNewsRepository(db_session)
    await repo.bulk_upsert(
        [
            _article("https://example.com/market", category=NewsCategory.MARKET),
            _article("https://example.com/reliance", category=NewsCategory.COMPANY, symbols=["RELIANCE"]),
            _article("https://example.com/tcs", category=NewsCategory.COMPANY, symbols=["TCS"]),
        ]
    )

    by_category = await repo.list_latest(NewsCategory.COMPANY, None, limit=10, offset=0)
    by_symbol = await repo.list_latest(None, "RELIANCE", limit=10, offset=0)

    assert {a.url for a in by_category} == {"https://example.com/reliance", "https://example.com/tcs"}
    assert [a.url for a in by_symbol] == ["https://example.com/reliance"]


async def test_list_latest_respects_limit_and_offset(db_session):
    repo = SqlAlchemyNewsRepository(db_session)
    await repo.bulk_upsert(
        [_article(f"https://example.com/{i}", published_at=datetime(2026, 7, i + 1, tzinfo=UTC)) for i in range(5)]
    )

    page = await repo.list_latest(None, None, limit=2, offset=1)

    assert len(page) == 2
    assert page[0].url == "https://example.com/3"
