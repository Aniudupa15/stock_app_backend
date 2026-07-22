from datetime import UTC, datetime

import pytest

from app.core.exceptions import StockNotFoundError
from app.domain.entities import NewsArticle, NewsCategory
from app.services.news_service import NewsService
from tests.conftest import FakeNewsProvider, FakeNewsRepository, FakeStockRepository


def _article(url: str, symbols: list[str] | None = None, category: NewsCategory = NewsCategory.MARKET) -> NewsArticle:
    return NewsArticle(
        headline="Headline",
        summary="Summary",
        source="example.com",
        url=url,
        category=category,
        related_symbols=symbols or [],
        published_at=datetime(2026, 7, 22, tzinfo=UTC),
    )


async def test_sync_passes_active_symbols_to_provider_and_upserts(sample_stock):
    stock_repo = FakeStockRepository([sample_stock])
    provider = FakeNewsProvider(articles=[_article("https://example.com/1")])
    repository = FakeNewsRepository()
    service = NewsService(provider, repository, stock_repo)

    upserted = await service.sync()

    assert upserted == 1
    assert provider.last_known_symbols == {"RELIANCE"}
    assert "https://example.com/1" in repository.articles_by_url


async def test_list_latest_filters_by_category():
    repository = FakeNewsRepository(
        articles=[
            _article("https://example.com/market", category=NewsCategory.MARKET),
            _article("https://example.com/company", category=NewsCategory.COMPANY),
        ]
    )
    service = NewsService(FakeNewsProvider(), repository, FakeStockRepository())

    result = await service.list_latest(NewsCategory.COMPANY, None, limit=10, offset=0)

    assert len(result) == 1
    assert result[0].url == "https://example.com/company"


async def test_get_for_symbol_unknown_symbol_raises():
    service = NewsService(FakeNewsProvider(), FakeNewsRepository(), FakeStockRepository())

    with pytest.raises(StockNotFoundError):
        await service.get_for_symbol("DOESNOTEXIST")


async def test_get_for_symbol_returns_matching_articles(sample_stock):
    stock_repo = FakeStockRepository([sample_stock])
    repository = FakeNewsRepository(
        articles=[
            _article("https://example.com/reliance", symbols=["RELIANCE"]),
            _article("https://example.com/other", symbols=["TCS"]),
        ]
    )
    service = NewsService(FakeNewsProvider(), repository, stock_repo)

    result = await service.get_for_symbol("reliance")

    assert len(result) == 1
    assert result[0].url == "https://example.com/reliance"
