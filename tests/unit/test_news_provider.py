import httpx
import pytest
import respx

from app.core.exceptions import ProviderUnavailableError
from app.domain.entities import NewsCategory
from app.providers.news.feeds import RSS_FEEDS
from app.providers.news.news_provider import RssNewsProvider

_RELIANCE_ITEM = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
<item>
  <title>RELIANCE posts record profit this quarter</title>
  <description>Strong results from Reliance Industries.</description>
  <link>https://example.com/reliance-news</link>
  <pubDate>Wed, 22 Jul 2026 09:00:00 GMT</pubDate>
</item>
</channel></rss>
"""

_REGULATION_ITEM = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
<item>
  <title>SEBI tightens regulation on market intermediaries</title>
  <description>New rules from the regulator.</description>
  <link>https://example.com/sebi-news</link>
  <pubDate>Wed, 22 Jul 2026 10:00:00 GMT</pubDate>
</item>
</channel></rss>
"""

_GENERIC_ITEM = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
<item>
  <title>Sensex ends higher amid global rally</title>
  <description>Broad market update.</description>
  <link>https://example.com/generic-news</link>
  <pubDate>Wed, 22 Jul 2026 11:00:00 GMT</pubDate>
</item>
</channel></rss>
"""

_EMPTY_FEED = b'<?xml version="1.0"?><rss version="2.0"><channel></channel></rss>'


def _mock_all_feeds(bodies_by_index: dict[int, bytes] | None = None, fail_indices: set[int] | None = None):
    bodies_by_index = bodies_by_index or {}
    fail_indices = fail_indices or set()
    for i, (url, _category) in enumerate(RSS_FEEDS):
        if i in fail_indices:
            respx.get(url).mock(return_value=httpx.Response(500))
        else:
            respx.get(url).mock(return_value=httpx.Response(200, content=bodies_by_index.get(i, _EMPTY_FEED)))


@respx.mock
async def test_article_mentioning_known_symbol_is_tagged_company():
    _mock_all_feeds({0: _RELIANCE_ITEM})

    provider = RssNewsProvider()
    articles = await provider.fetch_latest({"RELIANCE", "TCS"})

    reliance_articles = [a for a in articles if a.url == "https://example.com/reliance-news"]
    assert len(reliance_articles) == 1
    assert reliance_articles[0].category == NewsCategory.COMPANY
    assert reliance_articles[0].related_symbols == ["RELIANCE"]


@respx.mock
async def test_article_with_regulation_keyword_is_tagged_regulation():
    _mock_all_feeds({0: _REGULATION_ITEM})

    provider = RssNewsProvider()
    articles = await provider.fetch_latest(set())

    matches = [a for a in articles if a.url == "https://example.com/sebi-news"]
    assert len(matches) == 1
    assert matches[0].category == NewsCategory.REGULATION


@respx.mock
async def test_article_with_no_symbol_or_keyword_match_uses_feed_default_category():
    _mock_all_feeds({0: _GENERIC_ITEM})

    provider = RssNewsProvider()
    articles = await provider.fetch_latest(set())

    matches = [a for a in articles if a.url == "https://example.com/generic-news"]
    assert len(matches) == 1
    assert matches[0].category == RSS_FEEDS[0][1]
    assert matches[0].related_symbols == []


@respx.mock
async def test_partial_feed_failure_still_returns_articles_from_working_feeds():
    _mock_all_feeds({0: _GENERIC_ITEM}, fail_indices={1, 2, 3, 4})

    provider = RssNewsProvider()
    articles = await provider.fetch_latest(set())

    assert len(articles) == 1


@respx.mock
async def test_all_feeds_failing_raises_provider_unavailable():
    _mock_all_feeds(fail_indices=set(range(len(RSS_FEEDS))))

    provider = RssNewsProvider()
    with pytest.raises(ProviderUnavailableError):
        await provider.fetch_latest(set())
