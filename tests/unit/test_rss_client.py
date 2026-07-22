from datetime import UTC, datetime

import httpx
import pytest
import respx

from app.providers.news.rss_client import RssFetchError, fetch_feed

_SAMPLE_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Sample Feed</title>
    <item>
      <title>RELIANCE shares surge on strong Q3 results</title>
      <description>Reliance Industries posted a strong quarter.</description>
      <link>https://example.com/article1</link>
      <pubDate>Wed, 22 Jul 2026 09:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Market closes flat amid mixed global cues</title>
      <description></description>
      <link>https://example.com/article2</link>
    </item>
    <item>
      <!-- missing title, should be skipped -->
      <link>https://example.com/article3</link>
    </item>
  </channel>
</rss>
"""


@respx.mock
async def test_fetch_feed_parses_items_and_skips_incomplete_ones():
    respx.get("https://example.com/feed.xml").mock(return_value=httpx.Response(200, content=_SAMPLE_FEED))

    async with httpx.AsyncClient() as client:
        items = await fetch_feed(client, "https://example.com/feed.xml")

    assert len(items) == 2
    assert items[0].title == "RELIANCE shares surge on strong Q3 results"
    assert items[0].link == "https://example.com/article1"
    assert items[0].published_at == datetime(2026, 7, 22, 9, 0, 0, tzinfo=UTC)
    assert items[1].description is None


@respx.mock
async def test_fetch_feed_raises_rss_fetch_error_on_http_failure():
    respx.get("https://example.com/feed.xml").mock(return_value=httpx.Response(500))

    async with httpx.AsyncClient() as client:
        with pytest.raises(RssFetchError):
            await fetch_feed(client, "https://example.com/feed.xml")


@respx.mock
async def test_fetch_feed_raises_rss_fetch_error_on_malformed_xml():
    respx.get("https://example.com/feed.xml").mock(return_value=httpx.Response(200, content=b"not xml"))

    async with httpx.AsyncClient() as client:
        with pytest.raises(RssFetchError):
            await fetch_feed(client, "https://example.com/feed.xml")
