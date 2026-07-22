"""Fetches and parses standard RSS 2.0 feeds (channel/item/title/description/
link/pubDate) via stdlib ElementTree - the format is simple enough that the
`feedparser` dependency isn't needed, same reasoning as XBRL parsing in
Phase 3's fundamentals ingestion.
"""

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2
_USER_AGENT = "Mozilla/5.0 (compatible; StockAppNewsBot/1.0)"


class RssFetchError(Exception):
    """One feed failed to fetch or parse - callers catch this per-feed so a
    single dead feed doesn't take down the whole sync."""


@dataclass(frozen=True, slots=True)
class RssItem:
    title: str
    description: str | None
    link: str
    published_at: datetime


async def _fetch_once(client: httpx.AsyncClient, url: str) -> bytes:
    resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
    resp.raise_for_status()
    return resp.content


async def fetch_feed(client: httpx.AsyncClient, url: str) -> list[RssItem]:
    retrying = retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        reraise=True,
    )(_fetch_once)

    try:
        content = await retrying(client, url)
    except httpx.HTTPError as exc:
        raise RssFetchError(f"{url}: {exc}") from exc

    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise RssFetchError(f"{url}: malformed XML ({exc})") from exc

    items = []
    for item_el in root.findall("./channel/item"):
        title = (item_el.findtext("title") or "").strip()
        link = (item_el.findtext("link") or "").strip()
        if not title or not link:
            continue
        description = item_el.findtext("description")
        items.append(
            RssItem(
                title=title,
                description=description.strip() if description else None,
                link=link,
                published_at=_parse_pub_date(item_el.findtext("pubDate")),
            )
        )
    return items


def _parse_pub_date(raw: str | None) -> datetime:
    if raw:
        try:
            parsed = parsedate_to_datetime(raw)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except (TypeError, ValueError):
            logger.debug("unparseable RSS pubDate %r, falling back to now", raw)
    return datetime.now(UTC)
