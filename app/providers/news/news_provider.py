import logging
import re

import httpx

from app.core.exceptions import ProviderUnavailableError
from app.domain.entities import NewsArticle, NewsCategory
from app.domain.ports import NewsProviderPort
from app.providers.news.feeds import RSS_FEEDS
from app.providers.news.rss_client import RssFetchError, fetch_feed

logger = logging.getLogger(__name__)

_REGULATION_KEYWORDS = ("rbi", "sebi", "regulation", "regulatory", "government", "budget", "ministry")
_MAX_RELATED_SYMBOLS = 10
_REQUEST_TIMEOUT_SECONDS = 10.0


def _build_symbol_pattern(known_symbols: set[str]) -> re.Pattern | None:
    if not known_symbols:
        return None
    # Longest-first so a symbol that's a prefix of another (rare, but
    # possible) doesn't shadow it under alternation.
    alternation = "|".join(re.escape(s) for s in sorted(known_symbols, key=len, reverse=True))
    return re.compile(rf"\b({alternation})\b", re.IGNORECASE)


# Known, accepted limitation: several real NSE symbols are also common
# English words (RELIANCE as in "self-reliance", GLOBAL, TOTAL, IDEA, ...).
# Word-boundary matching alone can't tell "Vodafone Idea" the company from
# "an idea" the noun - confirmed live (2026-07-22) against real ET headlines
# ("self-reliance" -> false-positive RELIANCE tag). A real fix needs company
# NAME matching (multi-word, capitalization-aware) or an NER model, both out
# of scope here; `related_symbols` should be treated as a best-effort hint,
# not a reliable tag.


def _categorize(text: str, related_symbols: list[str], default: NewsCategory) -> NewsCategory:
    if related_symbols:
        return NewsCategory.COMPANY
    lowered = text.lower()
    if any(keyword in lowered for keyword in _REGULATION_KEYWORDS):
        return NewsCategory.REGULATION
    return default


class RssNewsProvider(NewsProviderPort):
    async def fetch_latest(self, known_symbols: set[str]) -> list[NewsArticle]:
        pattern = _build_symbol_pattern(known_symbols)
        articles: list[NewsArticle] = []
        failed_feeds = 0

        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
            for url, default_category in RSS_FEEDS:
                try:
                    items = await fetch_feed(client, url)
                except RssFetchError as exc:
                    failed_feeds += 1
                    logger.warning("news feed unavailable, skipping: %s", exc)
                    continue

                source = httpx.URL(url).host
                for item in items:
                    text = f"{item.title} {item.description or ''}"
                    related = sorted({m.group(1).upper() for m in pattern.finditer(text)}) if pattern else []
                    articles.append(
                        NewsArticle(
                            headline=item.title,
                            summary=item.description,
                            source=source,
                            url=item.link,
                            category=_categorize(text, related, default_category),
                            related_symbols=related[:_MAX_RELATED_SYMBOLS],
                            published_at=item.published_at,
                        )
                    )

        if failed_feeds == len(RSS_FEEDS):
            raise ProviderUnavailableError("news", "all RSS feeds unreachable")
        return articles
