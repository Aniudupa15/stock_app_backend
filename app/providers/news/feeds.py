"""Free, publisher-syndicated RSS feeds - NSE itself publishes no general
market/economy news, only regulatory filings (see corporate actions/financial
results). Each feed's default category is used when an article can't be
tagged COMPANY (via a related-symbol match) or REGULATION (via keywords).
"""

from app.domain.entities import NewsCategory

RSS_FEEDS: list[tuple[str, NewsCategory]] = [
    ("https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms", NewsCategory.MARKET),
    ("https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms", NewsCategory.ECONOMY),
    ("https://www.moneycontrol.com/rss/latestnews.xml", NewsCategory.MARKET),
    ("https://www.moneycontrol.com/rss/marketreports.xml", NewsCategory.MARKET),
    ("https://www.moneycontrol.com/rss/business.xml", NewsCategory.ECONOMY),
]
