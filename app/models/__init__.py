"""Importing this package registers every ORM model on `Base.metadata` -
required so foreign keys between models (e.g. `watchlists.user_id` ->
`users.id`) resolve correctly, since a model whose module nothing else
happens to import would otherwise never reach SQLAlchemy's metadata.
"""

from app.models import (  # noqa: F401
    alert,
    corporate_action,
    financial_result,
    historical_price,
    intraday_signal_snapshot,
    ipo_filing,
    long_term_signal_snapshot,
    news_article,
    notification,
    portfolio,
    portfolio_transaction,
    refresh_token,
    search_history,
    stock,
    stock_indicator_snapshot,
    user,
    watchlist,
    watchlist_item,
)
