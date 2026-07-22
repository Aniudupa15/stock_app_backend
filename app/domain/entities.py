"""Framework-agnostic domain entities. No FastAPI, no SQLAlchemy, no NSE-specific shapes."""

import uuid
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum


class InstrumentType(str, Enum):
    EQUITY = "EQUITY"
    ETF = "ETF"
    REIT = "REIT"
    INVIT = "INVIT"


@dataclass(frozen=True, slots=True)
class Stock:
    symbol: str
    isin: str | None
    name: str
    series: str | None
    sector: str | None
    industry: str | None
    instrument_type: InstrumentType
    listing_date: date | None
    face_value: Decimal | None
    is_active: bool


@dataclass(frozen=True, slots=True)
class StockMasterRecord:
    """One row of the NSE equity master, as pulled from the universe sync source."""

    symbol: str
    isin: str | None
    name: str
    series: str | None
    listing_date: date | None
    face_value: Decimal | None


@dataclass(frozen=True, slots=True)
class Quote:
    symbol: str
    last_price: Decimal
    change: Decimal
    change_percent: Decimal
    open: Decimal
    high: Decimal
    low: Decimal
    previous_close: Decimal
    volume: int
    as_of: datetime


@dataclass(frozen=True, slots=True)
class BhavcopyRecord:
    """One equity row from an NSE daily Bhavcopy archive, already filtered
    down from the full Cash Market file (which also contains SGBs, T-Bills,
    and other non-equity instruments).
    """

    symbol: str
    trade_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


@dataclass(frozen=True, slots=True)
class OhlcvBar:
    """A single daily price bar, as read back from persistence for indicator
    computation and chart rendering.
    """

    trade_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


@dataclass(frozen=True, slots=True)
class CorporateAction:
    symbol: str
    purpose: str
    face_value: Decimal | None
    ex_date: date | None
    record_date: date | None
    book_closure_start: date | None
    book_closure_end: date | None


@dataclass(frozen=True, slots=True)
class FinancialResultFiling:
    """One row from the NSE financial-results filing index - metadata about a
    quarterly filing, not yet the parsed figures inside it.
    """

    symbol: str
    period_start: date
    period_end: date
    consolidated: bool
    xbrl_url: str


@dataclass(frozen=True, slots=True)
class FinancialResultRecord:
    """Parsed quarter figures from one filing's XBRL document."""

    symbol: str
    period_start: date
    period_end: date
    consolidated: bool
    revenue: Decimal | None
    profit: Decimal | None
    eps_basic: Decimal | None
    eps_diluted: Decimal | None


@dataclass(frozen=True, slots=True)
class MarketMover:
    """One row of a gainers/losers/most-active/52-week leaderboard, derived
    entirely from stored `historical_prices` (no live NSE call).
    """

    symbol: str
    name: str
    last_price: Decimal
    change: Decimal | None
    change_percent: Decimal | None
    volume: int


@dataclass(frozen=True, slots=True)
class Watchlist:
    id: uuid.UUID
    user_id: uuid.UUID
    name: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class WatchlistItem:
    symbol: str
    name: str
    added_at: datetime


class TransactionType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True, slots=True)
class Portfolio:
    id: uuid.UUID
    user_id: uuid.UUID
    name: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class PortfolioTransaction:
    symbol: str
    transaction_type: TransactionType
    quantity: Decimal
    price: Decimal
    transaction_date: date


class NewsCategory(str, Enum):
    MARKET = "MARKET"
    COMPANY = "COMPANY"
    ECONOMY = "ECONOMY"
    REGULATION = "REGULATION"
    SECTOR = "SECTOR"


@dataclass(frozen=True, slots=True)
class NewsArticle:
    headline: str
    summary: str | None
    source: str
    url: str
    category: NewsCategory
    related_symbols: list[str]
    published_at: datetime


class AlertType(str, Enum):
    PRICE_ABOVE = "PRICE_ABOVE"
    PRICE_BELOW = "PRICE_BELOW"
    PERCENT_CHANGE_ABOVE = "PERCENT_CHANGE_ABOVE"
    PERCENT_CHANGE_BELOW = "PERCENT_CHANGE_BELOW"
    RSI_ABOVE = "RSI_ABOVE"
    RSI_BELOW = "RSI_BELOW"
    VOLUME_SPIKE = "VOLUME_SPIKE"
    NEW_52_WEEK_HIGH = "NEW_52_WEEK_HIGH"
    NEW_52_WEEK_LOW = "NEW_52_WEEK_LOW"


class AlertStatus(str, Enum):
    ACTIVE = "ACTIVE"
    TRIGGERED = "TRIGGERED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class Alert:
    id: uuid.UUID
    user_id: uuid.UUID
    symbol: str
    alert_type: AlertType
    condition: dict
    status: AlertStatus
    created_at: datetime
    triggered_at: datetime | None


@dataclass(frozen=True, slots=True)
class Notification:
    id: uuid.UUID
    user_id: uuid.UUID
    alert_id: uuid.UUID | None
    title: str
    message: str
    created_at: datetime
    read_at: datetime | None


@dataclass(frozen=True, slots=True)
class MarketStatus:
    """One market segment's open/closed status, as NSE reports it (raw
    `market`/`status` strings from NSE's own vocabulary - not normalized into
    an enum, since NSE's own values aren't fully cataloged/stable enough to
    trust an exhaustive enum here).
    """

    market: str
    status: str
    as_of: str


@dataclass(frozen=True, slots=True)
class IndexQuote:
    index_name: str
    last_price: Decimal
    change: Decimal
    change_percent: Decimal


@dataclass(frozen=True, slots=True)
class User:
    """Only used within the auth service/repository layer - every other
    feature (watchlists, portfolios, alerts, ...) works with a bare
    `user_id: uuid.UUID`, never a full User, so `password_hash` being here
    never leaks beyond auth code.
    """

    id: uuid.UUID
    email: str
    display_name: str
    password_hash: str | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RefreshToken:
    id: uuid.UUID
    user_id: uuid.UUID
    token_hash: str
    expires_at: datetime
    revoked_at: datetime | None
    created_at: datetime
