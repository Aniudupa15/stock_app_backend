"""Framework-agnostic domain entities. No FastAPI, no SQLAlchemy, no NSE-specific shapes."""

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
