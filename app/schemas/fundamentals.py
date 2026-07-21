from datetime import date
from decimal import Decimal

from pydantic import BaseModel


class FundamentalsOut(BaseModel):
    symbol: str
    has_data: bool = False
    latest_period_end: date | None = None

    revenue_growth_yoy: Decimal | None = None
    revenue_growth_qoq: Decimal | None = None
    profit_growth_yoy: Decimal | None = None
    profit_growth_qoq: Decimal | None = None
    ttm_eps: Decimal | None = None
    pe_ratio: Decimal | None = None
    dividend_yield: Decimal | None = None

    # No confirmed free bulk data source exists for these (they need Balance
    # Sheet/Annual Report filings, a different category than the quarterly
    # "Financial Results" filings this service parses) - see Phase 3 plan.
    # Always null; kept visible in the schema rather than silently omitted.
    book_value: Decimal | None = None
    roe: Decimal | None = None
    roce: Decimal | None = None
    debt_to_equity: Decimal | None = None
