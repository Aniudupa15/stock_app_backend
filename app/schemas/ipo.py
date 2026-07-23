from datetime import date

from pydantic import BaseModel


class IpoFilingOut(BaseModel):
    symbol: str
    company_name: str
    status: str
    price_range: str | None
    issue_size: str | None
    issue_start_date: date | None
    issue_end_date: date | None
    listing_date: date | None
    series: str | None
