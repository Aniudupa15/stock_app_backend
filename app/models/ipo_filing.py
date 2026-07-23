from datetime import date

from sqlalchemy import BigInteger, Date, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class IpoFilingModel(Base, TimestampMixin):
    """Not FK'd to `stocks` - a pre-IPO/upcoming company generally isn't in
    the equity universe yet, only listed ones eventually are.
    """

    __tablename__ = "ipo_filings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    company_name: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    price_range: Mapped[str | None] = mapped_column(String(64), nullable=True)
    issue_size: Mapped[str | None] = mapped_column(String(64), nullable=True)
    issue_start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    issue_end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    listing_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    series: Mapped[str | None] = mapped_column(String(8), nullable=True)
