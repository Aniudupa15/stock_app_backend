from datetime import date
from decimal import Decimal

import pytest

from app.core.exceptions import StockNotFoundError
from app.domain.entities import BhavcopyRecord, CorporateAction, FinancialResultRecord
from app.services.fundamentals_service import FundamentalsService
from app.services.long_term_signal_service import LongTermSignalService
from tests.conftest import (
    FakeCorporateActionRepository,
    FakeFinancialResultRepository,
    FakeHistoricalPriceRepository,
    FakeStockRepository,
)


def _quarter(period_end: date, revenue: str, profit: str, eps: str) -> FinancialResultRecord:
    return FinancialResultRecord(
        symbol="RELIANCE",
        period_start=date(period_end.year, 1, 1),
        period_end=period_end,
        consolidated=False,
        revenue=Decimal(revenue),
        profit=Decimal(profit),
        eps_basic=Decimal(eps),
        eps_diluted=Decimal(eps),
    )


def _build_service(quarters: list[FinancialResultRecord], sample_stock, dividend_purpose: str | None = None):
    stock_repo = FakeStockRepository([sample_stock])
    financial_repo = FakeFinancialResultRepository({"RELIANCE": quarters})
    price_repo = FakeHistoricalPriceRepository()
    corp_repo = FakeCorporateActionRepository()
    if dividend_purpose:
        corp_repo = FakeCorporateActionRepository(
            {
                "RELIANCE": [
                    CorporateAction(
                        symbol="RELIANCE",
                        purpose=dividend_purpose,
                        face_value=None,
                        ex_date=date.today(),
                        record_date=None,
                        book_closure_start=None,
                        book_closure_end=None,
                    )
                ]
            }
        )
    fundamentals_service = FundamentalsService(stock_repo, financial_repo, price_repo, corp_repo)
    return LongTermSignalService(stock_repo, fundamentals_service)


async def test_get_signal_raises_when_stock_unknown():
    stock_repo = FakeStockRepository()
    service = LongTermSignalService(
        stock_repo,
        FundamentalsService(stock_repo, FakeFinancialResultRepository(), FakeHistoricalPriceRepository(), FakeCorporateActionRepository()),
    )
    with pytest.raises(StockNotFoundError):
        await service.get_signal("DOESNOTEXIST")


async def test_no_financial_data_returns_has_data_false(sample_stock):
    service = _build_service([], sample_stock)
    result = await service.get_signal("RELIANCE")
    assert result.has_data is False


async def test_strong_growth_and_dividend_yields_buy(sample_stock):
    quarters = [
        _quarter(date(2024, 12, 31), "1000", "150", "5.0"),
        _quarter(date(2023, 12, 31), "800", "100", "4.0"),  # YoY: revenue +25%, profit +50%
    ]
    service = _build_service(quarters, sample_stock)
    result = await service.get_signal("RELIANCE")

    assert result.has_data is True
    assert result.signal == "BUY"
    assert result.growth_potential == "High"
    assert any("Revenue grew" in s for s in result.strengths)
    assert any("Profit grew" in s for s in result.strengths)


async def test_declining_growth_yields_avoid(sample_stock):
    quarters = [
        _quarter(date(2024, 12, 31), "800", "50", "2.0"),
        _quarter(date(2023, 12, 31), "1000", "100", "4.0"),  # YoY: revenue -20%, profit -50%
    ]
    service = _build_service(quarters, sample_stock)
    result = await service.get_signal("RELIANCE")

    assert result.signal == "AVOID"
    assert result.risk_level == "High"
    assert any("declined" in w for w in result.weaknesses)


async def test_modest_growth_yields_hold(sample_stock):
    quarters = [
        _quarter(date(2024, 12, 31), "1000", "100", "5.0"),
        _quarter(date(2023, 12, 31), "980", "98", "4.9"),  # YoY: ~2% growth both - modest
    ]
    service = _build_service(quarters, sample_stock)
    result = await service.get_signal("RELIANCE")

    assert result.signal == "HOLD"


async def test_margin_pressure_risk_flagged_when_profit_declines_but_revenue_grows(sample_stock):
    quarters = [
        _quarter(date(2024, 12, 31), "1200", "80", "3.0"),
        _quarter(date(2023, 12, 31), "1000", "100", "4.0"),  # revenue +20%, profit -20%
    ]
    service = _build_service(quarters, sample_stock)
    result = await service.get_signal("RELIANCE")

    assert any("margin pressure" in r.lower() for r in result.risks)


async def test_pe_ratio_never_scored_only_referenced_in_reasoning(sample_stock):
    quarters = [
        _quarter(date(2024, 12, 31), "1000", "100", "5.0"),
        _quarter(date(2023, 12, 31), "1000", "100", "5.0"),  # flat growth - no score contribution
        _quarter(date(2024, 9, 30), "900", "90", "4.5"),
        _quarter(date(2024, 6, 30), "850", "80", "4.0"),
        _quarter(date(2024, 3, 31), "800", "70", "3.5"),
    ]
    stock_repo = FakeStockRepository([sample_stock])
    financial_repo = FakeFinancialResultRepository({"RELIANCE": quarters})
    price_repo = FakeHistoricalPriceRepository()
    await price_repo.bulk_upsert_bars(
        [
            BhavcopyRecord(
                symbol="RELIANCE",
                trade_date=date.today(),
                open=Decimal("100"),
                high=Decimal("101"),
                low=Decimal("99"),
                close=Decimal("100.00"),
                volume=1000,
            )
        ]
    )
    fundamentals_service = FundamentalsService(stock_repo, financial_repo, price_repo, FakeCorporateActionRepository())
    service = LongTermSignalService(stock_repo, fundamentals_service)

    result = await service.get_signal("RELIANCE")

    assert any("PE ratio" in r and "no sector-relative" in r for r in result.reasoning)
