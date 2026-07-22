from datetime import date
from decimal import Decimal

import pytest

from app.core.exceptions import StockNotFoundError
from app.domain.entities import BhavcopyRecord, CorporateAction, FinancialResultRecord
from app.services.fundamentals_service import FundamentalsService, _sum_dividend_amount
from tests.conftest import (
    FakeCorporateActionRepository,
    FakeFinancialResultRepository,
    FakeHistoricalPriceRepository,
    FakeStockRepository,
)


def _quarter(
    period_end: date, revenue: str, profit: str, eps: str, consolidated: bool = False
) -> FinancialResultRecord:
    return FinancialResultRecord(
        symbol="RELIANCE",
        period_start=date(period_end.year, 1, 1),
        period_end=period_end,
        consolidated=consolidated,
        revenue=Decimal(revenue),
        profit=Decimal(profit),
        eps_basic=Decimal(eps),
        eps_diluted=Decimal(eps),
    )


@pytest.mark.parametrize(
    "purpose,expected",
    [
        ("Dividend - Rs 10 Per Share/Special Dividend - Rs 30 Per Share", Decimal("40")),
        ("Interim Dividend - Rs 4 Per Share", Decimal("4")),
        ("Dividend - Re 0.70 Per Share", Decimal("0.70")),
        ("Face Value Split (Sub-Division) - From Rs 5/- Per Share To Rs 2/- Per Share", Decimal("0")),
    ],
)
def test_sum_dividend_amount(purpose, expected):
    assert _sum_dividend_amount(purpose) == expected


async def test_get_fundamentals_raises_when_stock_unknown():
    service = FundamentalsService(
        FakeStockRepository(),
        FakeFinancialResultRepository(),
        FakeHistoricalPriceRepository(),
        FakeCorporateActionRepository(),
    )
    with pytest.raises(StockNotFoundError):
        await service.get_fundamentals("DOESNOTEXIST")


async def test_get_fundamentals_has_data_false_with_no_quarters(sample_stock):
    service = FundamentalsService(
        FakeStockRepository([sample_stock]),
        FakeFinancialResultRepository(),
        FakeHistoricalPriceRepository(),
        FakeCorporateActionRepository(),
    )
    result = await service.get_fundamentals("RELIANCE")
    assert result.has_data is False


async def test_growth_rates_computed_from_comparable_quarters(sample_stock):
    # 5 quarters: latest (Q4), one QoQ back (~90d), one YoY back (~365d)
    quarters = [
        _quarter(date(2024, 12, 31), "1000", "100", "5.0"),  # latest
        _quarter(date(2024, 9, 30), "900", "90", "4.5"),  # QoQ comparable (~92 days back)
        _quarter(date(2024, 6, 30), "850", "80", "4.0"),
        _quarter(date(2024, 3, 31), "800", "70", "3.5"),
        _quarter(date(2023, 12, 31), "800", "80", "4.0"),  # YoY comparable (~366 days back)
    ]
    financial_repo = FakeFinancialResultRepository({"RELIANCE": quarters})
    service = FundamentalsService(
        FakeStockRepository([sample_stock]),
        financial_repo,
        FakeHistoricalPriceRepository(),
        FakeCorporateActionRepository(),
    )

    result = await service.get_fundamentals("RELIANCE")

    assert result.has_data is True
    assert result.latest_period_end == date(2024, 12, 31)
    # revenue: (1000-900)/900*100 = 11.11
    assert result.revenue_growth_qoq == pytest.approx(Decimal("11.11"))
    # revenue: (1000-800)/800*100 = 25.0
    assert result.revenue_growth_yoy == pytest.approx(Decimal("25.0"))
    # profit: (100-90)/90*100 = 11.11
    assert result.profit_growth_qoq == pytest.approx(Decimal("11.11"))
    # profit: (100-80)/80*100 = 25.0
    assert result.profit_growth_yoy == pytest.approx(Decimal("25.0"))
    # TTM EPS = sum of latest 4 quarters' eps_basic = 5.0+4.5+4.0+3.5 = 17.0
    assert result.ttm_eps == Decimal("17.0000")


async def test_ttm_eps_and_pe_ratio_require_four_quarters_and_price(sample_stock):
    quarters = [
        _quarter(date(2024, 12, 31), "1000", "100", "5.0"),
        _quarter(date(2024, 9, 30), "900", "90", "4.5"),
    ]
    financial_repo = FakeFinancialResultRepository({"RELIANCE": quarters})
    price_repo = FakeHistoricalPriceRepository()
    service = FundamentalsService(
        FakeStockRepository([sample_stock]), financial_repo, price_repo, FakeCorporateActionRepository()
    )

    result = await service.get_fundamentals("RELIANCE")

    assert result.ttm_eps is None  # only 2 quarters available, need 4
    assert result.pe_ratio is None


async def test_pe_ratio_computed_when_ttm_eps_and_price_available(sample_stock):
    quarters = [
        _quarter(date(2024, 12, 31), "1000", "100", "5.0"),
        _quarter(date(2024, 9, 30), "900", "90", "4.5"),
        _quarter(date(2024, 6, 30), "850", "80", "4.0"),
        _quarter(date(2024, 3, 31), "800", "70", "3.5"),
    ]
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
                close=Decimal("170.00"),  # TTM EPS = 17.0 -> PE = 170/17 = 10.0
                volume=1000,
            )
        ]
    )

    service = FundamentalsService(
        FakeStockRepository([sample_stock]), financial_repo, price_repo, FakeCorporateActionRepository()
    )

    result = await service.get_fundamentals("RELIANCE")

    assert result.ttm_eps == Decimal("17.0000")
    assert result.pe_ratio == Decimal("10.0")


async def test_dividend_yield_computed_from_trailing_actions(sample_stock):
    quarters = [_quarter(date(2024, 12, 31), "1000", "100", "5.0")]
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
                close=Decimal("1000.00"),
                volume=1000,
            )
        ]
    )

    corp_action_repo = FakeCorporateActionRepository(
        {
            "RELIANCE": [
                CorporateAction(
                    symbol="RELIANCE",
                    purpose="Dividend - Rs 20 Per Share",
                    face_value=None,
                    ex_date=date.today().replace(year=date.today().year),
                    record_date=None,
                    book_closure_start=None,
                    book_closure_end=None,
                )
            ]
        }
    )

    service = FundamentalsService(FakeStockRepository([sample_stock]), financial_repo, price_repo, corp_action_repo)

    result = await service.get_fundamentals("RELIANCE")

    # dividend_yield = 20/1000*100 = 2.0%
    assert result.dividend_yield == Decimal("2.0")


async def test_fundamentals_never_populates_unavailable_metrics(sample_stock):
    quarters = [_quarter(date(2024, 12, 31), "1000", "100", "5.0")]
    financial_repo = FakeFinancialResultRepository({"RELIANCE": quarters})
    service = FundamentalsService(
        FakeStockRepository([sample_stock]),
        financial_repo,
        FakeHistoricalPriceRepository(),
        FakeCorporateActionRepository(),
    )

    result = await service.get_fundamentals("RELIANCE")

    assert result.book_value is None
    assert result.roe is None
    assert result.roce is None
    assert result.debt_to_equity is None
