from datetime import date, timedelta
from decimal import Decimal

from app.domain.entities import CorporateAction, OhlcvBar
from app.services.dividend_service import DividendService
from tests.conftest import FakeCorporateActionRepository, FakeHistoricalPriceRepository, FakeStockRepository


def _build_service(sample_stock, ex_date: date, purpose: str, close_price: Decimal):
    stock_repo = FakeStockRepository([sample_stock])
    corp_repo = FakeCorporateActionRepository(
        {
            "RELIANCE": [
                CorporateAction(
                    symbol="RELIANCE",
                    purpose=purpose,
                    face_value=Decimal("10.00"),
                    ex_date=ex_date,
                    record_date=ex_date,
                    book_closure_start=None,
                    book_closure_end=None,
                )
            ]
        }
    )
    price_repo = FakeHistoricalPriceRepository(
        {
            "RELIANCE": [
                OhlcvBar(
                    trade_date=date.today(),
                    open=close_price,
                    high=close_price,
                    low=close_price,
                    close=close_price,
                    volume=1000,
                )
            ]
        }
    )
    return DividendService(corp_repo, price_repo, stock_repo)


async def test_list_dividends_computes_yield_and_hold_recommendation_for_upcoming_high_yield(sample_stock):
    ex_date = date.today() + timedelta(days=10)
    service = _build_service(sample_stock, ex_date, "Dividend - Rs 40 Per Share", Decimal("1000"))

    results = await service.list_dividends(upcoming_only=True, sort="ex_date", limit=50)

    assert len(results) == 1
    entry = results[0]
    assert entry.symbol == "RELIANCE"
    assert entry.name == "Reliance Industries Limited"
    assert entry.dividend_amount == Decimal("40")
    assert entry.dividend_yield == Decimal("4.00")
    assert entry.ex_dividend_date == ex_date
    assert entry.buy_before_date == ex_date - timedelta(days=1)
    assert entry.recommendation == "Hold for dividend"
    assert entry.risk_level == "Low"
    assert entry.confidence == 80


async def test_list_dividends_excludes_non_dividend_corporate_actions(sample_stock):
    ex_date = date.today() + timedelta(days=10)
    service = _build_service(
        sample_stock,
        ex_date,
        "Face Value Split (Sub-Division) - From Rs 5/- Per Share To Rs 2/- Per Share",
        Decimal("1000"),
    )

    results = await service.list_dividends(upcoming_only=True, sort="ex_date", limit=50)

    assert results == []


async def test_list_dividends_upcoming_only_filters_out_past_ex_dates(sample_stock):
    ex_date = date.today() - timedelta(days=30)
    service = _build_service(sample_stock, ex_date, "Dividend - Rs 40 Per Share", Decimal("1000"))

    results = await service.list_dividends(upcoming_only=True, sort="ex_date", limit=50)

    assert results == []

    all_results = await service.list_dividends(upcoming_only=False, sort="ex_date", limit=50)
    assert len(all_results) == 1
    assert all_results[0].recommendation == "Hold"


async def test_list_dividends_sorts_by_yield_descending():
    from app.domain.entities import InstrumentType, Stock

    stock_a = Stock(
        symbol="AAA",
        isin=None,
        name="AAA Ltd",
        series="EQ",
        sector=None,
        industry=None,
        instrument_type=InstrumentType.EQUITY,
        listing_date=None,
        face_value=Decimal("10.00"),
        is_active=True,
    )
    stock_b = Stock(
        symbol="BBB",
        isin=None,
        name="BBB Ltd",
        series="EQ",
        sector=None,
        industry=None,
        instrument_type=InstrumentType.EQUITY,
        listing_date=None,
        face_value=Decimal("10.00"),
        is_active=True,
    )
    ex_date = date.today() + timedelta(days=10)
    stock_repo = FakeStockRepository([stock_a, stock_b])
    corp_repo = FakeCorporateActionRepository(
        {
            "AAA": [
                CorporateAction(
                    symbol="AAA",
                    purpose="Dividend - Rs 5 Per Share",
                    face_value=None,
                    ex_date=ex_date,
                    record_date=None,
                    book_closure_start=None,
                    book_closure_end=None,
                )
            ],
            "BBB": [
                CorporateAction(
                    symbol="BBB",
                    purpose="Dividend - Rs 50 Per Share",
                    face_value=None,
                    ex_date=ex_date,
                    record_date=None,
                    book_closure_start=None,
                    book_closure_end=None,
                )
            ],
        }
    )
    price_repo = FakeHistoricalPriceRepository(
        {
            "AAA": [
                OhlcvBar(
                    trade_date=date.today(),
                    open=Decimal("1000"),
                    high=Decimal("1000"),
                    low=Decimal("1000"),
                    close=Decimal("1000"),
                    volume=1,
                )
            ],
            "BBB": [
                OhlcvBar(
                    trade_date=date.today(),
                    open=Decimal("1000"),
                    high=Decimal("1000"),
                    low=Decimal("1000"),
                    close=Decimal("1000"),
                    volume=1,
                )
            ],
        }
    )
    service = DividendService(corp_repo, price_repo, stock_repo)

    results = await service.list_dividends(upcoming_only=True, sort="yield", limit=50)

    assert [r.symbol for r in results] == ["BBB", "AAA"]
