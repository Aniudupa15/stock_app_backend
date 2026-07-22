import io
import zipfile
from datetime import date
from decimal import Decimal

import httpx
import pytest
import respx

from app.core.config import Settings
from app.domain.entities import BhavcopyRecord
from app.providers.nse.client import NseClient
from app.providers.nse.nse_provider import NseStockDataProvider

# A trimmed real-shaped UDiFF CM Bhavcopy sample: one clean equity row
# (RELIANCE), one row with non-numeric OHLC (as SGB/derivative rows can have
# blank strike/expiry fields but here simulates a row that should be skipped),
# and one row missing a symbol entirely.
_UDIFF_HEADER = (
    "TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,"
    "FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,"
    "LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,"
    "TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4"
)
_UDIFF_ROW_RELIANCE = (
    "17-JUL-2026,17-JUL-2026,CM,NSE,,500325,INE002A01018,RELIANCE,EQ,,,,,"
    "Reliance Industries,2490.00,2510.00,2480.00,2500.00,2500.00,2487.50,,,,,"
    "1000000,2500000000,5000,F1,1,,,,,"
)
_UDIFF_ROW_BAD_OHLC = (
    "17-JUL-2026,17-JUL-2026,CM,NSE,,999999,INE999X99999,SGBJUL29,GB,,,,," "SGB,-,-,-,-,-,-,,,,,0,0,0,F1,1,,,,,"
)
_UDIFF_ROW_RELIANCE_BE_SERIES = (
    "17-JUL-2026,17-JUL-2026,CM,NSE,,500325,INE002A01018,RELIANCE,BE,,,,,"
    "Reliance Industries,2491.00,2511.00,2481.00,2501.00,2501.00,2488.50,,,,,"
    "2000,5000000,10,F1,1,,,,,"
)
_UDIFF_ROW_NO_SYMBOL = (
    "17-JUL-2026,17-JUL-2026,CM,NSE,,000000,,,EQ,,,,,,100.00,101.00,99.00,100.50,"
    "100.50,99.50,,,,,500,50000,10,F1,1,,,,,"
)


def _make_bhavcopy_zip(rows: list[str]) -> bytes:
    csv_text = "\n".join([_UDIFF_HEADER, *rows]) + "\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("BhavCopy_NSE_CM_0_0_0_20260717_F_0000.csv", csv_text)
    return buf.getvalue()


@pytest.fixture
def settings() -> Settings:
    return Settings(DATABASE_URL="postgresql+asyncpg://test:test@localhost/test", NSE_RATE_LIMIT_PER_SECOND=1000)


@respx.mock
async def test_get_bhavcopy_rows_parses_csv_from_zip(settings):
    zip_bytes = _make_bhavcopy_zip([_UDIFF_ROW_RELIANCE])
    respx.get(url__regex=r".*BhavCopy_NSE_CM.*\.csv\.zip").mock(return_value=httpx.Response(200, content=zip_bytes))

    client = NseClient(settings)
    try:
        rows = await client.get_bhavcopy_rows(date(2026, 7, 17))
    finally:
        await client.aclose()

    assert len(rows) == 1
    assert rows[0]["TckrSymb"] == "RELIANCE"
    assert rows[0]["ClsPric"] == "2500.00"


@respx.mock
async def test_fetch_daily_bars_filters_bad_and_missing_rows(settings):
    zip_bytes = _make_bhavcopy_zip([_UDIFF_ROW_RELIANCE, _UDIFF_ROW_BAD_OHLC, _UDIFF_ROW_NO_SYMBOL])
    respx.get(url__regex=r".*BhavCopy_NSE_CM.*\.csv\.zip").mock(return_value=httpx.Response(200, content=zip_bytes))

    client = NseClient(settings)
    try:
        provider = NseStockDataProvider(client)
        records = await provider.fetch_daily_bars(date(2026, 7, 17))
    finally:
        await client.aclose()

    assert len(records) == 1
    assert records[0] == BhavcopyRecord(
        symbol="RELIANCE",
        trade_date=date(2026, 7, 17),
        open=Decimal("2490.00"),
        high=Decimal("2510.00"),
        low=Decimal("2480.00"),
        close=Decimal("2500.00"),
        volume=1000000,
    )


@respx.mock
async def test_fetch_daily_bars_dedupes_multi_series_symbol_preferring_eq(settings):
    """Regression test: NSE's Bhavcopy has one row per (symbol, series) - a
    company can have simultaneously active EQ and BE (or other) series rows
    on the same date. Since `stocks` has one row per symbol, two Bhavcopy rows
    for the same symbol used to reach the repository's upsert batch together
    and crash Postgres with "ON CONFLICT DO UPDATE command cannot affect row
    a second time". Exactly one record per symbol must come out, preferring EQ.
    """
    zip_bytes = _make_bhavcopy_zip([_UDIFF_ROW_RELIANCE_BE_SERIES, _UDIFF_ROW_RELIANCE])
    respx.get(url__regex=r".*BhavCopy_NSE_CM.*\.csv\.zip").mock(return_value=httpx.Response(200, content=zip_bytes))

    client = NseClient(settings)
    try:
        provider = NseStockDataProvider(client)
        records = await provider.fetch_daily_bars(date(2026, 7, 17))
    finally:
        await client.aclose()

    assert len(records) == 1
    assert records[0].symbol == "RELIANCE"
    assert records[0].close == Decimal("2500.00")  # the EQ row's close, not BE's 2501.00


@respx.mock
async def test_fetch_daily_bars_returns_empty_on_holiday_404(settings):
    respx.get(url__regex=r".*BhavCopy_NSE_CM.*\.csv\.zip").mock(return_value=httpx.Response(404))

    client = NseClient(settings)
    try:
        provider = NseStockDataProvider(client)
        records = await provider.fetch_daily_bars(date(2026, 7, 18))  # a Saturday
    finally:
        await client.aclose()

    assert records == []
