import logging
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation

from app.core.exceptions import ProviderUnavailableError
from app.domain.entities import (
    BhavcopyRecord,
    CorporateAction,
    FinancialResultFiling,
    FinancialResultRecord,
    IndexQuote,
    IpoFiling,
    MarketStatus,
    Quote,
    StockMasterRecord,
)
from app.domain.ports import StockDataProviderPort
from app.providers.nse.client import NseClient
from app.providers.nse.endpoints import (
    ALL_INDICES_PATH,
    CORPORATE_ACTIONS_PATH,
    FINANCIAL_RESULTS_INDEX_PATH,
    MARKET_STATUS_PATH,
    PAST_IPO_PATH,
    QUOTE_EQUITY_PATH,
    UPCOMING_IPO_PATH,
)
from app.providers.nse.exceptions import NseError, NseNotFoundError
from app.providers.nse.xbrl import select_quarterly_context

# Only the recent, SEBI-standardized taxonomy is parsed - see Phase 3 plan for
# why older filings (different taxonomy, different tag names) are out of scope.
_SUPPORTED_TAXONOMY = "Ind-AS New"

_CONCEPT_REVENUE = "RevenueFromOperations"
_CONCEPT_PROFIT = "ProfitLossForPeriod"
_CONCEPT_EPS_BASIC = "BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations"
_CONCEPT_EPS_DILUTED = "DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations"
_TARGET_CONCEPTS = {_CONCEPT_REVENUE, _CONCEPT_PROFIT, _CONCEPT_EPS_BASIC, _CONCEPT_EPS_DILUTED}

logger = logging.getLogger(__name__)


def _to_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _to_date(value: str | None):
    if not value:
        return None
    for fmt in ("%d-%b-%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


class NseStockDataProvider(StockDataProviderPort):
    """Implements StockDataProviderPort against NSE's public site.

    Translates NSE-specific failures (`NseError` subclasses) into the
    provider-agnostic `ProviderUnavailableError` so `StockService` never has
    to know this is NSE it's talking to.
    """

    def __init__(self, client: NseClient):
        self._client = client

    async def get_quote(self, symbol: str) -> Quote:
        try:
            data = await self._client.get_json(QUOTE_EQUITY_PATH, params={"symbol": symbol})
        except NseError as exc:
            raise ProviderUnavailableError("NSE", str(exc)) from exc

        price_info = data.get("priceInfo") or {}
        last_price = _to_decimal(price_info.get("lastPrice"))
        if last_price is None:
            raise ProviderUnavailableError("NSE", f"quote response for {symbol} missing lastPrice")

        intraday = price_info.get("intraDayHighLow") or {}
        volume = (
            (data.get("marketDeptOrderBook") or {}).get("tradeInfo", {}).get("totalTradedVolume")
            or (data.get("securityWiseDP") or {}).get("quantityTraded")
            or 0
        )

        return Quote(
            symbol=symbol,
            last_price=last_price,
            change=_to_decimal(price_info.get("change")) or Decimal("0"),
            change_percent=_to_decimal(price_info.get("pChange")) or Decimal("0"),
            open=_to_decimal(price_info.get("open")) or last_price,
            high=_to_decimal(intraday.get("max")) or last_price,
            low=_to_decimal(intraday.get("min")) or last_price,
            previous_close=_to_decimal(price_info.get("previousClose")) or last_price,
            volume=int(volume) if volume else 0,
            as_of=datetime.now(UTC),
        )

    async def fetch_equity_universe(self) -> list[StockMasterRecord]:
        try:
            rows = await self._client.get_equity_master_csv()
        except NseError as exc:
            raise ProviderUnavailableError("NSE", str(exc)) from exc

        records: list[StockMasterRecord] = []
        for row in rows:
            symbol = row.get("SYMBOL")
            if not symbol:
                continue
            records.append(
                StockMasterRecord(
                    symbol=symbol.strip().upper(),
                    isin=(row.get("ISIN NUMBER") or "").strip() or None,
                    name=(row.get("NAME OF COMPANY") or symbol).strip(),
                    series=(row.get("SERIES") or "").strip() or None,
                    listing_date=_to_date((row.get("DATE OF LISTING") or "").strip()),
                    face_value=_to_decimal((row.get("FACE VALUE") or "").strip()),
                )
            )

        if not records:
            raise ProviderUnavailableError("NSE", "equity master CSV parsed to zero rows")

        return records

    async def fetch_daily_bars(self, trade_date: date) -> list[BhavcopyRecord]:
        try:
            rows = await self._client.get_bhavcopy_rows(trade_date)
        except NseNotFoundError:
            # No Bhavcopy file for this date - normal for weekends/holidays, not a failure.
            return []
        except NseError as exc:
            raise ProviderUnavailableError("NSE", str(exc)) from exc

        # The Bhavcopy has one row per (symbol, series) - a company can have
        # several simultaneously active series (EQ, BE, BZ, ...), but `stocks`
        # has one row per symbol, so multiple Bhavcopy rows for the same
        # symbol would otherwise collide on (stock_id, trade_date) in the same
        # upsert batch (Postgres rejects a single INSERT..ON CONFLICT touching
        # the same conflict target twice). Keep at most one row per symbol,
        # preferring the "EQ" series when present.
        best_by_symbol: dict[str, tuple[str, BhavcopyRecord]] = {}
        for row in rows:
            symbol = (row.get("TckrSymb") or "").strip().upper()
            if not symbol:
                continue
            open_ = _to_decimal(row.get("OpnPric"))
            high = _to_decimal(row.get("HghPric"))
            low = _to_decimal(row.get("LwPric"))
            close = _to_decimal(row.get("ClsPric"))
            volume = _to_decimal(row.get("TtlTradgVol"))
            # The Bhavcopy covers every Cash Market instrument (equities, SGBs,
            # T-Bills, ...); rows without clean numeric OHLCV aren't a plain
            # equity trade and are skipped here. Whether the symbol is actually
            # one of ours is decided later by the repository's join against
            # `stocks`, not by guessing at NSE's series/instrument-type codes.
            if None in (open_, high, low, close, volume):
                continue

            series = (row.get("SctySrs") or "").strip().upper()
            record = BhavcopyRecord(
                symbol=symbol, trade_date=trade_date, open=open_, high=high, low=low, close=close, volume=int(volume)
            )

            existing = best_by_symbol.get(symbol)
            if existing is None or (series == "EQ" and existing[0] != "EQ"):
                best_by_symbol[symbol] = (series, record)

        return [record for _series, record in best_by_symbol.values()]

    async def fetch_corporate_actions(self, from_date: date, to_date: date) -> list[CorporateAction]:
        params = {
            "index": "equities",
            "from_date": from_date.strftime("%d-%m-%Y"),
            "to_date": to_date.strftime("%d-%m-%Y"),
        }
        try:
            data = await self._client.get_json(CORPORATE_ACTIONS_PATH, params=params)
        except NseError as exc:
            raise ProviderUnavailableError("NSE", str(exc)) from exc

        rows = data if isinstance(data, list) else (data.get("data") if isinstance(data, dict) else None)
        if rows is None:
            raise ProviderUnavailableError("NSE", "corporate-actions response had an unexpected shape")

        records: list[CorporateAction] = []
        for row in rows:
            symbol = (row.get("symbol") or "").strip().upper()
            purpose = (row.get("subject") or row.get("purpose") or "").strip()
            if not symbol or not purpose:
                continue
            records.append(
                CorporateAction(
                    symbol=symbol,
                    purpose=purpose,
                    face_value=_to_decimal(row.get("faceVal")),
                    ex_date=_to_date(row.get("exDate")),
                    record_date=_to_date(row.get("recDate")),
                    book_closure_start=_to_date(row.get("bcStartDate")),
                    book_closure_end=_to_date(row.get("bcEndDate")),
                )
            )
        return records

    async def fetch_financial_results_index(
        self, symbol: str, from_date: date, to_date: date
    ) -> list[FinancialResultFiling]:
        params = {
            "index": "equities",
            "symbol": symbol,
            "period": "Quarterly",
            "from_date": from_date.strftime("%d-%m-%Y"),
            "to_date": to_date.strftime("%d-%m-%Y"),
        }
        try:
            data = await self._client.get_json(FINANCIAL_RESULTS_INDEX_PATH, params=params)
        except NseError as exc:
            raise ProviderUnavailableError("NSE", str(exc)) from exc

        rows = data if isinstance(data, list) else (data.get("data") if isinstance(data, dict) else None)
        if rows is None:
            raise ProviderUnavailableError("NSE", "financial-results index response had an unexpected shape")

        filings: list[FinancialResultFiling] = []
        for row in rows:
            if row.get("indAs") != _SUPPORTED_TAXONOMY:
                continue
            xbrl_url = (row.get("xbrl") or "").strip()
            if not xbrl_url or xbrl_url.endswith("/-"):
                continue  # no usable XBRL attachment for this filing

            period_start = _to_date(row.get("fromDate"))
            period_end = _to_date(row.get("toDate"))
            if period_start is None or period_end is None:
                continue

            filings.append(
                FinancialResultFiling(
                    symbol=(row.get("symbol") or symbol).strip().upper(),
                    period_start=period_start,
                    period_end=period_end,
                    consolidated=(row.get("consolidated") == "Consolidated"),
                    xbrl_url=xbrl_url,
                )
            )
        return filings

    async def fetch_financial_result_detail(self, filing: FinancialResultFiling) -> FinancialResultRecord | None:
        try:
            doc = await self._client.get_xbrl_document(filing.xbrl_url)
        except NseError as exc:
            raise ProviderUnavailableError("NSE", str(exc)) from exc

        context_id = select_quarterly_context(doc.contexts, filing.period_end)
        if context_id is None:
            logger.info("No matching quarterly context found for %s filing %s", filing.symbol, filing.xbrl_url)
            return None

        values: dict[str, Decimal] = {}
        for fact in doc.facts:
            if fact.context_id != context_id or fact.concept not in _TARGET_CONCEPTS:
                continue
            value = _to_decimal(fact.value)
            if value is not None:
                values[fact.concept] = value

        revenue = values.get(_CONCEPT_REVENUE)
        profit = values.get(_CONCEPT_PROFIT)
        eps_basic = values.get(_CONCEPT_EPS_BASIC)
        eps_diluted = values.get(_CONCEPT_EPS_DILUTED)

        if revenue is None and profit is None and eps_basic is None:
            logger.info("No usable facts parsed for %s filing %s", filing.symbol, filing.xbrl_url)
            return None

        return FinancialResultRecord(
            symbol=filing.symbol,
            period_start=filing.period_start,
            period_end=filing.period_end,
            consolidated=filing.consolidated,
            revenue=revenue,
            profit=profit,
            eps_basic=eps_basic,
            eps_diluted=eps_diluted,
        )

    async def fetch_market_status(self) -> list[MarketStatus]:
        try:
            data = await self._client.get_json(MARKET_STATUS_PATH)
        except NseError as exc:
            raise ProviderUnavailableError("NSE", str(exc)) from exc

        rows = data.get("marketState") if isinstance(data, dict) else None
        if rows is None:
            raise ProviderUnavailableError("NSE", "market-status response had an unexpected shape")

        statuses: list[MarketStatus] = []
        for row in rows:
            market = (row.get("market") or "").strip()
            status = (row.get("marketStatus") or "").strip()
            if not market or not status:
                continue
            statuses.append(MarketStatus(market=market, status=status, as_of=(row.get("tradeDate") or "").strip()))
        return statuses

    async def fetch_indices(self) -> list[IndexQuote]:
        try:
            data = await self._client.get_json(ALL_INDICES_PATH)
        except NseError as exc:
            raise ProviderUnavailableError("NSE", str(exc)) from exc

        rows = data.get("data") if isinstance(data, dict) else None
        if rows is None:
            raise ProviderUnavailableError("NSE", "all-indices response had an unexpected shape")

        indices: list[IndexQuote] = []
        for row in rows:
            name = (row.get("indexSymbol") or row.get("index") or "").strip()
            last_price = _to_decimal(row.get("last"))
            if not name or last_price is None:
                continue
            indices.append(
                IndexQuote(
                    index_name=name,
                    last_price=last_price,
                    change=_to_decimal(row.get("variation")) or Decimal("0"),
                    change_percent=_to_decimal(row.get("percentChange")) or Decimal("0"),
                )
            )
        return indices

    async def fetch_ipo_filings(self) -> list[IpoFiling]:
        filings: list[IpoFiling] = []
        failures = 0

        try:
            data = await self._client.get_ipo_json(UPCOMING_IPO_PATH, params={"category": "ipo"})
            rows = data if isinstance(data, list) else []
            for row in rows:
                symbol = (row.get("symbol") or "").strip().upper()
                if not symbol:
                    continue
                filings.append(
                    IpoFiling(
                        symbol=symbol,
                        company_name=(row.get("companyName") or symbol).strip(),
                        status=(row.get("status") or "ACTIVE").strip().upper(),
                        price_range=(row.get("issuePrice") or "").strip() or None,
                        issue_size=(row.get("issueSize") or "").strip() or None,
                        issue_start_date=_to_date(row.get("issueStartDate")),
                        issue_end_date=_to_date(row.get("issueEndDate")),
                        listing_date=None,
                        series=(row.get("series") or "").strip() or None,
                    )
                )
        except NseError as exc:
            failures += 1
            logger.warning("upcoming IPO fetch failed: %s", exc)

        try:
            data = await self._client.get_ipo_json(PAST_IPO_PATH, params={"index": "equities"})
            rows = data if isinstance(data, list) else []
            for row in rows:
                symbol = (row.get("symbol") or "").strip().upper()
                if not symbol:
                    continue
                filings.append(
                    IpoFiling(
                        symbol=symbol,
                        company_name=(row.get("company") or symbol).strip(),
                        status="LISTED",
                        price_range=(row.get("priceRange") or "").strip() or None,
                        issue_size=None,
                        issue_start_date=_to_date(row.get("ipoStartDate")),
                        issue_end_date=_to_date(row.get("ipoEndDate")),
                        listing_date=_to_date(row.get("listingDate")),
                        series=(row.get("securityType") or "").strip() or None,
                    )
                )
        except NseError as exc:
            failures += 1
            logger.warning("past IPO fetch failed: %s", exc)

        if failures == 2:
            raise ProviderUnavailableError("NSE", "both IPO endpoints unreachable")
        return filings
