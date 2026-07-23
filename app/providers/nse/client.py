"""Resilient HTTP client for NSE's public (undocumented) JSON API and CSV archives.

Handles what makes NSE painful to call directly:
- it requires browser-like session cookies, obtained by hitting a normal page first
- it aggressively rate-limits/blocks bot-like traffic (401/403/429)
- it has no SLA - timeouts and 5xx are common

This class only implements the two operations Phase 1 needs (`get_quote_json`,
`get_equity_master_csv`). It has no knowledge of `Stock`/`Quote` domain
entities - that translation happens in `nse_provider.py`.
"""

import asyncio
import csv
import io
import logging
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import date

import httpx
from aiolimiter import AsyncLimiter
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import Settings
from app.providers.nse.circuit_breaker import AsyncCircuitBreaker, CircuitOpenError
from app.providers.nse.endpoints import BHAVCOPY_ZIP_URL_TEMPLATE, EQUITY_LIST_CSV_URL, IPO_WARMUP_URL
from app.providers.nse.exceptions import (
    NseAuthExpiredError,
    NseNotFoundError,
    NseRateLimitedError,
    NseServerError,
    NseTimeoutError,
)
from app.providers.nse.xbrl import XbrlDocument, parse_xbrl_document

logger = logging.getLogger(__name__)

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]


def _wait_for_retry_after(retry_state):
    """Respect a 429's Retry-After header when present; otherwise exponential backoff."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, NseRateLimitedError) and exc.retry_after:
        return exc.retry_after
    return wait_exponential(multiplier=1, min=1, max=10)(retry_state)


class NseClient:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._http = httpx.AsyncClient(timeout=settings.NSE_REQUEST_TIMEOUT_SECONDS)
        self._session_ready = False
        self._cookies_fetched_at: float = 0.0
        self._cookie_lock = asyncio.Lock()
        self._ua_index = 0
        self._limiter = AsyncLimiter(max(settings.NSE_RATE_LIMIT_PER_SECOND, 0.1), 1)
        self._breaker = AsyncCircuitBreaker(
            fail_max=settings.NSE_CIRCUIT_FAIL_MAX,
            reset_timeout_seconds=settings.NSE_CIRCUIT_RESET_TIMEOUT_SECONDS,
            exclude=(NseNotFoundError,),
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    def _next_user_agent(self) -> str:
        ua = _USER_AGENTS[self._ua_index % len(_USER_AGENTS)]
        self._ua_index += 1
        return ua

    def _build_headers(self) -> dict[str, str]:
        return {
            "User-Agent": self._next_user_agent(),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": self._settings.NSE_BASE_URL + "/",
        }

    async def _bootstrap_session(self) -> None:
        # httpx.AsyncClient persists Set-Cookie responses in its own jar and
        # replays them on subsequent requests through the same client - no
        # need to capture/pass cookies manually.
        try:
            resp = await self._http.get(self._settings.NSE_BASE_URL + "/", headers=self._build_headers())
        except httpx.TimeoutException as exc:
            raise NseTimeoutError("/ (session bootstrap)") from exc
        except httpx.HTTPError as exc:
            raise NseServerError("/ (session bootstrap)", 0) from exc

        if resp.status_code >= 400:
            # NSE (or Akamai in front of it) can reject the bootstrap request
            # itself (401/403/429/5xx) - this must surface as a typed,
            # retryable error, not an unhandled httpx.HTTPStatusError.
            raise NseServerError("/ (session bootstrap)", resp.status_code)

        self._session_ready = True
        self._cookies_fetched_at = time.monotonic()
        logger.debug("NSE session bootstrapped (%d cookies)", len(self._http.cookies))

    async def _ensure_session(self, force: bool = False) -> None:
        async with self._cookie_lock:
            expired = (time.monotonic() - self._cookies_fetched_at) > self._settings.NSE_COOKIE_TTL_SECONDS
            if force or not self._session_ready or expired:
                await self._bootstrap_session()

    async def _get_json_once(self, path: str, params: dict | None = None) -> dict | list:
        await self._ensure_session()
        try:
            async with self._limiter:
                resp = await self._http.get(
                    self._settings.NSE_BASE_URL + path,
                    params=params,
                    headers=self._build_headers(),
                )
        except httpx.TimeoutException as exc:
            raise NseTimeoutError(path) from exc
        except httpx.HTTPError as exc:
            raise NseServerError(path, 0) from exc

        if resp.status_code == 404:
            raise NseNotFoundError(path)
        if resp.status_code in (401, 403):
            await self._ensure_session(force=True)
            raise NseAuthExpiredError(path, resp.status_code)
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            raise NseRateLimitedError(path, float(retry_after) if retry_after else None)
        if resp.status_code >= 500:
            raise NseServerError(path, resp.status_code)

        resp.raise_for_status()
        return resp.json()

    async def get_json(self, path: str, params: dict | None = None) -> dict | list:
        """Fetch JSON from a cookie-gated NSE API endpoint, with retry + circuit breaking."""

        retrying = retry(
            retry=retry_if_exception_type((NseTimeoutError, NseServerError, NseAuthExpiredError, NseRateLimitedError)),
            stop=stop_after_attempt(self._settings.NSE_MAX_RETRIES),
            wait=_wait_for_retry_after,
            reraise=True,
        )(self._get_json_once)

        try:
            return await self._breaker.call(retrying, path, params)
        except CircuitOpenError as exc:
            # CircuitOpenError is a plain Exception, not an NseError subclass
            # (the breaker is generic, has no NSE-specific knowledge) -
            # translate it here so every caller's `except NseError` (and,
            # downstream, nse_provider.py's translation to
            # ProviderUnavailableError) catches this the same as any other
            # NSE failure, instead of it leaking as an unhandled 500.
            raise NseServerError(path, 0) from exc

    async def _fetch_ipo_json_once(self, path: str, params: dict) -> dict | list:
        # IPO endpoints need a session bootstrapped against the IPO page
        # itself, not the homepage - confirmed live (2026-07-22): the shared
        # homepage bootstrap was blocked at the time, but warming up against
        # this specific page succeeded. Uses its own short-lived client
        # rather than `self._http`'s shared session/cookie state, since this
        # runs once a day (a sync job), not a hot request path.
        async with httpx.AsyncClient(timeout=self._settings.NSE_REQUEST_TIMEOUT_SECONDS) as client:
            headers = self._build_headers()
            try:
                warmup = await client.get(IPO_WARMUP_URL, headers=headers)
            except httpx.TimeoutException as exc:
                raise NseTimeoutError(f"{path} (ipo warmup)") from exc
            except httpx.HTTPError as exc:
                raise NseServerError(f"{path} (ipo warmup)", 0) from exc
            if warmup.status_code >= 400:
                raise NseServerError(f"{path} (ipo warmup)", warmup.status_code)

            try:
                resp = await client.get(self._settings.NSE_BASE_URL + path, params=params, headers=headers)
            except httpx.TimeoutException as exc:
                raise NseTimeoutError(path) from exc
            except httpx.HTTPError as exc:
                raise NseServerError(path, 0) from exc

        if resp.status_code == 404:
            raise NseNotFoundError(path)
        if resp.status_code in (401, 403):
            raise NseAuthExpiredError(path, resp.status_code)
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            raise NseRateLimitedError(path, float(retry_after) if retry_after else None)
        if resp.status_code >= 500:
            raise NseServerError(path, resp.status_code)

        resp.raise_for_status()
        return resp.json()

    async def get_ipo_json(self, path: str, params: dict | None = None) -> dict | list:
        """Fetch JSON from an IPO endpoint, with its own retry (no shared
        circuit breaker - this is a rare, once-daily call, isolating it means
        a run of IPO failures can't trip the breaker for every other NSE call).
        """
        retrying = retry(
            retry=retry_if_exception_type((NseTimeoutError, NseServerError, NseAuthExpiredError, NseRateLimitedError)),
            stop=stop_after_attempt(self._settings.NSE_MAX_RETRIES),
            wait=_wait_for_retry_after,
            reraise=True,
        )(self._fetch_ipo_json_once)
        return await retrying(path, params or {})

    async def _fetch_static_file_once(self, url: str, label: str) -> bytes:
        try:
            resp = await self._http.get(url, headers=self._build_headers())
        except httpx.TimeoutException as exc:
            raise NseTimeoutError(label) from exc
        except httpx.HTTPError as exc:
            raise NseServerError(label, 0) from exc

        if resp.status_code == 404:
            raise NseNotFoundError(label)
        if resp.status_code >= 400:
            raise NseServerError(label, resp.status_code)

        return resp.content

    async def _fetch_static_file(self, url: str, label: str) -> bytes:
        """Fetch a file from a no-cookie static NSE archive host, with retry +
        circuit breaking. Shared by the equity master CSV and Bhavcopy ZIP,
        both hosted on nsearchives.nseindia.com with no session required.
        """
        retrying = retry(
            retry=retry_if_exception_type((NseTimeoutError, NseServerError)),
            stop=stop_after_attempt(self._settings.NSE_MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )(self._fetch_static_file_once)

        try:
            return await self._breaker.call(retrying, url, label)
        except CircuitOpenError as exc:
            raise NseServerError(label, 0) from exc

    @staticmethod
    def _parse_csv_rows(text: str) -> list[dict[str, str]]:
        reader = csv.DictReader(io.StringIO(text))
        return [{k.strip(): (v.strip() if v else v) for k, v in row.items()} for row in reader]

    async def get_equity_master_csv(self) -> list[dict[str, str]]:
        """Fetch and parse the NSE bulk equity list CSV. No session/cookies needed - static archive."""
        content = await self._fetch_static_file(EQUITY_LIST_CSV_URL, "EQUITY_L.csv")
        return self._parse_csv_rows(content.decode("utf-8"))

    async def get_bhavcopy_rows(self, trade_date: date) -> list[dict[str, str]]:
        """Fetch one day's full Cash Market Bhavcopy (ZIP containing one CSV) and
        parse it. No session/cookies needed - static archive. The CSV covers
        every CM instrument (equities, SGBs, T-Bills, ...); filtering to
        equities is the caller's (`nse_provider.py`) job.
        """
        label = f"Bhavcopy {trade_date.isoformat()}"
        url = BHAVCOPY_ZIP_URL_TEMPLATE.format(date=trade_date.strftime("%Y%m%d"))
        content = await self._fetch_static_file(url, label)

        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
                if not csv_names:
                    raise NseServerError(label, 0)
                with zf.open(csv_names[0]) as f:
                    text = f.read().decode("utf-8")
        except zipfile.BadZipFile as exc:
            raise NseServerError(label, 0) from exc

        return self._parse_csv_rows(text)

    async def get_xbrl_document(self, url: str) -> XbrlDocument:
        """Fetch and parse a financial-results XBRL filing. No session/cookies
        needed - same static archive host as the equity master CSV and Bhavcopy.
        """
        content = await self._fetch_static_file(url, "XBRL filing")
        try:
            return parse_xbrl_document(content)
        except ET.ParseError as exc:
            raise NseServerError(url, 0) from exc
