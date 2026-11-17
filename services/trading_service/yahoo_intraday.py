"""Free intraday candles from Yahoo Finance's chart API (fallback when the
Kite Connect app has no historical-data permission).

Limits imposed by Yahoo (not by us):
  1m candles - only the last ~30 days, fetched in <=7-day windows
  5m candles - only the last ~60 days
Indices and stocks only - Yahoo has no NSE futures/options contracts, which
is fine for the F&O backtest (futures are proxied by the index, options are
modelled from the index + India VIX).
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from libs.trading_calendar.calendar import IST
from services.trading_service.fno_backtest import Candle

YAHOO_SYMBOLS = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
    "MIDCPNIFTY": "NIFTY_MID_SELECT.NS",
    "SENSEX": "^BSESN",
    "INDIA VIX": "^INDIAVIX",
}
LOOKBACK_DAYS = {"1m": 29, "5m": 59}
WINDOW_DAYS = {"1m": 7, "5m": 59}
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval={iv}&period1={p1}&period2={p2}"


def yahoo_symbol(name: str) -> str:
    return YAHOO_SYMBOLS.get(name.upper(), f"{name.upper()}.NS")


def parse_chart(payload: dict) -> list[Candle]:
    """Yahoo chart JSON -> candles (rows with missing prices skipped)."""
    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        err = (payload.get("chart") or {}).get("error")
        raise ValueError(f"Yahoo returned no data: {err}")
    r = result[0]
    stamps = r.get("timestamp") or []
    q = (r.get("indicators") or {}).get("quote", [{}])[0]
    out: list[Candle] = []
    for i, ts in enumerate(stamps):
        o, h, lo, c = (q.get(k, [None] * len(stamps))[i] for k in ("open", "high", "low", "close"))
        if None in (o, h, lo, c):
            continue
        out.append(
            Candle(
                datetime.fromtimestamp(ts, tz=UTC).astimezone(IST),
                Decimal(str(round(o, 2))),
                Decimal(str(round(h, 2))),
                Decimal(str(round(lo, 2))),
                Decimal(str(round(c, 2))),
            )
        )
    return out


def available_start(interval: str, today: date | None = None) -> date:
    today = today or date.today()
    return today - timedelta(days=LOOKBACK_DAYS[interval])


def fetch_candles(name: str, interval: str, start: date, end: date, *, opener=urllib.request.urlopen) -> list[Candle]:
    """Candles for [start, end] (clamped to what Yahoo serves)."""
    if interval not in LOOKBACK_DAYS:
        raise ValueError("interval must be 1m or 5m")
    start = max(start, available_start(interval))
    sym = yahoo_symbol(name)
    out: list[Candle] = []
    cursor = start
    while cursor <= end:
        stop = min(end, cursor + timedelta(days=WINDOW_DAYS[interval] - 1))
        p1 = int(IST.localize(datetime.combine(cursor, datetime.min.time())).timestamp())
        p2 = int(IST.localize(datetime.combine(stop + timedelta(days=1), datetime.min.time())).timestamp())
        req = urllib.request.Request(_URL.format(sym=sym, iv=interval, p1=p1, p2=p2), headers={"User-Agent": _UA})
        with opener(req, timeout=30) as resp:
            out += parse_chart(json.load(resp))
        cursor = stop + timedelta(days=1)
        time.sleep(0.5)
    seen: set[datetime] = set()
    return [c for c in sorted(out, key=lambda c: c.ts) if not (c.ts in seen or seen.add(c.ts))]
