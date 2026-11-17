"""Yahoo chart parsing + windowed fetching (fake opener - no network)."""

import io
import json
from datetime import date, datetime
from decimal import Decimal

import pytest

from libs.trading_calendar.calendar import IST
from services.trading_service import yahoo_intraday as y


def _payload(stamps, closes):
    return {
        "chart": {
            "result": [
                {
                    "timestamp": stamps,
                    "indicators": {
                        "quote": [
                            {
                                "open": closes,
                                "high": closes,
                                "low": closes,
                                "close": closes,
                                "volume": [0] * len(stamps),
                            }
                        ]
                    },
                }
            ]
        }
    }


T0 = int(IST.localize(datetime(2026, 11, 16, 9, 15)).timestamp())


def test_parse_chart_converts_to_ist_and_skips_gaps():
    candles = y.parse_chart(_payload([T0, T0 + 60, T0 + 120], [24000.123, None, 24010.5]))
    assert [c.ts for c in candles] == [
        IST.localize(datetime(2026, 11, 16, 9, 15)),
        IST.localize(datetime(2026, 11, 16, 9, 17)),
    ]
    assert candles[0].close == Decimal("24000.12")
    with pytest.raises(ValueError):
        y.parse_chart({"chart": {"result": None, "error": {"code": "Not Found"}}})


def test_symbols():
    assert y.yahoo_symbol("NIFTY") == "^NSEI"
    assert y.yahoo_symbol("india vix") == "^INDIAVIX"
    assert y.yahoo_symbol("RELIANCE") == "RELIANCE.NS"


def test_fetch_uses_7_day_windows_for_1m_and_dedupes(monkeypatch):
    monkeypatch.setattr(y.time, "sleep", lambda s: None)
    urls = []

    def opener(req, timeout):
        urls.append(req.full_url)
        return io.BytesIO(json.dumps(_payload([T0, T0 + 60], [1.0, 2.0])).encode())

    today = date.today()
    out = y.fetch_candles("NIFTY", "1m", date(2000, 1, 1), today, opener=opener)
    assert len(urls) == 5  # ~30 days clamped, in 7-day windows
    assert all("interval=1m" in u and "NSEI" in u for u in urls)
    assert len(out) == 2  # identical candles from each window de-duplicated
