"""Paper replay of the /nse-* skills' intraday opening-range breakout (ORB) on real 5-minute bars.

Rules copied from the skills' defaults:
  /nse-entry : opening range = first 15 minutes (the 9:15, 9:20, 9:25 five-minute bars); entry on the
               first COMPLETED bar that closes above the range high, at that bar's close. Long only.
  /nse-exit  : stop = opening-range midpoint (rounded down to NSE's tick), target = 2R, flat by 15:15.
Inside a bar, if both the stop and the target are touched, the stop is assumed first (conservative);
a bar that opens through the stop fills at its open. Charges: Zerodha MIS (intraday) schedule.
Stdlib only - runs in the Lambda and in research scripts.
"""

from __future__ import annotations

import json
import math
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"


def tick_size(price: float) -> float:
    """NSE tick by price band (0.01 under Rs 250, 0.05 to 1,000, 0.10 to 5,000, 0.50 to 10,000, 1 to 20,000, then 5)."""
    for limit, tick in ((250, 0.01), (1000, 0.05), (5000, 0.10), (10000, 0.50), (20000, 1.0)):
        if price < limit:
            return tick
    return 5.0


def fetch_5m(symbol: str, rng: str = "60d") -> dict[date, list[dict]]:
    """Yahoo 5-minute bars for SYMBOL.NS grouped by IST trading day: [{t, o, h, l, c, v}]."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.replace('&', '%26')}.NS?interval=5m&range={rng}"
    for attempt in range(3):
        try:
            doc = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=20))
            break
        except Exception:
            if attempt == 2:
                return {}
            time.sleep(2 * (attempt + 1))
    res = (doc.get("chart") or {}).get("result") or []
    if not res or "timestamp" not in res[0]:
        return {}
    q = res[0]["indicators"]["quote"][0]
    days: dict[date, list[dict]] = {}
    for i, ts in enumerate(res[0]["timestamp"]):
        o, h, low, c = (q[k][i] for k in ("open", "high", "low", "close"))
        if None in (o, h, low, c):
            continue
        t = datetime.fromtimestamp(ts, IST)
        days.setdefault(t.date(), []).append({"t": t, "o": o, "h": h, "l": low, "c": c, "v": q["volume"][i] or 0})
    return days


def mis_charges(buy_value: float, sell_value: float) -> float:
    """Zerodha intraday equity: brokerage min(Rs 20, 0.03%) per order, STT 0.025% on the sell, NSE txn
    0.00297%, SEBI Rs 10/crore, stamp 0.003% on the buy, GST 18% on brokerage + txn + SEBI."""
    brokerage = min(20.0, buy_value * 0.0003) + min(20.0, sell_value * 0.0003)
    txn = (buy_value + sell_value) * 0.0000297
    sebi = (buy_value + sell_value) * 0.000001
    return brokerage + sell_value * 0.00025 + txn + sebi + buy_value * 0.00003 + 0.18 * (brokerage + txn + sebi)


def simulate_day(bars: list[dict], capital: float = 10000.0) -> dict | None:
    """One ORB trade for one stock-day (or None if no breakout). P&L uses `capital` rupees per trade."""
    bars = sorted(bars, key=lambda b: b["t"])
    orb = [b for b in bars if (b["t"].hour, b["t"].minute) in ((9, 15), (9, 20), (9, 25))]
    if len(orb) < 3:
        return None
    hi, lo = max(b["h"] for b in orb), min(b["l"] for b in orb)
    after = [b for b in bars if (b["t"].hour, b["t"].minute) >= (9, 30)]
    entry_i = next((i for i, b in enumerate(after) if b["c"] > hi and (b["t"].hour, b["t"].minute) < (15, 10)), None)
    if entry_i is None:
        return None
    eb = after[entry_i]
    entry = eb["c"]
    tick = tick_size(entry)
    stop = math.floor(((hi + lo) / 2) / tick) * tick
    if stop >= entry:
        return None
    risk = entry - stop
    target = round(round((entry + 2 * risk) / tick) * tick, 2)
    exit_px, why = None, "flat 15:15"
    for b in after[entry_i + 1 :]:
        if (b["t"].hour, b["t"].minute) >= (15, 15):
            exit_px = b["o"]
            break
        if b["l"] <= stop:
            exit_px, why = min(b["o"], stop), "stop"
            break
        if b["h"] >= target:
            exit_px, why = max(b["o"], target), "target"
            break
    if exit_px is None:
        exit_px = after[-1]["c"]
    qty = int(capital // entry)
    if qty <= 0:
        return None
    gross = qty * (exit_px - entry)
    ch = mis_charges(qty * entry, qty * exit_px)
    return {
        "entry_time": eb["t"].strftime("%H:%M"),
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target": target,
        "exit": round(exit_px, 2),
        "why": why,
        "qty": qty,
        "r": round((exit_px - entry) / risk, 2),
        "gross": round(gross, 2),
        "charges": round(ch, 2),
        "net": round(gross - ch, 2),
    }
