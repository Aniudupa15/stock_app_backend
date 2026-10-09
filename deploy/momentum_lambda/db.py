"""Neon Postgres access (pg8000: pure Python, no native wheels) + NSE Bhavcopy sync.

Mirrors the backend exactly:
- ranking == services/trading_service/momentum/ranking.py compute_ranking()
- sync    == NseStockDataProvider.fetch_daily_bars() + bulk_upsert_bars()
"""

import csv
import io
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation

import pg8000.native

BHAVCOPY_URL = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{d}_F_0000.csv.zip"
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"


def connect() -> pg8000.native.Connection:
    u = urllib.parse.urlparse(os.environ["DATABASE_URL"])
    return pg8000.native.Connection(
        user=urllib.parse.unquote(u.username),
        password=urllib.parse.unquote(u.password),
        host=u.hostname,
        port=u.port or 5432,
        database=u.path.lstrip("/"),
        ssl_context=ssl.create_default_context(),
        timeout=60,
    )


def latest_trade_date(con) -> date | None:
    return con.run("select max(trade_date) from historical_prices")[0][0]


def day_counts(con) -> tuple[int, int]:
    rows = con.run(
        "select trade_date, count(*) from historical_prices where trade_date in "
        "(select distinct trade_date from historical_prices order by trade_date desc limit 2) "
        "group by trade_date order by trade_date desc"
    )
    return (rows[0][1] if rows else 0), (rows[1][1] if len(rows) > 1 else 0)


def ranking_candidates(con, lookback: int = 30, universe: int = 300, take: int = 20) -> list[str]:
    """compute_ranking(): liquid top-`universe` by turnover, ranked by `lookback`-day return."""
    latest = latest_trade_date(con)
    if latest is None:
        return []
    cutoff = latest - timedelta(days=max(100, lookback * 3))
    rows = con.run(
        "select s.symbol, hp.close, hp.volume from historical_prices hp join stocks s on s.id = hp.stock_id "
        "where hp.trade_date >= :cutoff and s.is_active = true order by s.symbol, hp.trade_date",
        cutoff=cutoff,
    )
    series: dict[str, tuple[list[float], list[float]]] = {}
    for sym, close, vol in rows:
        cs, ts = series.setdefault(sym, ([], []))
        cs.append(float(close))
        ts.append(float(close) * float(vol))
    ranked = []
    for sym, (cs, ts) in series.items():
        if len(cs) <= lookback or cs[-1 - lookback] <= 0:
            continue
        ranked.append((sym, (cs[-1] / cs[-1 - lookback] - 1) * 100, sum(ts) / len(ts)))
    liquid = sorted(ranked, key=lambda r: r[2], reverse=True)[:universe]
    liquid.sort(key=lambda r: r[1], reverse=True)
    return [r[0] for r in liquid[:take]]


def close_series(con, symbols: list[str], days: int = 70) -> dict[str, list[float]]:
    if not symbols:
        return {}
    latest = latest_trade_date(con)
    rows = con.run(
        "select s.symbol, hp.close from historical_prices hp join stocks s on s.id = hp.stock_id "
        "where hp.trade_date >= :since and s.symbol = any(:syms) order by s.symbol, hp.trade_date",
        since=latest - timedelta(days=days),
        syms=symbols,
    )
    out: dict[str, list[float]] = {}
    for sym, close in rows:
        out.setdefault(sym, []).append(float(close))
    return out


# ------------------------------------------------------------------ Bhavcopy sync


def _dec(v: str | None) -> Decimal | None:
    try:
        return Decimal(str(v).strip()) if v not in (None, "") else None
    except InvalidOperation:
        return None


def fetch_bhavcopy(d: date) -> list[tuple] | None:
    """[(symbol, open, high, low, close, volume)] one row per symbol (EQ preferred); None = no file (holiday)."""
    url = BHAVCOPY_URL.format(d=d.strftime("%Y%m%d"))
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Referer": "https://www.nseindia.com/"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            blob = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        text = zf.read(zf.namelist()[0]).decode("utf-8", errors="replace")
    best: dict[str, tuple[str, tuple]] = {}
    for row in csv.DictReader(io.StringIO(text)):
        sym = (row.get("TckrSymb") or "").strip().upper()
        vals = [_dec(row.get(k)) for k in ("OpnPric", "HghPric", "LwPric", "ClsPric", "TtlTradgVol")]
        if not sym or None in vals:
            continue
        series = (row.get("SctySrs") or "").strip().upper()
        rec = (sym, vals[0], vals[1], vals[2], vals[3], int(vals[4]))
        cur = best.get(sym)
        if cur is None or (series == "EQ" and cur[0] != "EQ"):
            best[sym] = (series, rec)
    return [r for _s, r in best.values()]


def upsert_day(con, d: date, recs: list[tuple]) -> int:
    if not recs:
        return 0
    cols = list(zip(*recs, strict=True))
    res = con.run(
        "insert into historical_prices (stock_id, trade_date, open, high, low, close, volume) "
        "select s.id, :d, x.o, x.h, x.l, x.c, x.v "
        "from unnest(cast(:sym as text[]), cast(:o as numeric[]), cast(:h as numeric[]), cast(:l as numeric[]), "
        "cast(:c as numeric[]), cast(:v as bigint[])) as x(sym, o, h, l, c, v) "
        "join stocks s on s.symbol = x.sym "
        "on conflict (stock_id, trade_date) do update set open = excluded.open, high = excluded.high, "
        "low = excluded.low, close = excluded.close, volume = excluded.volume "
        "returning 1",
        d=d,
        sym=list(cols[0]),
        o=[str(x) for x in cols[1]],
        h=[str(x) for x in cols[2]],
        l=[str(x) for x in cols[3]],
        c=[str(x) for x in cols[4]],
        v=list(cols[5]),
    )
    return len(res)


def catch_up(con, today: date, max_days: int = 10) -> list[str]:
    """Backfill every weekday after the latest stored date up to `today` (inclusive)."""
    latest = latest_trade_date(con) or today - timedelta(days=max_days)
    log = []
    d = max(latest + timedelta(days=1), today - timedelta(days=max_days))
    while d <= today:
        if d.weekday() < 5:
            recs = fetch_bhavcopy(d)
            if recs is None:
                log.append(f"{d}: no Bhavcopy (holiday or not published yet)")
            else:
                log.append(f"{d}: upserted {upsert_day(con, d, recs)} bars")
        d += timedelta(days=1)
    return log
