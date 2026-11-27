"""Download NSE F&O bhavcopies (real end-of-day option prices) for NIFTY, plus
daily NIFTY 50 and India VIX closes from Kite, for scripts/eod_options_backtest.py.

    python scripts/fetch_fo_bhavcopy.py                     # 2019-01-01 -> today
    python scripts/fetch_fo_bhavcopy.py --from 2018-01-01   # more history
    python scripts/fetch_fo_bhavcopy.py --skip-index        # NSE files only (no Kite login)

Resumable: days already saved are skipped; 404s are holidays. Keeps only NIFTY
index options with <=100 days to expiry and strikes within 25% of the near
future, gzip'd - a few hundred KB per day in data/fo_bhav/NIFTY/.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from libs.options_eod.bhav import parse_fo_bhavcopy, trim, url_for, write_day  # noqa: E402

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
DATA = ROOT / "data" / "fo_bhav"


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} | {msg}", flush=True)


def fetch_day(d: date, store: Path, symbol: str) -> str:
    dest = store / f"{d.isoformat()}.csv.gz"
    if dest.exists() or (store / f"{d.isoformat()}.holiday").exists():
        return "skip"
    req = urllib.request.Request(url_for(d), headers={"User-Agent": UA, "Referer": "https://www.nseindia.com/"})
    blob = None
    for attempt in range(3):
        try:
            blob = urllib.request.urlopen(req, timeout=60).read()
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                (store / f"{d.isoformat()}.holiday").touch()
                return "holiday"
            if attempt == 2:
                return f"error: HTTP {exc.code}"
        except Exception as exc:  # network blip
            if attempt == 2:
                return f"error: {exc}"
        time.sleep(3 * (attempt + 1))
    zf = zipfile.ZipFile(io.BytesIO(blob))
    rows, fut = parse_fo_bhavcopy(zf.read(zf.namelist()[0]).decode("utf-8", errors="replace"), symbol)
    if not rows:
        return "error: no option rows parsed (format change?)"
    write_day(dest, trim(rows, d, fut))
    with (store / "_futures.csv").open("a", newline="") as fh:
        w = csv.writer(fh)
        if fh.tell() == 0:
            w.writerow(["date", "fut_close"])
        w.writerow([d.isoformat(), fut or ""])
    return "ok"


def fetch_index_daily(start: date, end: date, out: Path) -> None:
    """NIFTY 50 and India VIX daily closes from Kite (needs the paid plan's data APIs)."""
    from scripts.fno_paper_trade import login
    from services.trading_service.fno_paper import ReadOnlyKite

    api_key = os.environ.get("KITE_API_KEY") or sys.exit("Missing KITE_API_KEY in .env")
    api_secret = os.environ.get("KITE_API_SECRET") or sys.exit("Missing KITE_API_SECRET in .env")
    kite = ReadOnlyKite(login(api_key, api_secret, os.environ.get("KITE_ACCESS_TOKEN")))
    tokens = {r["tradingsymbol"]: int(r["instrument_token"]) for r in kite.instruments("NSE")}
    series: dict[str, dict[date, float]] = {}
    for name, sym in (("nifty", "NIFTY 50"), ("vix", "INDIA VIX")):
        series[name] = {}
        cursor = start
        while cursor <= end:
            chunk_end = min(end, cursor + timedelta(days=1800))
            log(f"Kite daily {sym} {cursor} -> {chunk_end}")
            for r in kite.historical_data(tokens[sym], cursor, chunk_end, "day"):
                d = r["date"].date() if hasattr(r["date"], "date") else date.fromisoformat(str(r["date"])[:10])
                series[name][d] = float(r["close"])
            cursor = chunk_end + timedelta(days=1)
            time.sleep(0.4)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "nifty", "vix"])
        for d in sorted(series["nifty"]):
            v = series["vix"].get(d)
            w.writerow([d.isoformat(), series["nifty"][d], "" if v is None else v])
    log(f"wrote {len(series['nifty'])} days to {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Download NSE F&O bhavcopies + Kite index/VIX dailies")
    p.add_argument("--from", dest="start", type=date.fromisoformat, default=date(2019, 1, 1))
    p.add_argument("--to", dest="end", type=date.fromisoformat, default=date.today())
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--skip-index", action="store_true", help="don't fetch NIFTY/VIX dailies from Kite")
    args = p.parse_args()

    store = DATA / args.symbol
    store.mkdir(parents=True, exist_ok=True)
    stats = {"ok": 0, "holiday": 0, "skip": 0, "error": 0}
    d = args.start
    while d <= args.end:
        if d.weekday() < 5:
            res = fetch_day(d, store, args.symbol)
            stats[res.split(":")[0]] += 1
            if res.startswith("error"):
                log(f"{d} {res}")
            elif res == "ok":
                time.sleep(0.5)  # be polite to NSE's archive
            if d.day == 1 and res != "skip":
                log(f"{d} ... {stats}")
        d += timedelta(days=1)
    log(f"NSE bhavcopies done: {stats}")
    if not args.skip_index:
        fetch_index_daily(args.start - timedelta(days=400), args.end, DATA / "index_daily.csv")


if __name__ == "__main__":
    main()
