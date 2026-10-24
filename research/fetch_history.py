"""Download every NSE cash-market Bhavcopy since 2015 into research/data/raw/ (local only - the
Neon free tier can't hold this). Resumable: days already saved are skipped; 404s are holidays.

Unlike the `stocks` table, these files include companies that were later delisted, so research on
them has no survivorship bias. PREVCLOSE (NSE's adjusted previous close) lets build_panel.py
back-adjust prices for splits and bonuses.

    python research/fetch_history.py [START=2015-01-01] [END=today]
"""

import csv
import gzip
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

RAW = Path(__file__).resolve().parent / "data" / "raw"
UDIFF_FROM = date(2024, 7, 8)  # NSE switched to the UDiFF Bhavcopy format here
OLD = "https://nsearchives.nseindia.com/content/historical/EQUITIES/{y}/{mon}/cm{d}{mon}{y}bhav.csv.zip"
NEW = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
COLS = ["symbol", "series", "open", "high", "low", "close", "prevclose", "volume", "turnover", "isin"]


def url_for(d: date) -> str:
    if d >= UDIFF_FROM:
        return NEW.format(ymd=d.strftime("%Y%m%d"))
    return OLD.format(y=d.year, mon=d.strftime("%b").upper(), d=d.strftime("%d"))


def parse(d: date, text: str) -> list[list]:
    out = []
    for r in csv.DictReader(io.StringIO(text)):
        if d >= UDIFF_FROM:
            row = [
                r.get(k, "").strip()
                for k in (
                    "TckrSymb",
                    "SctySrs",
                    "OpnPric",
                    "HghPric",
                    "LwPric",
                    "ClsPric",
                    "PrvsClsgPric",
                    "TtlTradgVol",
                    "TtlTrfVal",
                    "ISIN",
                )
            ]
        else:
            row = [
                r.get(k, "").strip()
                for k in (
                    "SYMBOL",
                    "SERIES",
                    "OPEN",
                    "HIGH",
                    "LOW",
                    "CLOSE",
                    "PREVCLOSE",
                    "TOTTRDQTY",
                    "TOTTRDVAL",
                    "ISIN",
                )
            ]
        if row[0] and all(row[2:9]):
            out.append(row)
    return out


def fetch(d: date) -> str:
    """'ok', 'holiday' or 'error: ...'"""
    dest = RAW / f"{d:%Y%m%d}.csv.gz"
    if dest.exists() or (RAW / f"{d:%Y%m%d}.holiday").exists():
        return "skip"
    req = urllib.request.Request(url_for(d), headers={"User-Agent": UA, "Referer": "https://www.nseindia.com/"})
    for attempt in range(3):
        try:
            blob = urllib.request.urlopen(req, timeout=60).read()
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                (RAW / f"{d:%Y%m%d}.holiday").touch()
                return "holiday"
            if attempt == 2:
                return f"error: HTTP {exc.code}"
        except Exception as exc:  # network blip
            if attempt == 2:
                return f"error: {exc}"
        time.sleep(3 * (attempt + 1))
    zf = zipfile.ZipFile(io.BytesIO(blob))
    rows = parse(d, zf.read(zf.namelist()[0]).decode("utf-8", errors="replace"))
    with gzip.open(dest, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        w.writerows(rows)
    return "ok"


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    start = date.fromisoformat(args[0]) if args else date(2015, 1, 1)
    end = date.fromisoformat(args[1]) if len(args) > 1 else date.today()
    RAW.mkdir(parents=True, exist_ok=True)
    d, stats = start, {"ok": 0, "holiday": 0, "skip": 0, "error": 0}
    while d <= end:
        if d.weekday() < 5 or "--weekends" in sys.argv:  # weekend special sessions (Budget, Muhurat)
            res = fetch(d)
            stats[res.split(":")[0]] += 1
            if res.startswith("error"):
                print(f"{d} {res}", flush=True)
            elif res == "ok":
                time.sleep(0.4)  # be polite to NSE's archive
            if d.day == 1 and res != "skip":
                print(f"{d} ... {stats}", flush=True)
        d += timedelta(days=1)
    print("done", stats, flush=True)


if __name__ == "__main__":
    main()
