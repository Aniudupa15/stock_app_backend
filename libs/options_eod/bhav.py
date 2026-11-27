"""NSE F&O bhavcopy parsing (both formats) and the compact per-day store.

NSE publishes one end-of-day file per trading day with the OHLC/settlement of
every F&O contract:
  * old format (to 2024-07-05): INSTRUMENT,SYMBOL,EXPIRY_DT,STRIKE_PR,OPTION_TYP,
    OPEN,HIGH,LOW,CLOSE,SETTLE_PR,CONTRACTS,...,OPEN_INT,...
  * UDiFF format (from 2024-07-08): FinInstrmTp,TckrSymb,XpryDt,StrkPric,OptnTp,
    ClsPric,SttlmPric,TtlTradgVol,OpnIntrst,UndrlygPric,...
We keep only one index's options (+ its near-month future close as a spot
proxy) in a small gzip CSV per day, so a multi-year backtest loads quickly.
"""

from __future__ import annotations

import csv
import gzip
import io
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

URL_OLD = "https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{y}/{mon}/fo{d}{mon}{y}bhav.csv.zip"
URL_UDIFF = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{ymd}_F_0000.csv.zip"
UDIFF_FROM = date(2024, 7, 8)

STORE_COLS = ["expiry", "strike", "type", "close", "settle", "volume", "oi"]


def url_for(d: date) -> str:
    if d >= UDIFF_FROM:
        return URL_UDIFF.format(ymd=d.strftime("%Y%m%d"))
    mon = d.strftime("%b").upper()
    return URL_OLD.format(y=d.year, mon=mon, d=d.strftime("%d"))


@dataclass(frozen=True, slots=True)
class OptRow:
    expiry: date
    strike: float
    kind: str  # "CE" | "PE"
    close: float
    settle: float
    volume: int
    oi: int


def _date(text: str) -> date:
    text = text.strip()
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%b-%y", "%d%b%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognised date {text!r}")


def _num(text: str | None) -> float:
    try:
        return float(str(text).strip().replace(",", "")) if text not in (None, "", "-") else 0.0
    except ValueError:
        return 0.0


def parse_fo_bhavcopy(text: str, symbol: str = "NIFTY") -> tuple[list[OptRow], float | None]:
    """(index option rows for `symbol`, near-month future close or None)."""
    reader = csv.DictReader(io.StringIO(text))
    fields = {f.strip(): f for f in (reader.fieldnames or [])}
    udiff = "TckrSymb" in fields

    def get(r: dict, *names: str) -> str:
        for n in names:
            if n in fields:
                return (r.get(fields[n]) or "").strip()
        return ""

    rows: list[OptRow] = []
    futures: list[tuple[date, float]] = []
    for r in reader:
        if udiff:
            tp, sym = get(r, "FinInstrmTp"), get(r, "TckrSymb")
            is_opt, is_fut = tp == "IDO", tp == "IDF"
            exp_s, strike_s, kind = get(r, "XpryDt", "FininstrmActlXpryDt"), get(r, "StrkPric"), get(r, "OptnTp")
            close, settle = _num(get(r, "ClsPric")), _num(get(r, "SttlmPric"))
            volume, oi = int(_num(get(r, "TtlTradgVol"))), int(_num(get(r, "OpnIntrst")))
        else:
            tp, sym = get(r, "INSTRUMENT"), get(r, "SYMBOL")
            is_opt, is_fut = tp == "OPTIDX", tp == "FUTIDX"
            exp_s, strike_s, kind = get(r, "EXPIRY_DT"), get(r, "STRIKE_PR"), get(r, "OPTION_TYP")
            close, settle = _num(get(r, "CLOSE")), _num(get(r, "SETTLE_PR"))
            volume, oi = int(_num(get(r, "CONTRACTS"))), int(_num(get(r, "OPEN_INT")))
        if sym != symbol or not exp_s:
            continue
        if is_fut:
            futures.append((_date(exp_s), close or settle))
        elif is_opt and kind in ("CE", "PE"):
            rows.append(OptRow(_date(exp_s), _num(strike_s), kind, close, settle, volume, oi))
    near_future = min(futures)[1] if futures else None
    return rows, near_future


def write_day(path: Path, rows: list[OptRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(STORE_COLS)
        for r in rows:
            strike = int(r.strike) if r.strike == int(r.strike) else r.strike
            w.writerow([r.expiry.isoformat(), strike, r.kind, r.close, r.settle, r.volume, r.oi])


def read_day(path: Path) -> list[OptRow]:
    with gzip.open(path, "rt", newline="") as fh:
        return [
            OptRow(
                date.fromisoformat(r["expiry"]),
                float(r["strike"]),
                r["type"],
                float(r["close"]),
                float(r["settle"]),
                int(float(r["volume"])),
                int(float(r["oi"])),
            )
            for r in csv.DictReader(fh)
        ]


def trim(rows: list[OptRow], on: date, spot: float | None, max_days: int = 100, band: float = 0.25) -> list[OptRow]:
    """Drop far expiries and strikes far from spot (keeps the store small)."""
    out = [r for r in rows if 0 <= (r.expiry - on).days <= max_days]
    if spot:
        lo, hi = spot * (1 - band), spot * (1 + band)
        out = [r for r in out if lo <= r.strike <= hi]
    return out
