"""Load the compact store written by scripts/fetch_fo_bhavcopy.py as DayChains."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from datetime import date
from pathlib import Path

from libs.options_eod.bhav import read_day
from libs.options_eod.chain import DayChain


def load_index_daily(path: Path) -> dict[date, tuple[float, float | None]]:
    """date -> (NIFTY close, India VIX close or None)."""
    out: dict[date, tuple[float, float | None]] = {}
    if not path.exists():
        return out
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            vix = float(r["vix"]) if r.get("vix") else None
            out[date.fromisoformat(r["date"])] = (float(r["nifty"]), vix)
    return out


def load_futures_proxy(path: Path) -> dict[date, float]:
    if not path.exists():
        return {}
    with path.open(newline="") as fh:
        return {date.fromisoformat(r["date"]): float(r["fut_close"]) for r in csv.DictReader(fh) if r["fut_close"]}


def available_days(store: Path) -> list[date]:
    return sorted(date.fromisoformat(p.name[:10]) for p in store.glob("????-??-??.csv.gz"))


def iter_days(store: Path, index_daily: Path, start: date, end: date, warmup_days: int = 400) -> Iterator[DayChain]:
    """Days with option data in [start, end]. Before `start`, index-only days
    (no option rows) are yielded first so EMAs / VIX percentiles warm up -
    strategies can't trade on them (no quotes)."""
    index = load_index_daily(index_daily)
    futures = load_futures_proxy(store / "_futures.csv")
    days = [d for d in available_days(store) if start <= d <= end]
    if days:
        warm_from = date.fromordinal(days[0].toordinal() - warmup_days)
        for d in sorted(index):
            if warm_from <= d < days[0]:
                spot, vix = index[d]
                yield DayChain(d, spot, vix, [])
    for d in days:
        if not (start <= d <= end):
            continue
        spot, vix = index.get(d, (futures.get(d), None))
        if not spot:
            continue
        yield DayChain(d, spot, vix, read_day(store / f"{d.isoformat()}.csv.gz"))
