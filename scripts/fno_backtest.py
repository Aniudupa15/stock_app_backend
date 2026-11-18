"""Backtest the intraday F&O paper strategy on Zerodha 1-minute history.

    python scripts/fno_backtest.py --underlying NIFTY --instrument FUT --from 2025-11-01
    python scripts/fno_backtest.py --underlying NIFTY --instrument CE --from 2026-05-01 --to 2026-11-14
    python scripts/fno_backtest.py --underlying BANKNIFTY --instrument PE --lots 2

Data: Kite `historical_data` (needs historical-data access on your Kite
Connect app) for the index/stock and, for options, India VIX. Downloaded in
60-day chunks and cached as CSV in scripts/state/fno_backtest/, so re-runs
with other settings don't re-download. Read-only: no order API is touched.

No Kite? Pass --csv (columns: ts,open,high,low,close; ts ISO with +05:30) and,
for options, --vix-csv in the same format, plus --lot-size.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
import time as time_mod
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from libs.broker.kite_quotes import spot_key  # noqa: E402
from libs.derivatives.contracts import ContractRegistry, contracts_from_kite_instruments  # noqa: E402
from libs.trading_calendar.calendar import IST  # noqa: E402
from services.trading_service.fno_backtest import (  # noqa: E402
    Candle,
    FnoBacktestConfig,
    breakeven_variants,
    retry_variants,
    run_fno_backtest,
    sweep_table,
    sweep_variants,
    target_variants,
)  # noqa: E402
from services.trading_service.fno_paper import ReadOnlyKite  # noqa: E402
from services.trading_service.fno_rules import PRESETS, parse_with_preset  # noqa: E402

CACHE_DIR = ROOT / "scripts" / "state" / "fno_backtest"
CHUNK_DAYS = 60  # Kite's limit for minute candles per request


def log(msg: str) -> None:
    print(f"{datetime.now(IST).isoformat(timespec='seconds')} | {msg}", flush=True)


def _aware(dt: datetime) -> datetime:
    return IST.localize(dt) if dt.tzinfo is None else dt.astimezone(IST)


def read_csv(path: Path) -> list[Candle]:
    with path.open(newline="") as f:
        return [
            Candle(
                _aware(datetime.fromisoformat(r["ts"])),
                Decimal(r["open"]),
                Decimal(r["high"]),
                Decimal(r["low"]),
                Decimal(r["close"]),
            )
            for r in csv.DictReader(f)
        ]


def write_csv(path: Path, candles: list[Candle]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "open", "high", "low", "close"])
        for c in candles:
            w.writerow([c.ts.isoformat(), c.open, c.high, c.low, c.close])


def instrument_token(kite, key: str) -> int:
    exchange, symbol = key.split(":", 1)
    for row in kite.instruments(exchange):
        if row["tradingsymbol"] == symbol:
            return int(row["instrument_token"])
    raise SystemExit(f"instrument {key} not found in Kite's {exchange} list")


KITE_INTERVALS = {"1m": "minute", "5m": "5minute"}


def fetch_minutes(kite, key: str, start: date, end: date, interval: str = "1m") -> list[Candle]:
    cache = CACHE_DIR / f"{key.replace(':', '_').replace(' ', '_')}_{interval}_{start}_{end}.csv"
    if cache.exists():
        log(f"Using cached {cache.name}")
        return read_csv(cache)
    token = instrument_token(kite, key)
    out: list[Candle] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + timedelta(days=CHUNK_DAYS - 1))
        log(f"Downloading {key} {interval} candles from Kite {cursor} -> {chunk_end}")
        rows = kite.historical_data(token, cursor, chunk_end, KITE_INTERVALS[interval])
        out += [
            Candle(
                _aware(r["date"]),
                Decimal(str(r["open"])),
                Decimal(str(r["high"])),
                Decimal(str(r["low"])),
                Decimal(str(r["close"])),
            )
            for r in rows
        ]
        cursor = chunk_end + timedelta(days=1)
        time_mod.sleep(0.4)  # stay under Kite's 3 requests/second
    write_csv(cache, out)
    return out


def fetch_yahoo(name: str, start: date, end: date, interval: str) -> list[Candle]:
    from services.trading_service.yahoo_intraday import available_start, fetch_candles, yahoo_symbol

    first = max(start, available_start(interval))
    if first > start:
        log(f"Yahoo only serves {interval} candles from {first} - backtesting {first} -> {end}")
    cache = CACHE_DIR / f"yahoo_{yahoo_symbol(name).strip('^')}_{interval}_{first}_{end}.csv"
    if cache.exists():
        log(f"Using cached {cache.name}")
        return read_csv(cache)
    log(f"Downloading {name} {interval} candles from Yahoo Finance {first} -> {end}")
    out = fetch_candles(name, interval, first, end)
    write_csv(cache, out)
    return out


class CandleSource:
    """Kite historical data, falling back to Yahoo when the Kite Connect app
    lacks historical-data permission (or when --source yahoo)."""

    def __init__(self, kite, source: str, interval: str) -> None:
        self.kite, self.source, self.interval = kite, source, interval

    def get(self, name: str, kite_key: str, start: date, end: date) -> list[Candle]:
        if self.source != "yahoo":
            try:
                return fetch_minutes(self.kite, kite_key, start, end, self.interval)
            except Exception as exc:
                if self.source == "kite" or "permission" not in str(exc).lower():
                    raise
                log("Kite says this app has no historical-data permission - using Yahoo Finance instead")
                self.source = "yahoo"
        return fetch_yahoo(name, start, end, self.interval)


def current_lot_size(kite, underlying: str) -> int:
    registry = ContractRegistry(
        [c for c in contracts_from_kite_instruments(kite.instruments("NFO")) if c.underlying == underlying]
    )
    fut = registry.nearest_future(underlying, today=date.today())
    if fut is None:
        raise SystemExit(f"no NFO future found for {underlying} - pass --lot-size")
    return fut.lot_size


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest the intraday F&O strategy on 1-minute history")
    p.add_argument("--underlying", default="NIFTY")
    p.add_argument("--instrument", default="FUT", choices=["FUT", "CE", "PE"])
    p.add_argument("--from", dest="start", type=date.fromisoformat, default=date.today() - timedelta(days=365))
    p.add_argument("--to", dest="end", type=date.fromisoformat, default=date.today() - timedelta(days=1))
    p.add_argument("--lots", type=int, default=1)
    p.add_argument("--lot-size", type=int, help="default: today's lot size from Kite")
    p.add_argument("--capital", type=Decimal, default=Decimal("500000"))
    p.add_argument("--max-trades", type=int, default=5)
    p.add_argument("--csv", type=Path, help="underlying 1-minute candles (skip Kite)")
    p.add_argument("--vix-csv", type=Path, help="India VIX 1-minute candles (options, skip Kite)")
    p.add_argument("--access-token", default=os.environ.get("KITE_ACCESS_TOKEN"))
    p.add_argument("--request-token", help="request_token from the Kite login redirect URL")
    p.add_argument(
        "--source",
        default="auto",
        choices=["auto", "kite", "yahoo"],
        help="auto = Kite, falling back to Yahoo if Kite refuses historical data",
    )
    p.add_argument(
        "--interval",
        default="1m",
        choices=["1m", "5m"],
        help="candle size; Yahoo serves 1m for ~30 days, 5m for ~60 days",
    )
    p.add_argument("--preset", choices=sorted(PRESETS), help="a tested variant, e.g. pe-trend (flags override it)")
    p.add_argument("--sl-pct", type=Decimal, help="stop-loss %% (default: 0.5 futures / 15 options)")
    p.add_argument("--target-pct", type=Decimal, help="target %% (default: 1 futures / 30 options)")
    p.add_argument("--no-target", action="store_true", help="no fixed target: stop + trailing stop + 15:15 exit")
    p.add_argument("--trail-pct", type=Decimal, help="trailing-stop distance %%")
    p.add_argument(
        "--trend-filter",
        default="none",
        choices=["none", "open", "orb"],
        help="open: trade only with the day (index vs its open); orb: 15-min opening-range breakout",
    )
    p.add_argument("--expiry", default="near", choices=["near", "next"], help="options: nearest or next expiry")
    p.add_argument("--sweep", action="store_true", help="run a grid of variants and print a comparison table")
    p.add_argument("--breakeven-pct", type=Decimal, help="move the stop to the entry price once up this %%")
    p.add_argument(
        "--breakeven-sweep",
        action="store_true",
        help="same settings, only the breakeven rule changes: none vs +20/40/60%%",
    )
    p.add_argument(
        "--retry-after-skip",
        action="store_true",
        help="a too-expensive signal doesn't end the day; keep watching for a cheaper one",
    )
    p.add_argument(
        "--retry-sweep",
        action="store_true",
        help="compare: skip ends the day vs keep watching (everything else unchanged)",
    )
    p.add_argument(
        "--max-risk",
        type=Decimal,
        help="skip signals whose stop-loss risk exceeds this, like the live bot (e.g. 3000)",
    )
    p.add_argument(
        "--target-sweep",
        action="store_true",
        help="same settings, only the exit changes: no target vs 30/60/100%% targets",
    )
    p.add_argument("--trades-csv", type=Path, help="also write every trade to this CSV")
    args = parse_with_preset(p)

    underlying = args.underlying.upper()
    is_option = args.instrument != "FUT"
    kite = None
    if not args.csv or (is_option and not args.vix_csv) or not args.lot_size:
        from scripts.fno_paper_trade import login

        api_key = os.environ.get("KITE_API_KEY") or sys.exit("Missing KITE_API_KEY in .env")
        api_secret = os.environ.get("KITE_API_SECRET") or sys.exit("Missing KITE_API_SECRET in .env")
        kite = ReadOnlyKite(login(api_key, api_secret, args.access_token, args.request_token))

    source = CandleSource(kite, args.source, args.interval)
    candles = read_csv(args.csv) if args.csv else source.get(underlying, spot_key(underlying), args.start, args.end)
    vix = None
    if is_option:
        vix_candles = (
            read_csv(args.vix_csv) if args.vix_csv else source.get("INDIA VIX", "NSE:INDIA VIX", args.start, args.end)
        )
        vix = [(c.ts, c.close) for c in vix_candles]
    lot_size = args.lot_size or current_lot_size(kite, underlying)
    if not candles:
        sys.exit("no candles in the requested range")

    config = FnoBacktestConfig(
        underlying=underlying,
        instrument=args.instrument,
        lots=args.lots,
        lot_size=lot_size,
        capital=args.capital,
        max_trades=args.max_trades,
        candle_interval=args.interval,
        max_trade_risk=args.max_risk,
        breakeven_pct=args.breakeven_pct,
        retry_after_skip=args.retry_after_skip,
        sl_pct=args.sl_pct,
        target_pct=args.target_pct,
        no_target=args.no_target,
        trail_pct=args.trail_pct,
        trend_filter=args.trend_filter,
        expiry=args.expiry,
    )
    if args.sweep or args.target_sweep or args.breakeven_sweep or args.retry_sweep:
        if args.retry_sweep:
            variants = retry_variants(config)
        elif args.breakeven_sweep:
            variants = breakeven_variants(config)
        elif args.target_sweep:
            variants = target_variants(config)
        else:
            variants = sweep_variants(config)
        log(f"Sweeping {len(variants)} variants over {len(candles):,} candles ...")
        results = [asyncio.run(run_fno_backtest(candles, v, vix=vix)) for v in variants]
        first = results[0]
        print(f"Sweep: {underlying} {args.instrument}, {args.lots} lot(s) x {lot_size}, {args.interval} bars, "
              f"{first.first_day} -> {first.last_day} ({first.days} trading days)")  # fmt: skip
        print("\n".join(sweep_table(results)))
        best = max(results, key=lambda r: r.metrics.net_pnl)
        print(f"\nBest: {best.config.label} ({best.config.describe()})")
        print("A small sample: the best of several variants usually looks better than it will trade.")
        result = best
    else:
        log(f"Replaying {len(candles):,} candles ...")
        result = asyncio.run(run_fno_backtest(candles, config, vix=vix))
        print("\n".join(result.lines()))

    if args.trades_csv:
        with args.trades_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["entry_ts", "exit_ts", "qty", "entry", "exit", "gross", "charges", "net", "reason"])
            for t in result.trades:
                w.writerow([t.entry_ts, t.exit_ts, t.qty, t.entry_price, t.exit_price, t.pnl_gross,
                            t.charges_total, t.pnl_net, t.exit_reason.value if t.exit_reason else ""])  # fmt: skip
        log(f"Trades written to {args.trades_csv}")


if __name__ == "__main__":
    main()
