"""Backtest the end-of-day NIFTY option strategies on REAL NSE option prices.

First download data (once; resumable):  python scripts/fetch_fo_bhavcopy.py

    python scripts/eod_options_backtest.py --strategy condor            # default settings, full report
    python scripts/eod_options_backtest.py --strategy condor --grid     # settings grid + 70/30 walk-forward
    python scripts/eod_options_backtest.py --strategy credit --grid
    python scripts/eod_options_backtest.py --strategy vixspike --grid
    python scripts/eod_options_backtest.py --strategy debit --grid
    python scripts/eod_options_backtest.py --strategy all --grid         # everything in one pass

Grid rows are sorted by the TRAIN period (first 70%); the TEST columns are the
untouched last 30%. Pick settings on train only; believe them only if test
holds up and neighbouring settings look similar.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from libs.options_eod.data import available_days, iter_days  # noqa: E402
from libs.options_eod.engine import run_eod  # noqa: E402
from libs.options_eod.report import grid_table, split_date, summary_lines  # noqa: E402
from libs.options_eod.strategies import (  # noqa: E402
    MonthlyIronCondor,
    TrendCreditSpread,
    TrendDebitSpread,
    VixSpikeCondor,
)

DATA = ROOT / "data" / "fo_bhav"


def grid(name: str) -> list:
    if name == "condor":
        return [
            MonthlyIronCondor(delta=d, wing=w, take=t, exit_dte=(None if t is None else 14), stop_mult=s)
            for d, w, t, s in itertools.product((0.10, 0.15, 0.20), (100, 200, 300), (0.5, 0.75, None), (1.5, 2.0, 3.0))
        ]
    if name == "credit":
        return [
            TrendCreditSpread(fast=f, slow=s, delta=d, cycle=c)
            for (f, s), d, c in itertools.product(
                ((10, 30), (20, 50), (50, 200)), (0.20, 0.25, 0.30, 0.35), ("weekly", "monthly")
            )
        ]
    if name == "vixspike":
        return [VixSpikeCondor(pct=p, wait_turn=w) for p, w in itertools.product((0.7, 0.8, 0.9), (True, False))]
    if name == "debit":
        return [
            TrendDebitSpread(fast=f, slow=s, width=w)
            for (f, s), w in itertools.product(((10, 30), (20, 50), (50, 200)), (200, 300))
        ]
    raise SystemExit(f"unknown strategy {name}")


def default(name: str):
    return {
        "condor": MonthlyIronCondor(),
        "credit": TrendCreditSpread(),
        "vixspike": VixSpikeCondor(),
        "debit": TrendDebitSpread(),
    }[name]


def main() -> None:
    p = argparse.ArgumentParser(description="EOD NIFTY option strategy backtests on real NSE prices")
    p.add_argument("--strategy", default="condor", choices=["condor", "credit", "vixspike", "debit", "all"])
    p.add_argument("--grid", action="store_true", help="run the settings grid with a 70/30 train/test split")
    p.add_argument("--from", dest="start", type=date.fromisoformat, default=date(2019, 1, 1))
    p.add_argument("--to", dest="end", type=date.fromisoformat, default=date.today())
    p.add_argument("--capital", type=float, default=200000, help="for the 20%% max-drawdown criterion")
    p.add_argument("--lot", type=int, default=65, help="today's NIFTY lot (P&L is points x this)")
    p.add_argument("--slippage", type=float, default=1.0, help="points per leg per side")
    p.add_argument("--train", type=float, default=0.7, help="train fraction for the walk-forward split")
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--data", type=Path, default=DATA, help="folder written by fetch_fo_bhavcopy.py")
    args = p.parse_args()

    data = args.data
    store = data / args.symbol
    days = [d for d in available_days(store) if args.start <= d <= args.end]
    if not days:
        sys.exit(f"No data in {store} - run: python scripts/fetch_fo_bhavcopy.py")
    if not (data / "index_daily.csv").exists():
        print("WARNING: data/fo_bhav/index_daily.csv missing - no India VIX (condor/vixspike won't trade); "
              "spot falls back to the near future")  # fmt: skip
    cut = split_date(days[0], days[-1], args.train)
    names = ["condor", "credit", "vixspike", "debit"] if args.strategy == "all" else [args.strategy]
    strategies = [s for n in names for s in (grid(n) if args.grid else [default(n)])]
    print(f"{len(days)} trading days {days[0]} -> {days[-1]}; train < {cut} <= test; "
          f"{len(strategies)} variant(s); lot {args.lot}, slippage {args.slippage} pt/leg/side")  # fmt: skip
    books = run_eod(iter_days(store, data / "index_daily.csv", days[0], days[-1]), strategies,
                    lot=args.lot, slippage=args.slippage)  # fmt: skip

    for name in names:
        group = [b for b, s in zip(books, strategies, strict=True) if type(s) is type(default(name))]
        print()
        if args.grid:
            print(f"=== {group[0].label}: {len(group)} settings, sorted by TRAIN profit factor ===")
            print("\n".join(grid_table(group, cut, args.capital)))
            best = group and max(group, key=lambda b: (len([t for t in b.trades if t.entry_date < cut]) >= 15,
                                                       _train_pf(b, cut)))  # fmt: skip
            print("\nBest on TRAIN, full report (judge it by its TEST line):")
            print("\n".join(summary_lines(best, cut, args.capital)))
        else:
            print("\n".join(summary_lines(group[0], cut, args.capital)))
    print("\nEnd-of-day closes only: stops/targets are checked at 15:30, fills at the close +/- slippage.")
    print("Past results do not guarantee future returns. Not investment advice.")


def _train_pf(book, cut) -> float:
    from libs.options_eod.report import stats

    s = stats([t for t in book.trades if t.entry_date < cut])
    return s.profit_factor or 0.0


if __name__ == "__main__":
    main()
