"""Running total of the daily F&O paper sessions (scripts/state/fno_paper/*.json).

    python scripts/fno_paper_report.py

Compare it with the backtest: for the pe-trend preset the full-year backtest
(modelled premiums) made ~Rs420 per trade after charges with ~40% winners.
Live paper results on REAL premiums are the test that matters.
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

DIR = Path(__file__).resolve().parent / "state" / "fno_paper"


def main() -> None:
    files = sorted(DIR.glob("*.json"))
    if not files:
        sys.exit(f"No paper sessions yet in {DIR}")
    total = Decimal(0)
    trades: list[dict] = []
    print(f"{'Date':10}  {'Contract':24} {'Trades':>6} {'Net P&L':>11} {'Running':>11}")
    for f in files:
        d = json.loads(f.read_text())
        pnl = Decimal(d["net_pnl"])
        total += pnl
        trades += d["trades"]
        print(f"{f.name[:10]:10}  {d['contract']:24} {len(d['trades']):>6} {pnl:>+11,.2f} {total:>+11,.2f}")
    wins = [t for t in trades if Decimal(t["net"]) > 0]
    losses = [t for t in trades if Decimal(t["net"]) < 0]
    gross_win = sum((Decimal(t["net"]) for t in wins), Decimal(0))
    gross_loss = -sum((Decimal(t["net"]) for t in losses), Decimal(0))
    print()
    print(f"Sessions: {len(files)}   Trades: {len(trades)}   Win rate: {len(wins) / len(trades):.0%}" if trades
          else f"Sessions: {len(files)}   Trades: 0")  # fmt: skip
    if trades:
        pf = f"{gross_win / gross_loss:.2f}" if gross_loss else "n/a"
        print(f"Profit factor: {pf}   Net per trade: Rs{total / len(trades):+,.2f}   Total: Rs{total:+,.2f}")
    if len(trades) < 30:
        print(f"Only {len(trades)} trades - too few to judge yet (aim for 30+ before deciding anything).")
    print("PAPER TRADING - no real orders were placed.")


if __name__ == "__main__":
    main()
