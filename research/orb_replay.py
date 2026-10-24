"""Integration #4 (instant evidence): replay the skills' ORB rules over the last ~60 trading days on the
momentum leaders of each morning (30-day momentum top-20 of the liquid 300, as of the previous
close - the bot's own ranking), using Yahoo's 5-minute history. Rs 10,000 per trade, MIS charges.

    python research/orb_replay.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from backtest import Backtester, Config, load_panel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy" / "momentum_lambda"))
import orb  # noqa: E402


def main() -> None:
    bt = Backtester(load_panel())
    cfg = Config(top=20)
    leaders: dict = {}
    for i in range(len(bt.dates) - 70, len(bt.dates)):
        leaders[bt.dates[i].date()] = bt.picks(i, cfg)
    symbols = sorted({s for v in leaders.values() for s in v})
    print(
        f"{len(leaders)} days ({min(leaders)} -> {max(leaders)}), {len(symbols)} distinct leaders; fetching 5m bars..."
    )
    bars = {s: orb.fetch_5m(s) for s in symbols}
    trades = []
    for day, syms in leaders.items():
        for s in syms:
            b = bars.get(s, {}).get(day)
            if b:
                t = orb.simulate_day(b)
                if t:
                    trades.append({"day": day, "symbol": s, **t})
    df = pd.DataFrame(trades)
    if df.empty:
        print("no trades")
        return
    days_with_bars = {d for s in bars for d in bars[s]}
    print(f"\n{len(df)} paper trades over {df['day'].nunique()} days (5m data covers {len(days_with_bars)} days)")
    print(
        f"win rate {(df['net'] > 0).mean():.0%}   avg R {df['r'].mean():+.2f}   outcomes {df['why'].value_counts().to_dict()}"
    )
    print(
        f"gross Rs{df['gross'].sum():+,.0f}   charges Rs{df['charges'].sum():,.0f}   NET Rs{df['net'].sum():+,.0f}"
        f"   (avg Rs{df['net'].mean():+.1f}/trade on Rs10k = {df['net'].mean() / 100:+.2f}%)"
    )
    daily = df.groupby("day")["net"].sum()
    print(f"days up {(daily > 0).mean():.0%}, best day Rs{daily.max():+,.0f}, worst day Rs{daily.min():+,.0f}")
    by_week = defaultdict(float)
    for d, v in daily.items():
        by_week[pd.Timestamp(d).to_period("W").start_time.date()] += v
    print("weekly net:", " ".join(f"{v:+.0f}" for _k, v in sorted(by_week.items())))


if __name__ == "__main__":
    main()
