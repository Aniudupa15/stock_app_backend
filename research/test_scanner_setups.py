"""Integration #1: do the /nse-scanner daily setups have an edge? Event study on 2015-2026.

Setups are copied exactly from ~/.claude/skills/nse-scanner/scan.py find_setups() (closes only):
  breakout_20d     last close > highest close of the previous 20 sessions
  bull_flag        +10% run from c[-16] to c[-6], then the last 5 closes within a 5% range
  reclaim_ma20     yesterday below its 20-day average, today above it
  oversold         RSI(14, Wilder) < 30
  oversold_bounce  yesterday RSI < 30 and today's close > yesterday's
Universe: the liquid top-500 by 60-day turnover (a stand-in for the Nifty 500), price >= Rs 20,
volume >= 100k shares - the scanner's own liquidity rule.

For each signal on day t, entry is the next day's OPEN (you can only act after the close), exit at
the close N trading days later, minus 0.4% round-trip costs. Compared with the average stock in the
same universe over the same days ("excess"), so a rising market doesn't flatter a setup.
Judged on design (2015-2021) and exam (2022-2026) separately, like the strategy tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtest import load_panel

COST = 0.004
HOLDS = (5, 10, 20)


def wilder_rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def main() -> None:
    p = load_panel()
    C, OP, V, T = p["close"], p["open"], p["volume"], p["turnover"]
    OP = OP.reindex_like(C)
    turn60 = T.reindex_like(C).rolling(60, min_periods=40).mean()
    liquid = (turn60.rank(axis=1, ascending=False) <= 500) & (C >= 20) & (V.reindex_like(C) >= 100_000)

    hi20 = C.shift(1).rolling(20).max()
    ma20 = C.rolling(20).mean()
    run = C.shift(5) / C.shift(15) - 1
    rng5 = (C.rolling(5).max() - C.rolling(5).min()) / C
    r = wilder_rsi(C)
    setups = {
        "breakout_20d": C > hi20,
        "bull_flag": (run > 0.10) & (rng5 < 0.05),
        "reclaim_ma20": (C.shift(1) < ma20.shift(1)) & (C > ma20),
        "oversold": r < 30,
        "oversold_bounce": (r.shift(1) < 30) & (C > C.shift(1)),
    }

    fwd = {h: C.shift(-h) / OP.shift(-1) - 1 - COST for h in HOLDS}  # buy next open, sell close h days later
    base = {h: fwd[h].where(liquid).mean(axis=1) for h in HOLDS}  # average liquid stock, same days

    rows = []
    for name, sig in setups.items():
        s = sig & liquid
        for period, a, b in (("design 2015-21", "2015", "2021"), ("exam 2022-26", "2022", "2026")):
            ss = s.loc[a:b]
            n = int(ss.sum().sum())
            row = {"setup": name, "period": period, "signals": n}
            for h in HOLDS:
                f = fwd[h].loc[a:b][ss]
                exc = f.sub(base[h].loc[a:b], axis=0)
                row[f"{h}d avg"] = np.nanmean(f.values)
                row[f"{h}d excess"] = np.nanmean(exc.values)
                row[f"{h}d win%"] = np.nanmean((f.values > 0)[~np.isnan(f.values)])
            rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    fmt = {c: "{:+.2%}".format for c in df.columns if "avg" in c or "excess" in c}
    fmt |= {c: "{:.0%}".format for c in df.columns if "win" in c}
    print(df.to_string(index=False, formatters=fmt))


if __name__ == "__main__":
    main()
