"""Monthly cross-sectional momentum backtester on the adjusted panel (research/data/panel.pkl).

Mirrors the live bot: rank on the previous close among the liquid universe (top-N by 60-day
turnover), sanity-filter split-like jumps, enter at the rebalance day's open, hold kept names
(no resize), stop measured from the original entry (intraday low, gap -> fill at open), exited
slots sit in cash until next rebalance. Costs are charged per traded leg.

Variants are judged on two separate periods: design (2015-2021) and exam (2022-2026).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class Config:
    name: str = "baseline (live)"
    lookback: int = 30  # trading days
    skip: int = 0  # skip the most recent N days (e.g. 12-1 momentum)
    rank: str = "ret"  # "ret" | "ret_vol" (return / volatility)
    top: int = 10
    universe: int = 300
    stop: float | None = 0.15
    market_filter: str = "none"  # "none" | "ma200" | "breadth"
    cost: float = 0.002  # per leg (~0.4% round trip)
    min_price: float = 10.0
    extra: dict = field(default_factory=dict, compare=False)


def load_panel() -> dict[str, pd.DataFrame]:
    return pd.read_pickle(DATA / "panel.pkl")


def month_starts(idx: pd.DatetimeIndex) -> list[int]:
    s = pd.Series(range(len(idx)), index=idx)
    return list(s.groupby([idx.year, idx.month]).first().values)


class Backtester:
    def __init__(self, panel: dict[str, pd.DataFrame]):
        self.C = panel["close"]
        self.O = panel["open"].reindex_like(self.C)
        self.L = panel["low"].reindex_like(self.C)
        self.T = panel["turnover"].reindex_like(self.C)
        self.dates = self.C.index
        self.rets = self.C.pct_change(fill_method=None)
        self.turn60 = self.T.rolling(60, min_periods=40).mean()
        # equal-weight index of the broad liquid market, for the market filters
        liquid = self.turn60.rank(axis=1, ascending=False) <= 300
        self.mkt = (1 + self.rets.where(liquid).mean(axis=1).fillna(0)).cumprod()
        above = self.C > self.C.rolling(200, min_periods=150).mean()
        n_liquid = liquid.sum(axis=1).astype(float).replace(0.0, np.nan)
        self.breadth = above.where(liquid).sum(axis=1).astype(float) / n_liquid

    def picks(self, i: int, cfg: Config) -> list[str]:
        j = i - 1 - cfg.skip  # rank on data up to the previous close
        if j < cfg.lookback + 60:
            return []
        t = self.turn60.iloc[i - 1]
        px = self.C.iloc[i - 1]
        uni = t[(px >= cfg.min_price)].dropna().sort_values(ascending=False).index[: cfg.universe]
        win = self.C[uni].iloc[j - cfg.lookback : j + 1]
        ret = win.iloc[-1] / win.iloc[0] - 1
        if cfg.rank == "ret_vol":
            vol = self.rets[uni].iloc[j - 60 : j + 1].std()
            score = ret / vol.replace(0, np.nan)
        else:
            score = ret
        score = score.dropna().sort_values(ascending=False)
        out = []
        for s in score.index:
            w = win[s].values
            if np.isnan(w).any() or (w <= 0).any() or ret[s] > 2.0:
                continue
            if (np.abs(w[1:] / w[:-1] - 1) > 0.35).any():
                continue
            out.append(s)
            if len(out) == cfg.top:
                break
        return out

    def risk_on(self, i: int, cfg: Config) -> bool:
        if cfg.market_filter == "ma200":
            m = self.mkt.iloc[:i]
            return len(m) < 200 or m.iloc[-1] >= m.iloc[-200:].mean()
        if cfg.market_filter == "breadth":
            b = self.breadth.iloc[i - 1]
            return np.isnan(b) or b >= 0.4
        return True

    def run(self, cfg: Config, start: str = "2015-07-01", end: str | None = None) -> pd.Series:
        """Monthly portfolio returns indexed by rebalance date."""
        reb = [i for i in month_starts(self.dates) if self.dates[i] >= pd.Timestamp(start)]
        if end:
            reb = [i for i in reb if self.dates[i] <= pd.Timestamp(end)]
        held: dict[str, float] = {}  # symbol -> original entry price
        out = {}
        for k, i in enumerate(reb):
            j = reb[k + 1] if k + 1 < len(reb) else len(self.dates) - 1
            picks = self.picks(i, cfg) if self.risk_on(i, cfg) else []
            cost = 0.0
            for s in list(held):
                if s not in picks:
                    held.pop(s)
                    cost += cfg.cost / cfg.top
            for s in picks:
                if s not in held and not np.isnan(self.O[s].iloc[i]):
                    held[s] = self.O[s].iloc[i]
                    cost += cfg.cost / cfg.top
            slot = []
            for s in picks:
                if s not in held:
                    slot.append(0.0)
                    continue
                entry, start_px = held[s], self.O[s].iloc[i]
                stop_px = entry * (1 - cfg.stop) if cfg.stop else None
                exit_px = None
                if stop_px is not None:
                    lows = self.L[s].iloc[i:j].values
                    opens = self.O[s].iloc[i:j].values
                    hit = np.where(lows <= stop_px)[0]
                    if len(hit):
                        d = hit[0]
                        exit_px = min(opens[d], stop_px) if d > 0 else stop_px
                if exit_px is not None:
                    held.pop(s)
                    cost += cfg.cost / cfg.top
                    slot.append(exit_px / start_px - 1)
                else:
                    end_px = self.O[s].iloc[j]
                    if np.isnan(end_px):
                        end_px = self.C[s].iloc[: j + 1].dropna().iloc[-1]
                    slot.append(end_px / start_px - 1)
            slot += [0.0] * (cfg.top - len(slot))
            out[self.dates[i]] = float(np.mean(slot)) - cost
        return pd.Series(out)

    def benchmark(self, start: str = "2015-07-01") -> pd.Series:
        reb = [i for i in month_starts(self.dates) if self.dates[i] >= pd.Timestamp(start)]
        out = {}
        for k, i in enumerate(reb):
            j = reb[k + 1] if k + 1 < len(reb) else len(self.dates) - 1
            uni = self.turn60.iloc[i - 1].dropna().sort_values(ascending=False).index[:300]
            r = (self.O[uni].iloc[j] / self.O[uni].iloc[i] - 1).dropna().clip(-0.95, 3)
            out[self.dates[i]] = float(r.mean())
        return pd.Series(out)


def stats(m: pd.Series) -> dict:
    if m.empty:
        return {}
    nav = (1 + m).cumprod()
    yrs = len(m) / 12
    return {
        "months": len(m),
        "CAGR": nav.iloc[-1] ** (1 / yrs) - 1,
        "mean/mo": m.mean(),
        "maxDD": (nav / nav.cummax() - 1).min(),
        "worst mo": m.min(),
        "win%": (m > 0).mean(),
        "sharpe": m.mean() / m.std() * np.sqrt(12) if m.std() else np.nan,
    }


def split_report(m: pd.Series, bench: pd.Series) -> str:
    rows = []
    for label, a, b in (("design 2015-21", "2015", "2021"), ("exam 2022-26", "2022", "2026"), ("all", "2015", "2026")):
        s, bm = stats(m[a:b]), stats(bench[a:b])
        rows.append(
            f"  {label:15} CAGR {s['CAGR']:+7.1%}  mean/mo {s['mean/mo']:+6.2%}  maxDD {s['maxDD']:+7.1%}  "
            f"worst {s['worst mo']:+6.1%}  win {s['win%']:4.0%}  sharpe {s['sharpe']:4.2f}   | market CAGR {bm['CAGR']:+6.1%}"
        )
    return "\n".join(rows)
