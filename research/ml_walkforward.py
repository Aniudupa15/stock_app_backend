"""Step 3: self-training ML ranker, evaluated walk-forward (no look-ahead).

At every monthly rebalance t, for each stock in the liquid universe, features are computed from
data up to the previous close: momentum over several windows, volatility, risk-adjusted momentum,
RSI, distance from moving averages / 52-week high, volume surge, biggest daily move, and the market
regime (index trend, breadth). The label is the stock's return over the following month.

The model for month t is trained ONLY on months whose label had fully ended before t (one-month
embargo), so every pick is made the way a monthly self-retraining bot would make it live.
Portfolio simulation reuses Backtester (same entry/stop/cost rules as the live bot).

    python research/ml_walkforward.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtest import Backtester, Config, load_panel, month_starts, split_report
from sklearn.ensemble import HistGradientBoostingRegressor


def rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


class Features:
    def __init__(self, bt: Backtester):
        C, R = bt.C, bt.rets
        self.bt = bt
        self.f = {
            "ret_21": C / C.shift(21) - 1,
            "ret_63": C / C.shift(63) - 1,
            "ret_126": C / C.shift(126) - 1,
            "ret_252_21": C.shift(21) / C.shift(252) - 1,
            "vol_60": R.rolling(60, min_periods=40).std(),
            "rsi_14": rsi(C),
            "dist_ma50": C / C.rolling(50, min_periods=40).mean() - 1,
            "dist_ma200": C / C.rolling(200, min_periods=150).mean() - 1,
            "dist_hi252": C / C.rolling(252, min_periods=150).max() - 1,
            "turn_surge": bt.T.rolling(20, min_periods=15).mean() / bt.turn60,
            "max_move_21": R.abs().rolling(21, min_periods=15).max(),
        }
        self.f["ret21_per_vol"] = self.f["ret_21"] / self.f["vol_60"]
        self.mkt_trend = bt.mkt / bt.mkt.rolling(200, min_periods=150).mean() - 1
        self.names = [*self.f, "mkt_trend", "breadth"]

    def at(self, i: int, uni: pd.Index) -> pd.DataFrame:
        X = pd.DataFrame({k: v.iloc[i - 1].reindex(uni) for k, v in self.f.items()})
        X["mkt_trend"] = self.mkt_trend.iloc[i - 1]
        X["breadth"] = self.bt.breadth.iloc[i - 1]
        return X


def universe(bt: Backtester, i: int, size: int = 300) -> pd.Index:
    t, px = bt.turn60.iloc[i - 1], bt.C.iloc[i - 1]
    return t[px >= 10].dropna().sort_values(ascending=False).index[:size]


class MLBacktester(Backtester):
    """Backtester whose picks come from precomputed walk-forward model scores."""

    def __init__(self, panel, scores: dict[int, pd.Series]):
        super().__init__(panel)
        self.scores = scores

    def picks(self, i: int, cfg: Config) -> list[str]:
        sc = self.scores.get(i)
        if sc is None or sc.empty:
            return []
        out = []
        win = self.C.iloc[i - 1 - 30 : i]
        for s in sc.sort_values(ascending=False).index:
            w = win[s].values
            if np.isnan(w).any() or (np.abs(w[1:] / w[:-1] - 1) > 0.35).any():
                continue  # same split/jump sanity filter as the live bot
            out.append(s)
            if len(out) == cfg.top:
                break
        return out


def walk_forward(panel, min_train_months: int = 24) -> dict[int, pd.Series]:
    bt = Backtester(panel)
    F = Features(bt)
    reb = [i for i in month_starts(bt.dates) if i >= 260]
    rows = []  # (month index k, X, y)
    for k, i in enumerate(reb[:-1]):
        uni = universe(bt, i)
        X = F.at(i, uni)
        j = reb[k + 1]
        y = (bt.O[uni].iloc[j] / bt.O[uni].iloc[i] - 1).clip(-0.9, 2.0)
        rows.append((k, X, y))
    scores: dict[int, pd.Series] = {}
    for k, (_, X, _y) in enumerate(rows):
        train = [(Xk, yk) for kk, Xk, yk in rows if kk <= k - 2]  # label of month k-1 ends at month k: embargo it
        if len(train) < min_train_months:
            continue
        Xtr = pd.concat([x for x, _ in train])
        ytr = pd.concat([y for _, y in train])
        # rank-transform the target per month: learn "which stocks beat their peers", not market level
        ytr = pd.concat([y.rank(pct=True) for _, y in train])
        ok = ytr.notna()
        model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=200,
            l2_regularization=1.0,
            random_state=0,
        )
        model.fit(Xtr[ok], ytr[ok])
        scores[reb[k]] = pd.Series(model.predict(X), index=X.index)
    return scores


def main() -> None:
    panel = load_panel()
    scores = walk_forward(panel)
    first = min(scores)
    mbt = MLBacktester(panel, scores)
    start = str(mbt.dates[first].date())
    bench = mbt.benchmark(start=start)
    base = Backtester(panel).run(Config(), start=start)
    print(f"walk-forward ML from {start} ({len(scores)} monthly models)\n")
    print("live baseline (30d momentum, -15% stop)")
    print(split_report(base, bench), "\n")
    for cfg in (
        Config(name="ML, -15% stop"),
        Config(name="ML, no stop", stop=None),
        Config(name="ML, no stop, ma200 filter", stop=None, market_filter="ma200"),
    ):
        print(cfg.name)
        print(split_report(mbt.run(cfg, start=start), bench), "\n")


if __name__ == "__main__":
    main()
