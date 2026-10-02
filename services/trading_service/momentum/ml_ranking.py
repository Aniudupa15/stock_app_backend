"""ML-based momentum picks: a multi-feature learned alternative to
compute_ranking()'s single trailing-return rule.

Walk-forward backtested before this was built (not in this repo - a
research pass done in conversation): an 8-feature HistGradientBoosting
classifier, predicting whether a stock lands in the top quartile of
21-day-forward returns, beat the raw 30-day-return baseline on every
dimension tested - Information Coefficient +0.049 (vs ~0 for the raw
signal), positive median pick return (vs negative for the baseline), better
Sharpe (0.94 vs 0.68) and lower max drawdown (-21% vs -36%), evaluated with
a properly embargoed expanding-window walk-forward split (no lookahead).

Still experimental: only ~17 independent out-of-sample periods went into
that validation, and the period tested overlaps a stretch of unusually
strong momentum names - not proof of a permanent edge. Treat its picks as
informational, same caution as the rule-based reports, until it has a real
live track record.

Two entry points:
    train_and_save_model(session)  - rebuild the model from all history to date
    compute_ml_ranking(session)    - score today's liquid universe with the
                                      saved model, return the top picks

Both are async and reuse the SAME historical_prices/stocks tables as
compute_ranking() - no separate data source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

FORWARD_HOLD = 21  # ~1 trading month, matches the strategy's stated hold period
UNIVERSE_SIZE = 300  # liquid-universe cap, matches compute_ranking()
TURNOVER_WINDOW = 60
MIN_HISTORY = 70  # warm-up before a symbol enters consideration (needs the 60d lookback + buffer)
FEATURES = ["ret_1d", "ret_5d", "ret_20d", "ret_60d", "vol_20d", "volume_trend", "dist_ma20", "dist_ma60"]

_MODEL_DIR = Path(__file__).resolve().parents[3] / "data" / "ml_models"
_MODEL_PATH = _MODEL_DIR / "momentum_30d.pkl"
_META_PATH = _MODEL_DIR / "momentum_30d_meta.json"


class ModelNotTrainedError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class MLPick:
    symbol: str
    name: str
    last_close: float
    ret_20d: float  # closest available feature to the 30d single-feature picks' headline number
    pred_proba: float  # model's predicted probability of landing in the top quartile of 21d fwd returns

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "last_close": round(self.last_close, 2),
            "ret_20d": round(self.ret_20d, 2),
            "pred_proba": round(self.pred_proba, 4),
        }


async def _load_price_frame(session: AsyncSession, cutoff_days: int | None = None) -> pd.DataFrame:
    """Bulk-load historical_prices+stocks into a DataFrame, same shape used
    throughout - one row per (symbol, trade_date). cutoff_days limits how far
    back to load (None = everything), purely as a perf knob for scoring-only
    calls that don't need the full multi-year history."""
    where = "s.is_active = true"
    params: dict = {}
    if cutoff_days is not None:
        where += " and hp.trade_date >= (select max(trade_date) from historical_prices) - CAST(:cutoff_days AS integer)"
        params["cutoff_days"] = cutoff_days
    rows = (
        await session.execute(
            text(
                f"select s.symbol, s.name, hp.trade_date, hp.close, hp.volume "
                f"from historical_prices hp join stocks s on s.id = hp.stock_id where {where} "
                f"order by s.symbol, hp.trade_date"
            ),
            params,
        )
    ).all()
    df = pd.DataFrame(rows, columns=["symbol", "name", "trade_date", "close", "volume"])
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def _add_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds ret_1d/5d/20d/60d, vol_20d, volume_trend, dist_ma20/60, and
    turnover_avg_60d - all backward-looking only, safe to call for both
    training (where a forward label gets added afterwards) and live scoring
    (where there is no forward label yet)."""
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    g = df.groupby("symbol", group_keys=False)
    df["turnover"] = df["close"] * df["volume"]

    for lb in (1, 5, 20, 60):
        df[f"ret_{lb}d"] = g["close"].pct_change(lb) * 100

    daily_ret = g["close"].pct_change()
    df["daily_ret"] = daily_ret
    df["vol_20d"] = g["daily_ret"].transform(lambda s: s.rolling(20, min_periods=15).std()) * 100

    df["turnover_avg_5d"] = g["turnover"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    df["turnover_avg_60d"] = g["turnover"].transform(lambda s: s.rolling(TURNOVER_WINDOW, min_periods=20).mean())
    df["volume_trend"] = df["turnover_avg_5d"] / df["turnover_avg_60d"]

    ma20 = g["close"].transform(lambda s: s.rolling(20, min_periods=15).mean())
    ma60 = g["close"].transform(lambda s: s.rolling(60, min_periods=40).mean())
    df["dist_ma20"] = (df["close"] / ma20 - 1) * 100
    df["dist_ma60"] = (df["close"] / ma60 - 1) * 100
    return df


def _build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = _add_feature_columns(df)
    g = df.groupby("symbol", group_keys=False)

    df["fwd_close"] = g["close"].shift(-FORWARD_HOLD)
    df["fwd_ret_raw"] = (df["fwd_close"] / df["close"] - 1) * 100

    def _rank_within_liquid(group: pd.DataFrame) -> pd.Series:
        liquid = group.nlargest(UNIVERSE_SIZE, "turnover_avg_60d")
        pct = liquid["fwd_ret_raw"].rank(pct=True)
        return pct.reindex(group.index)

    valid = df.dropna(subset=["fwd_ret_raw", "turnover_avg_60d"])
    rank_pct = valid.groupby("trade_date", group_keys=False).apply(_rank_within_liquid, include_groups=False)
    df["fwd_rank_pct"] = rank_pct
    df["target_top_quartile"] = (df["fwd_rank_pct"] >= 0.75).astype("float")

    out = df.dropna(subset=FEATURES + ["target_top_quartile"]).copy()
    out["_rownum"] = out.groupby("symbol").cumcount()
    out = out[out["_rownum"] >= MIN_HISTORY].drop(columns="_rownum")
    return out


async def train_and_save_model(session: AsyncSession) -> dict:
    """Rebuilds the training set from ALL history to date and fits a fresh
    model, overwriting the saved one. Call this periodically (weekly is
    plenty - an expanding-window model only gets more data, never less)."""
    import joblib

    df = await _load_price_frame(session)
    train = _build_training_frame(df)
    if len(train) < 1000:
        raise ModelNotTrainedError(f"only {len(train)} labeled training rows available - need more price history")

    clf = HistGradientBoostingClassifier(max_iter=150, max_depth=4, learning_rate=0.08, random_state=42)
    clf.fit(train[FEATURES], train["target_top_quartile"])

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, _MODEL_PATH)
    meta = {
        "trained_at": datetime.now(UTC).isoformat(),
        "n_rows": len(train),
        "n_symbols": int(train["symbol"].nunique()),
        "date_range": [str(train["trade_date"].min().date()), str(train["trade_date"].max().date())],
        "features": FEATURES,
        "forward_hold": FORWARD_HOLD,
    }
    _META_PATH.write_text(json.dumps(meta, indent=2))
    return meta


def load_model_metadata() -> dict | None:
    if not _META_PATH.exists():
        return None
    return json.loads(_META_PATH.read_text())


async def compute_ml_ranking(session: AsyncSession, *, top: int = 10) -> list[MLPick]:
    """Scores today's liquid universe with the saved model. Raises
    ModelNotTrainedError if no model has been trained yet - run
    train_and_save_model() (or scripts/train_ml_model.py) first."""
    import joblib

    if not _MODEL_PATH.exists():
        raise ModelNotTrainedError("no trained model found - run scripts/train_ml_model.py first")
    clf = joblib.load(_MODEL_PATH)

    # Only need enough trailing history to compute the longest feature window (60d) + buffer,
    # not the full multi-year history, since we're scoring the single latest date only.
    df = await _load_price_frame(session, cutoff_days=TURNOVER_WINDOW + 40)
    feats = _add_feature_columns(df)
    latest_date = feats["trade_date"].max()
    today = feats[feats["trade_date"] == latest_date].dropna(subset=FEATURES + ["turnover_avg_60d"])
    if today.empty:
        return []

    liquid = today.nlargest(UNIVERSE_SIZE, "turnover_avg_60d").copy()
    liquid["pred_proba"] = clf.predict_proba(liquid[FEATURES])[:, 1]
    top_picks = liquid.nlargest(top, "pred_proba")

    return [
        MLPick(
            symbol=row.symbol,
            name=row.name,
            last_close=float(row.close),
            ret_20d=float(row.ret_20d) if not np.isnan(row.ret_20d) else 0.0,
            pred_proba=float(row.pred_proba),
        )
        for row in top_picks.itertuples()
    ]
