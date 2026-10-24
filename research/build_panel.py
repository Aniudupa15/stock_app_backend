"""Turn research/data/raw/*.csv.gz into a split/bonus-adjusted daily panel (research/data/panel.pkl).

One row per (symbol, day): EQ series preferred, else BE/BZ (stocks NSE moved into surveillance keep
their history). Adjustment: NSE's PREVCLOSE on a corporate-action ex-date is already adjusted, so
    factor_t = prevclose_t / close_(t-1)
is 1.0 on normal days and the split/bonus ratio on ex-dates. Prices are back-adjusted by the
product of all later factors (the usual "adjusted close"); volume is adjusted the other way.

Output: dict of wide DataFrames (index=date, columns=symbol): open/high/low/close (adjusted),
volume (adjusted), turnover (rupees, unadjusted), series.

    python research/build_panel.py
"""

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
SERIES_RANK = {"EQ": 0, "BE": 1, "BZ": 2}


def load_raw() -> pd.DataFrame:
    frames = []
    for f in sorted((DATA / "raw").glob("*.csv.gz")):
        df = pd.read_csv(gzip.open(f, "rt"), dtype={"symbol": str, "series": str, "isin": str})
        df = df[df["series"].isin(SERIES_RANK)]
        df["date"] = pd.Timestamp(f.name[:8])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for c in ("open", "high", "low", "close", "prevclose", "volume", "turnover"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close", "prevclose"])
    df = df[(df["close"] > 0) & (df["prevclose"] > 0)]
    df["srank"] = df["series"].map(SERIES_RANK)
    df = df.sort_values(["symbol", "date", "srank"]).drop_duplicates(["symbol", "date"], keep="first")
    return df.drop(columns="srank")


# Split / bonus / consolidation ratios (new price / old price). Bonus a:b -> b/(a+b); split 10->2 -> 1/5.
NICE = np.array([1 / 2, 1 / 3, 2 / 3, 1 / 4, 3 / 4, 1 / 5, 2 / 5, 3 / 5, 1 / 6, 1 / 8, 1 / 10, 1 / 20, 2.0, 5.0, 10.0])


def adjust(df: pd.DataFrame) -> pd.DataFrame:
    """Back-adjust for splits/bonuses. NSE's old-format PREVCLOSE is NOT adjusted (checked: INFY's
    1:1 bonus on 2015-06-15 shows prevclose 1975 vs open ~990), and the corporate-actions API is
    Akamai-blocked, so detect them from the price: an overnight gap open/prev_close that sits within
    3% of a standard ratio and is at least a 20% move. Real crashes almost never land on such ratios;
    the residual (actual move that day) is kept. Uses PREVCLOSE when NSE did adjust it (UDiFF era)."""
    df = df.sort_values(["symbol", "date"]).copy()
    prev_close = df.groupby("symbol")["close"].shift(1)
    # 1) NSE-adjusted PREVCLOSE (newer files), ignoring market-wide data-gap days
    f_nse = (df["prevclose"] / prev_close).where(prev_close.notna(), 1.0)
    f_nse = f_nse.where((f_nse - 1).abs() > 0.005, 1.0)
    share = (f_nse != 1.0).groupby(df["date"]).transform("mean")
    f_nse = f_nse.where((share < 0.2) & (f_nse > 0.02) & (f_nse < 12), 1.0)
    # 2) price-gap detection against standard ratios, on the open OR the close (a bonus day can also
    #    have a big real move); ratios <= 1/2 get 6% tolerance (a -50% crash is far rarer than a split)
    best, best_err = np.ones(len(df)), np.full(len(df), np.inf)
    for px in (df["open"], df["close"]):
        g = (px / prev_close).to_numpy()
        near = NICE[np.abs(g[:, None] / NICE[None, :] - 1).argmin(axis=1)] if len(df) else np.array([])
        err = np.abs(g / near - 1)
        tol = np.where(near <= 0.5, 0.06, 0.03)
        hit = (err < tol) & (np.abs(g - 1) >= 0.2) & ~np.isnan(g) & (err < best_err)
        best, best_err = np.where(hit, near, best), np.where(hit, err, best_err)
    # (penny stocks are skipped: one price tick there can look like a 3/4 ratio)
    f_gap = pd.Series(np.where(prev_close.to_numpy() >= 5, best, 1.0), index=df.index)
    df["factor"] = f_nse.where(f_nse != 1.0, f_gap)
    # adj_t = raw_t * prod(factor_s for s > t): reverse cumulative product per symbol, shifted by one
    later = df.groupby("symbol")["factor"].transform(lambda s: s[::-1].cumprod()[::-1].shift(-1).fillna(1.0))
    for c in ("open", "high", "low", "close"):
        df[c] = df[c] * later
    df["volume"] = df["volume"] / later
    return df


def main() -> None:
    raw = load_raw()
    print(
        f"raw rows {len(raw):,}, symbols {raw['symbol'].nunique():,}, days {raw['date'].nunique():,} "
        f"({raw['date'].min().date()} -> {raw['date'].max().date()})"
    )
    adj = adjust(raw)
    n_ca = int((adj["factor"] != 1.0).sum())
    print(f"corporate-action adjustments applied: {n_ca:,}")
    panel = {
        c: adj.pivot(index="date", columns="symbol", values=c).sort_index()
        for c in ("open", "high", "low", "close", "volume", "turnover")
    }
    panel["series"] = adj.pivot(index="date", columns="symbol", values="series").sort_index()
    pd.to_pickle(panel, DATA / "panel.pkl")
    c = panel["close"]
    big = (c.pct_change(fill_method=None).abs() > 0.5).sum().sum()
    print(f"saved panel {c.shape}; remaining daily moves >50%: {big} (should be rare: real crashes/circuit cases)")
    print("example adjusted symbols:", adj.loc[adj["factor"] != 1.0, "symbol"].value_counts().head(5).to_dict())
    _ = np  # numpy kept for interactive use


if __name__ == "__main__":
    main()
