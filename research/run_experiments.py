"""Step 2: test the "quick win" variants against the live config on 2015-2026 adjusted data.

Each variant changes ONE thing vs the live bot (plus a couple of combinations at the end), and is
reported separately for the design period (2015-2021) and the exam period (2022-2026). A variant
only counts as better if it wins in BOTH periods.

    python research/build_panel.py && python research/run_experiments.py
"""

from backtest import Backtester, Config, load_panel, split_report

VARIANTS = [
    Config(),  # live: 30d momentum, top-10 of liquid 300, -15% stop
    Config(name="no stop", stop=None),
    Config(name="stop -25%", stop=0.25),
    Config(name="market filter: index > 200d avg", market_filter="ma200"),
    Config(name="market filter: breadth >= 40%", market_filter="breadth"),
    Config(name="volatility-adjusted rank", rank="ret_vol"),
    Config(name="3-month momentum", lookback=63),
    Config(name="6-month momentum", lookback=126),
    Config(name="12-1 momentum", lookback=231, skip=21),
    Config(name="top-20", top=20),
    Config(name="combo: vol-adj + ma200 + stop -25%", rank="ret_vol", market_filter="ma200", stop=0.25),
    Config(name="combo: 6m + ma200 + no stop", lookback=126, market_filter="ma200", stop=None),
]


def main() -> None:
    bt = Backtester(load_panel())
    bench = bt.benchmark()
    print(f"data {bt.dates[0].date()} -> {bt.dates[-1].date()}, {bt.C.shape[1]} symbols (incl. later-delisted)\n")
    for cfg in VARIANTS:
        m = bt.run(cfg)
        print(cfg.name)
        print(split_report(m, bench))
        print()


if __name__ == "__main__":
    main()
