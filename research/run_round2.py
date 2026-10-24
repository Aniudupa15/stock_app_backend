"""Round 2: combine the two variants that won BOTH periods in round 1 (12-1 momentum and
volatility-adjusted ranking), each with the live -15% stop, a -25% stop and no stop."""

from backtest import Backtester, Config, load_panel, split_report

VARIANTS = [
    Config(name="12-1, -15% stop", lookback=231, skip=21),
    Config(name="12-1, no stop", lookback=231, skip=21, stop=None),
    Config(name="12-1 vol-adjusted, -15% stop", lookback=231, skip=21, rank="ret_vol"),
    Config(name="12-1 vol-adjusted, -25% stop", lookback=231, skip=21, rank="ret_vol", stop=0.25),
    Config(name="12-1 vol-adjusted, no stop", lookback=231, skip=21, rank="ret_vol", stop=None),
    Config(name="6m vol-adjusted, -15% stop", lookback=126, rank="ret_vol"),
]

if __name__ == "__main__":
    bt = Backtester(load_panel())
    bench = bt.benchmark()
    for cfg in VARIANTS:
        print(cfg.name)
        print(split_report(bt.run(cfg), bench), "\n")
