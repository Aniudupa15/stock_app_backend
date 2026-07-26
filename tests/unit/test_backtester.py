"""End-to-end backtest over synthetic bars, exercising the full reuse of
venue + OMS + risk + strategy through historical replay. Pure."""

from datetime import date
from decimal import Decimal

from libs.backtest.bar import Bar
from libs.backtest.runner import Backtester
from libs.execution.slippage import SlippageModel
from libs.strategy.engine import ExitRule, Strategy
from libs.trading_domain.enums import ExitReason, Product, Side

NO_SLIP = SlippageModel(base_bps=Decimal("0"), impact_coeff_bps=Decimal("0"), illiquid_penalty_bps=Decimal("0"))
RULE = {"op": "GT", "left": {"feature": "close"}, "right": {"feature": "EMA_20"}}


def _bar(d, o, h, low, c, v=100000):
    return Bar(
        day=d, open=Decimal(str(o)), high=Decimal(str(h)), low=Decimal(str(low)), close=Decimal(str(c)), volume=v
    )


def _strategy():
    return Strategy(
        name="EMA breakout",
        rule=RULE,
        side=Side.BUY,
        product=Product.CNC,
        exit=ExitRule(stop_loss_pct=Decimal("5"), target_pct=Decimal("5")),
        quantity=10,
    )


async def test_backtest_target_hit_produces_winning_trade():
    bars = [
        _bar(date(2026, 1, 5), 100, 100, 100, 100),  # no entry (close 100 !> EMA 101)
        _bar(date(2026, 1, 6), 100, 100, 100, 100),  # entry at close 100 (EMA 99)
        _bar(date(2026, 1, 7), 102, 106, 99, 104),  # intrabar high 106 >= target 105 -> fill
    ]

    # Features per bar index; bar 2 set NOT to re-fire so exactly one trade.
    features = {
        0: ({"close": 100, "EMA_20": 101}, None),
        1: ({"close": 100, "EMA_20": 99}, {"close": 100, "EMA_20": 101}),
        2: ({"close": 104, "EMA_20": 105}, {"close": 100, "EMA_20": 99}),
    }

    bt = Backtester(symbol="INFY", strategy=_strategy(), starting_cash=Decimal("1000000"), slippage=NO_SLIP)
    result = await bt.run(bars, lambda i: features[i])

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason is ExitReason.TARGET
    assert trade.entry_price == Decimal("100")
    assert trade.exit_price == Decimal("105.00")  # +5% target
    assert trade.pnl_gross == Decimal("50")  # (105-100)*10
    assert result.metrics.total_trades == 1
    assert result.metrics.win_rate == 1.0
    assert result.final_equity > result.starting_cash


async def test_backtest_stop_loss_hit_produces_losing_trade():
    bars = [
        _bar(date(2026, 1, 6), 100, 100, 100, 100),  # entry at close 100
        _bar(date(2026, 1, 7), 99, 100, 94, 96),  # intrabar low 94 <= SL 95 -> stop
    ]
    features = {
        0: ({"close": 100, "EMA_20": 99}, None),
        1: ({"close": 96, "EMA_20": 100}, {"close": 100, "EMA_20": 99}),  # non-firing
    }
    bt = Backtester(symbol="INFY", strategy=_strategy(), starting_cash=Decimal("1000000"), slippage=NO_SLIP)
    result = await bt.run(bars, lambda i: features[i])

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason is ExitReason.STOP_LOSS
    # Stop is a stop-MARKET order: the intrabar low (94) breaches the 95 trigger
    # and fills at market = the extreme, worse than the trigger. Deliberately
    # conservative ("never rosier") - trigger-price fills are a later refinement.
    assert result.trades[0].exit_price == Decimal("94.0000")
    assert result.trades[0].pnl_gross == Decimal("-60")  # (94-100)*10
    assert result.metrics.losses == 1


async def test_backtest_no_signal_no_trades():
    bars = [_bar(date(2026, 1, 6), 100, 101, 99, 100), _bar(date(2026, 1, 7), 100, 101, 99, 100)]
    features = {0: ({"close": 100, "EMA_20": 200}, None), 1: ({"close": 100, "EMA_20": 200}, None)}
    bt = Backtester(symbol="INFY", strategy=_strategy(), starting_cash=Decimal("1000000"), slippage=NO_SLIP)
    result = await bt.run(bars, lambda i: features[i])
    assert result.trades == []
    assert result.metrics.total_trades == 0
    assert result.final_equity == result.starting_cash
