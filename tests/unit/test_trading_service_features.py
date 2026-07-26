"""Tests for the data-service <-> engine bridges: real indicator features and
OHLCV->Bar mapping, plus an end-to-end backtest driven by the REAL 12-indicator
engine. Pure - no DB, no network (indicators are pure numpy)."""

from datetime import date
from decimal import Decimal

from app.domain.entities import OhlcvBar
from libs.backtest.bar import Bar
from libs.backtest.runner import Backtester
from libs.execution.slippage import SlippageModel
from libs.strategy.engine import ExitRule, Strategy
from libs.trading_domain.enums import Product, Side
from services.trading_service.features import IndicatorFeatureBuilder, build_feature_rows
from services.trading_service.historical import to_backtest_bars

NO_SLIP = SlippageModel(base_bps=Decimal("0"), impact_coeff_bps=Decimal("0"), illiquid_penalty_bps=Decimal("0"))


def _bars(closes: list[float]) -> list[Bar]:
    out = []
    for i, c in enumerate(closes):
        out.append(Bar(day=date(2026, 1, 1) + _days(i), open=Decimal(str(c)), high=Decimal(str(c + 1)), low=Decimal(str(c - 1)), close=Decimal(str(c)), volume=100000))
    return out


def _days(n):
    from datetime import timedelta

    return timedelta(days=n)


def test_ohlcv_to_backtest_bar_mapping():
    ohlcv = [OhlcvBar(trade_date=date(2026, 1, 5), open=Decimal("100"), high=Decimal("102"), low=Decimal("99"), close=Decimal("101"), volume=12345)]
    bars = to_backtest_bars(ohlcv)
    assert bars[0].day == date(2026, 1, 5)
    assert bars[0].close == Decimal("101")
    assert bars[0].volume == 12345


def test_feature_rows_omit_warmup_and_include_when_ready():
    bars = _bars([100 + i for i in range(60)])  # 60 rising bars
    rows = build_feature_rows(bars)
    assert len(rows) == 60
    # Raw fields always present
    assert rows[0]["close"] == 100.0
    # EMA_20 not ready at index 0 (needs 20 bars) -> omitted
    assert "EMA_20" not in rows[0]
    # ready well past warmup
    assert "EMA_20" in rows[30]
    assert "RSI_14" in rows[30]
    assert "SMA_50" in rows[55]


def test_feature_builder_returns_current_and_prev():
    bars = _bars([100 + i for i in range(40)])
    builder = IndicatorFeatureBuilder(bars)
    cur, prev = builder(30)
    assert prev is not None
    assert cur["close"] == 130.0
    first_cur, first_prev = builder(0)
    assert first_prev is None


async def test_backtest_with_real_indicator_features():
    # Uptrend then a dip: price rises to establish EMA, strategy buys when
    # close > EMA_20, exits on a fixed target. Proves real indicators feed the
    # engine end-to-end.
    closes = [100 + i for i in range(30)] + [131, 133, 135, 138, 140]
    bars = _bars(closes)
    builder = IndicatorFeatureBuilder(bars)

    strategy = Strategy(
        name="EMA20 trend",
        rule={"op": "GT", "left": {"feature": "close"}, "right": {"feature": "EMA_20"}},
        side=Side.BUY,
        product=Product.CNC,
        exit=ExitRule(stop_loss_pct=Decimal("3"), target_pct=Decimal("2")),
        quantity=10,
    )
    bt = Backtester(symbol="TEST", strategy=strategy, starting_cash=Decimal("1000000"), slippage=NO_SLIP)
    result = await bt.run(bars, builder)

    # In a persistent uptrend close stays above EMA_20, so at least one bracket
    # opens and its +2% target is hit -> at least one trade recorded.
    assert result.metrics.total_trades >= 1
    assert result.final_equity != result.starting_cash
