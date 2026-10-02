from decimal import Decimal

from services.trading_service.momentum.live_plan import (
    plan_buys,
    plan_sells,
    reconcile_ledger,
    round_to_tick,
    stop_loss_prices,
)

D = Decimal


def test_round_to_tick():
    assert round_to_tick(D("101.03"), D("0.05"), up=True) == D("101.05")
    assert round_to_tick(D("101.03"), D("0.05"), up=False) == D("101.00")
    assert round_to_tick(D("101.00"), D("0.05"), up=False) == D("101.00")


def test_reconcile_never_claims_more_than_broker_holds():
    ledger = {"A": 10, "B": 5, "C": 3}
    broker = {"A": 25, "B": 0}  # A: user also owns some; B: GTT fired; C: gone
    assert reconcile_ledger(ledger, broker) == {"A": 10}


def test_sells_only_bot_holdings_that_dropped_out():
    ledger = {"A": 10, "B": 5}
    sells = plan_sells(["A", "X"], ledger, {"A": D(100), "B": D(50)})
    assert [(o.symbol, o.side, o.qty) for o in sells] == [("B", "SELL", 5)]


def test_buys_new_picks_equal_weight_within_cap():
    prices = {"A": D(100), "X": D(200), "Y": D(50)}
    plan = plan_buys(["A", "X", "Y"], {"A": 10}, prices, free_cash=D(100000), capital_cap=D(3000), top=3)
    # budget = min(3000, 1000 kept + 100000) = 3000, slot 1000; room = 2000 -> 1000 each.
    assert plan.budget == D(3000)
    assert [(o.symbol, o.qty) for o in plan.orders] == [("X", 5), ("Y", 20)]


def test_buys_limited_by_free_cash():
    prices = {"X": D(100), "Y": D(100)}
    plan = plan_buys(["X", "Y"], {}, prices, free_cash=D(1000), capital_cap=D(100000), top=2)
    # 1000 * 0.98 buffer / 2 = 490 each -> 4 shares
    assert [(o.symbol, o.qty) for o in plan.orders] == [("X", 4), ("Y", 4)]


def test_too_expensive_pick_is_skipped():
    plan = plan_buys(["X"], {}, {"X": D(5000)}, free_cash=D(10000), capital_cap=D(10000), top=10)
    assert plan.orders == [] and plan.skipped == ["X"]


def test_one_share_allowed_up_to_overrun_if_cash_left():
    prices = {"A": D(100), "B": D(1200), "C": D(1600)}
    plan = plan_buys(["A", "B", "C"], {}, prices, free_cash=D(3000), capital_cap=D(3000), top=3)
    # slot 980 (cash buffer); B 1200 <= 1.5x -> 1 share; C 1600 > 1470 -> skipped
    assert [(o.symbol, o.qty) for o in plan.orders] == [("A", 9), ("B", 1)]
    assert plan.skipped == ["C"]


def test_overrun_never_exceeds_cash():
    prices = {"A": D(100), "B": D(1400)}
    plan = plan_buys(["A", "B"], {}, prices, free_cash=D(2000), capital_cap=D(2000), top=2)
    # A takes 900 of 1960 -> 1060 left < 1400 -> B skipped
    assert [(o.symbol, o.qty) for o in plan.orders] == [("A", 9)] and plan.skipped == ["B"]


def test_stop_loss_prices():
    trigger, limit = stop_loss_prices(D(100), D(15), D("0.05"))
    assert trigger == D("85.00")
    assert limit == D("82.45")
