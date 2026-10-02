from datetime import date
from decimal import Decimal

from services.trading_service.momentum.live_plan import LiveOrder
from services.trading_service.momentum.live_safety import (
    check_account,
    check_buy_orders,
    check_coverage,
    check_freshness,
    filter_picks,
    pick_problem,
    price_deviations,
)

D = Decimal
WED, TUE, MON, FRI = date(2026, 9, 30), date(2026, 9, 29), date(2026, 9, 28), date(2026, 9, 25)


def test_freshness_passes_with_previous_trading_day():
    assert check_freshness(TUE, WED, set()).ok


def test_freshness_blocks_stale_data():
    c = check_freshness(MON, WED, set())
    assert not c.ok and "NSE_HOLIDAYS" in c.detail


def test_freshness_monday_expects_friday_and_respects_holidays():
    assert check_freshness(FRI, MON, set()).ok
    assert check_freshness(MON, WED, {TUE}).ok  # Tuesday a listed holiday
    assert not check_freshness(None, WED, set()).ok


def test_coverage():
    assert check_coverage(2300, 2310).ok
    assert not check_coverage(900, 2300).ok  # half-loaded day
    assert not check_coverage(1600, 2300).ok  # below 90% of previous day


def _series(start=100.0, daily=1.01, n=31):
    out = [start]
    for _ in range(n - 1):
        out.append(out[-1] * daily)
    return out


def test_pick_problem_flags_split_like_jump_and_absurd_return():
    assert pick_problem(_series()) is None
    jumpy = _series()
    jumpy[20:] = [x * 2 for x in jumpy[20:]]
    assert "single-day" in pick_problem(jumpy)
    assert "unadjusted" in pick_problem(_series(daily=1.04))  # ~224% in 30d
    assert pick_problem([100.0] * 5) == "incomplete price history"


def test_filter_picks_backfills_from_next_candidates():
    bad = _series(daily=1.04)
    series = {"A": _series(), "B": bad, "C": _series(), "D": _series()}
    picks, excluded, check = filter_picks(["A", "B", "C", "D"], series, top=3)
    assert picks == ["A", "C", "D"] and check.ok and excluded[0].startswith("B:")
    _, _, short = filter_picks(["A", "B"], series, top=3)
    assert not short.ok


def test_account_check():
    assert check_account(None, "AB1234").ok
    assert check_account("ab1234", "AB1234").ok
    assert not check_account("XY9999", "AB1234").ok


def test_price_deviations():
    dev = price_deviations({"A": D(100), "B": D(50)}, {"A": D(98), "B": D(100)})
    assert list(dev) == ["B"]


def test_buy_order_limits():
    ok = [LiveOrder("X", "BUY", 5, D(190))]
    assert check_buy_orders(ok, kept_value=D(1000), per_stock=D(1000), capital_cap=D(3000), top=3).ok
    big = [LiveOrder("X", "BUY", 50, D(190))]
    assert not check_buy_orders(big, kept_value=D(0), per_stock=D(1000), capital_cap=D(3000), top=3).ok
    over_cap = [LiveOrder("X", "BUY", 5, D(190))]
    assert not check_buy_orders(over_cap, kept_value=D(2900), per_stock=D(1000), capital_cap=D(3000), top=3).ok


def test_single_share_overrun_allowed_up_to_limit():
    one = [LiveOrder("X", "BUY", 1, D(1400))]
    assert check_buy_orders(one, kept_value=D(0), per_stock=D(1000), capital_cap=D(10000), top=10).ok
    too_big = [LiveOrder("X", "BUY", 1, D(1600))]
    assert not check_buy_orders(too_big, kept_value=D(0), per_stock=D(1000), capital_cap=D(10000), top=10).ok
