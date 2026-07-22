from datetime import date

from app.finance.xirr import xirr


def test_simple_one_year_return_matches_known_rate():
    cash_flows = [(date(2025, 1, 1), -1000.0), (date(2026, 1, 1), 1100.0)]

    rate = xirr(cash_flows)

    assert rate is not None
    assert abs(rate - 0.10) < 1e-4


def test_multiple_investments_and_a_final_valuation():
    cash_flows = [
        (date(2025, 1, 1), -1000.0),
        (date(2025, 7, 1), -1000.0),
        (date(2026, 1, 1), 2300.0),
    ]

    rate = xirr(cash_flows)

    assert rate is not None
    assert rate > 0


def test_returns_none_for_fewer_than_two_cash_flows():
    assert xirr([(date(2025, 1, 1), -1000.0)]) is None
    assert xirr([]) is None


def test_returns_none_when_all_cash_flows_are_same_sign():
    cash_flows = [(date(2025, 1, 1), -1000.0), (date(2025, 6, 1), -500.0)]

    assert xirr(cash_flows) is None
