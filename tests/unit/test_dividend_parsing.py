from decimal import Decimal

import pytest

from app.services.dividend_parsing import sum_dividend_amount


@pytest.mark.parametrize(
    "purpose,expected",
    [
        ("Dividend - Rs 10 Per Share/Special Dividend - Rs 30 Per Share", Decimal("40")),
        ("Interim Dividend - Rs 4 Per Share", Decimal("4")),
        ("Dividend - Re 0.70 Per Share", Decimal("0.70")),
        ("Face Value Split (Sub-Division) - From Rs 5/- Per Share To Rs 2/- Per Share", Decimal("0")),
    ],
)
def test_sum_dividend_amount(purpose, expected):
    assert sum_dividend_amount(purpose) == expected
