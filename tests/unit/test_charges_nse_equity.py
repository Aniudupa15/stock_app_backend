"""Unit tests for libs/charges (NSE equity cost calculator).

Expected totals are hand-computed from the verified 2026 Zerodha schedule so a
future rate drift or logic regression fails loudly. Pure - no DB, no Docker.
"""

from decimal import Decimal

import pytest

from libs.charges.models import Charges, Product, Side
from libs.charges.nse_equity import ChargeSchedule, compute


def test_intraday_buy_small_turnover():
    # MIS BUY 100 @ 100 -> turnover 10,000
    # brokerage min(20, 0.03%*10000=3)=3 | stt 0 (buy) | exch 0.297->0.30
    # sebi 0.01 | stamp 0.003%*10000=0.30 | gst 18%*(3+0.297+0.01)=0.60 | dp 0
    c = compute(Side.BUY, Product.MIS, 100, Decimal("100"))
    assert c.brokerage == Decimal("3.00")
    assert c.stt == Decimal("0.00")
    assert c.exchange_txn == Decimal("0.30")
    assert c.sebi == Decimal("0.01")
    assert c.stamp_duty == Decimal("0.30")
    assert c.gst == Decimal("0.60")
    assert c.dp == Decimal("0.00")
    assert c.total == Decimal("4.21")


def test_intraday_sell_charges_stt_on_sell_only():
    # MIS SELL 100 @ 100 -> stt 0.025%*10000=2.50, no stamp (sell), no dp
    c = compute(Side.SELL, Product.MIS, 100, Decimal("100"))
    assert c.stt == Decimal("2.50")
    assert c.stamp_duty == Decimal("0.00")
    assert c.dp == Decimal("0.00")
    # brokerage 3, exch 0.30, sebi 0.01, gst 0.60, total = 3+2.5+0.30+0.01+0+0.60 = 6.41
    assert c.total == Decimal("6.41")


def test_intraday_brokerage_capped_at_20():
    # MIS BUY 1000 @ 500 -> turnover 500,000 -> 0.03% = 150 -> capped 20
    c = compute(Side.BUY, Product.MIS, 1000, Decimal("500"))
    assert c.brokerage == Decimal("20.00")


def test_delivery_buy():
    # CNC BUY 10 @ 2000 -> turnover 20,000
    # brokerage 0 | stt 0.10%*20000=20 | exch 0.594->0.59 | sebi 0.02
    # stamp 0.015%*20000=3.00 | gst 18%*(0+0.594+0.02)=0.11 | dp 0
    c = compute(Side.BUY, Product.CNC, 10, Decimal("2000"))
    assert c.brokerage == Decimal("0.00")
    assert c.stt == Decimal("20.00")
    assert c.stamp_duty == Decimal("3.00")
    assert c.dp == Decimal("0.00")
    assert c.total == Decimal("23.72")


def test_delivery_sell_includes_dp_and_no_stamp():
    # CNC SELL 10 @ 2000 -> stt 20, exch 0.59, sebi 0.02, stamp 0 (sell)
    # gst 0.11 | dp 13.5*1.18=15.93 | total 20+0.59+0.02+0+0.11+15.93 = 36.65
    c = compute(Side.SELL, Product.CNC, 10, Decimal("2000"))
    assert c.stamp_duty == Decimal("0.00")
    assert c.dp == Decimal("15.93")
    assert c.total == Decimal("36.65")


def test_delivery_brokerage_is_zero_regardless_of_size():
    c = compute(Side.BUY, Product.CNC, 100000, Decimal("3000"))
    assert c.brokerage == Decimal("0.00")


def test_schedule_override_changes_result():
    # Override exchange txn to 0 -> that line and its GST contribution drop out
    sched = ChargeSchedule(exchange_txn_rate=Decimal("0"))
    c = compute(Side.BUY, Product.MIS, 100, Decimal("100"), schedule=sched)
    assert c.exchange_txn == Decimal("0.00")
    # gst now 18%*(brokerage 3 + 0 + sebi 0.01) = 0.5418 -> 0.54
    assert c.gst == Decimal("0.54")


def test_charges_as_dict_roundtrips_total():
    c = compute(Side.BUY, Product.MIS, 100, Decimal("100"))
    d = c.as_dict()
    assert d["total"] == str(c.total)
    assert set(d) == {"brokerage", "stt", "exchange_txn", "sebi", "stamp_duty", "gst", "dp", "total"}


@pytest.mark.parametrize("qty,price", [(0, Decimal("100")), (-1, Decimal("100")), (10, Decimal("0")), (10, Decimal("-5"))])
def test_rejects_non_positive_inputs(qty, price):
    with pytest.raises(ValueError):
        compute(Side.BUY, Product.MIS, qty, price)


def test_total_equals_sum_of_parts():
    c = compute(Side.SELL, Product.CNC, 37, Decimal("1234.55"))
    parts = c.brokerage + c.stt + c.exchange_txn + c.sebi + c.stamp_duty + c.gst + c.dp
    assert c.total == parts.quantize(Decimal("0.01"))
    assert isinstance(c, Charges)
