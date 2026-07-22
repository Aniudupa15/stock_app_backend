from datetime import date
from decimal import Decimal

from app.domain.entities import PortfolioTransaction, TransactionType
from app.finance.holdings import compute_holdings


def _tx(symbol, txn_type, qty, price, d):
    return PortfolioTransaction(
        symbol=symbol, transaction_type=txn_type, quantity=Decimal(qty), price=Decimal(price), transaction_date=d
    )


def test_single_buy_produces_matching_avg_price():
    transactions = [_tx("RELIANCE", TransactionType.BUY, "10", "100", date(2026, 1, 1))]

    holdings = compute_holdings(transactions)

    assert len(holdings) == 1
    assert holdings[0].symbol == "RELIANCE"
    assert holdings[0].quantity == Decimal("10")
    assert holdings[0].avg_price == Decimal("100")
    assert holdings[0].cost_basis == Decimal("1000")


def test_multiple_buys_computes_weighted_average():
    transactions = [
        _tx("RELIANCE", TransactionType.BUY, "10", "100", date(2026, 1, 1)),
        _tx("RELIANCE", TransactionType.BUY, "10", "200", date(2026, 2, 1)),
    ]

    holdings = compute_holdings(transactions)

    assert holdings[0].quantity == Decimal("20")
    assert holdings[0].avg_price == Decimal("150")
    assert holdings[0].cost_basis == Decimal("3000")


def test_sell_reduces_cost_basis_proportionally_to_avg_price():
    transactions = [
        _tx("RELIANCE", TransactionType.BUY, "10", "100", date(2026, 1, 1)),
        _tx("RELIANCE", TransactionType.SELL, "4", "150", date(2026, 2, 1)),  # sale price irrelevant to cost basis
    ]

    holdings = compute_holdings(transactions)

    assert holdings[0].quantity == Decimal("6")
    assert holdings[0].avg_price == Decimal("100")
    assert holdings[0].cost_basis == Decimal("600")


def test_fully_sold_position_excluded_from_holdings():
    transactions = [
        _tx("RELIANCE", TransactionType.BUY, "10", "100", date(2026, 1, 1)),
        _tx("RELIANCE", TransactionType.SELL, "10", "150", date(2026, 2, 1)),
    ]

    holdings = compute_holdings(transactions)

    assert holdings == []


def test_transactions_are_processed_in_date_order_regardless_of_input_order():
    transactions = [
        _tx("RELIANCE", TransactionType.SELL, "4", "150", date(2026, 2, 1)),
        _tx("RELIANCE", TransactionType.BUY, "10", "100", date(2026, 1, 1)),
    ]

    holdings = compute_holdings(transactions)

    assert holdings[0].quantity == Decimal("6")


def test_multiple_symbols_are_independent():
    transactions = [
        _tx("RELIANCE", TransactionType.BUY, "10", "100", date(2026, 1, 1)),
        _tx("TCS", TransactionType.BUY, "5", "200", date(2026, 1, 1)),
    ]

    holdings = {h.symbol: h for h in compute_holdings(transactions)}

    assert holdings["RELIANCE"].quantity == Decimal("10")
    assert holdings["TCS"].quantity == Decimal("5")
