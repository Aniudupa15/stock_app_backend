"""Integration tests for the trading-schema repositories against a real
Postgres (testcontainers). Exercises the core persistence round-trips the
paper loop + API depend on. Requires Docker."""

import uuid
from datetime import UTC, datetime
from decimal import Decimal

from app.core.auth import DEFAULT_USER_ID
from libs.trading_domain.entities import OrderIntent, Signal, Trade
from libs.trading_domain.enums import ExitReason, OrderType, Product, Side
from services.trading_service.persistence.repositories import (
    AuditLogRepository,
    BacktestRepository,
    BrokerSessionRepository,
    EquitySnapshotRepository,
    FillRepository,
    OrderIntentRepository,
    OrderRepository,
    PositionRepository,
    RiskProfileRepository,
    SignalRepository,
    StrategyRepository,
    TradeRepository,
    TradingAccountRepository,
)
from tests.conftest import requires_docker

pytestmark = requires_docker


async def _account(db_session):
    return await TradingAccountRepository(db_session).create(DEFAULT_USER_ID, "PAPER", Decimal("1000000"))


async def test_account_create_and_read(db_session):
    account = await _account(db_session)
    assert account.mode == "PAPER"
    fetched = await TradingAccountRepository(db_session).get(account.id)
    assert fetched.virtual_balance == Decimal("1000000.00")
    await TradingAccountRepository(db_session).update_balance(account.id, Decimal("987654.32"))
    assert (await TradingAccountRepository(db_session).get(account.id)).virtual_balance == Decimal("987654.32")


async def test_risk_profile_upsert_is_idempotent_per_account(db_session):
    account = await _account(db_session)
    repo = RiskProfileRepository(db_session)
    await repo.upsert(account.id, max_daily_loss=Decimal("5000"), per_trade_risk_pct=Decimal("1.5"))
    await repo.upsert(account.id, max_daily_loss=Decimal("8000"))  # update, not duplicate
    profile = await repo.get(account.id)
    assert profile.max_daily_loss == Decimal("8000.00")
    await repo.set_kill_switch(account.id, True)
    assert (await repo.get(account.id)).kill_switch is True


async def test_strategy_crud(db_session):
    repo = StrategyRepository(db_session)
    rule = {"op": "GT", "left": {"feature": "close"}, "right": {"feature": "EMA_20"}}
    strat = await repo.create(
        DEFAULT_USER_ID, name="EMA breakout", rule_tree=rule, side="BUY", product="MIS", quantity=10
    )
    assert (await repo.get(strat.id, DEFAULT_USER_ID)).name == "EMA breakout"
    await repo.update(strat.id, DEFAULT_USER_ID, status="VALIDATED")
    assert (await repo.get(strat.id, DEFAULT_USER_ID)).status == "VALIDATED"
    assert len(await repo.list_for_user(DEFAULT_USER_ID)) >= 1
    assert await repo.delete(strat.id, DEFAULT_USER_ID) is True
    assert await repo.get(strat.id, DEFAULT_USER_ID) is None


async def test_broker_session_upsert_and_token(db_session):
    repo = BrokerSessionRepository(db_session)
    s1 = await repo.upsert_credentials(DEFAULT_USER_ID, "zerodha", b"enc-key", b"enc-secret")
    s2 = await repo.upsert_credentials(DEFAULT_USER_ID, "zerodha", b"enc-key-2", b"enc-secret-2")  # update same row
    assert s1.id == s2.id
    expires = datetime(2026, 1, 7, 6, 0, tzinfo=UTC)
    await repo.set_access_token(s1.id, b"enc-token", expires, status="CONNECTED")
    fetched = await repo.get(DEFAULT_USER_ID, "zerodha")
    assert fetched.status == "CONNECTED"
    assert bytes(fetched.access_token_enc) == b"enc-token"


async def test_order_intent_order_fill_flow(db_session):
    account = await _account(db_session)
    intent = OrderIntent(uuid.uuid4(), account.id, "INFY", Side.BUY, OrderType.MARKET, Product.MIS, 10)
    intent_id = await OrderIntentRepository(db_session).save(intent, account.id, risk_verdict={"decision": "ALLOW"})
    assert intent_id == intent.intent_id

    order_repo = OrderRepository(db_session)
    order = await order_repo.create(intent_id, account.id, "PAPER", "PAPER-000001", "SUBMITTED", leg="ENTRY")
    await order_repo.update_from_event("PAPER", "PAPER-000001", "COMPLETE", 10, 0, Decimal("1500.25"))
    orders = await order_repo.list_for_account(account.id)
    assert orders[0].state == "COMPLETE"
    assert orders[0].average_price == Decimal("1500.2500")

    await FillRepository(db_session).add(order.id, "INFY", "BUY", 10, Decimal("1500.25"), {"total": "4.21"})


async def test_trade_persist_and_read_back_as_domain(db_session):
    account = await _account(db_session)
    trade = Trade(
        account_id=account.id,
        symbol="INFY",
        qty=10,
        entry_price=Decimal("1000"),
        exit_price=Decimal("1020"),
        pnl_gross=Decimal("200"),
        charges_total=Decimal("12.50"),
        pnl_net=Decimal("187.50"),
        entry_ts=datetime(2026, 1, 5, 10, 0, tzinfo=UTC),
        exit_ts=datetime(2026, 1, 5, 14, 0, tzinfo=UTC),
        exit_reason=ExitReason.TARGET,
    )
    repo = TradeRepository(db_session)
    await repo.add(trade)
    trades = await repo.list_for_account(account.id)
    assert len(trades) == 1
    assert trades[0].pnl_net == Decimal("187.50")
    assert trades[0].exit_reason is ExitReason.TARGET


async def test_position_upsert_and_equity_curve(db_session):
    account = await _account(db_session)
    pos_repo = PositionRepository(db_session)
    await pos_repo.upsert(account.id, "INFY", "MIS", 10, Decimal("1500"), Decimal("0"))
    await pos_repo.upsert(account.id, "INFY", "MIS", 0, Decimal("0"), Decimal("200"))  # closed, realized 200
    positions = await pos_repo.list_for_account(account.id)
    assert positions[0].realized_pnl == Decimal("200.00")
    assert positions[0].product is Product.MIS

    eq_repo = EquitySnapshotRepository(db_session)
    await eq_repo.append(account.id, Decimal("1000000"), Decimal("1000000"), ts=datetime(2026, 1, 5, 10, 0, tzinfo=UTC))
    await eq_repo.append(account.id, Decimal("1000200"), Decimal("1000200"), ts=datetime(2026, 1, 5, 15, 0, tzinfo=UTC))
    curve = await eq_repo.curve(account.id)
    assert [c.equity for c in curve] == [Decimal("1000000.00"), Decimal("1000200.00")]


async def test_backtest_signal_and_audit(db_session):
    account = await _account(db_session)
    bt_id = await BacktestRepository(db_session).save(
        symbol="INFY",
        starting_cash=Decimal("1000000"),
        final_equity=Decimal("1050000"),
        metrics={"win_rate": 0.6, "sharpe": 1.2},
    )
    assert bt_id is not None
    assert len(await BacktestRepository(db_session).list_recent()) >= 1

    signal = Signal(
        None, "INFY", Side.BUY, Decimal("1000"), Decimal("980"), [Decimal("1020")], Decimal("75"), "EMA cross"
    )
    await SignalRepository(db_session).add(signal, account.id)
    assert len(await SignalRepository(db_session).list_for_account(account.id)) == 1

    audit = AuditLogRepository(db_session)
    await audit.append("system", "ORDER_PLACED", account_id=account.id, payload={"symbol": "INFY"})
    logs = await audit.list_for_account(account.id)
    assert logs[0].event_type == "ORDER_PLACED"
