"""API tests for the trading service against the real DB (testcontainers).
Requires Docker (via db_session -> postgres_url)."""

import uuid
from datetime import date, timedelta
from decimal import Decimal

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.core.auth import DEFAULT_USER_ID
from app.core.config import get_settings
from app.core.security import create_access_token
from app.domain.entities import BhavcopyRecord, IntradaySignalSnapshot, StockMasterRecord
from app.infrastructure.db.session import get_db_session
from app.repositories.historical_price_repository import SqlAlchemyHistoricalPriceRepository
from app.repositories.intraday_signal_snapshot_repository import SqlAlchemyIntradaySignalSnapshotRepository
from app.repositories.stock_repository import SqlAlchemyStockRepository
from services.trading_service.api.app import create_trading_app

EMA_RULE = {"op": "GT", "left": {"feature": "close"}, "right": {"feature": "SMA_20"}}


@pytest_asyncio.fixture
async def trading_client(db_session):
    app = create_trading_app()

    async def _get_db():
        yield db_session

    app.dependency_overrides[get_db_session] = _get_db
    token = create_access_token(DEFAULT_USER_ID, get_settings())
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", headers={"Authorization": f"Bearer {token}"}
    ) as client:
        yield client
    app.dependency_overrides.clear()


def _weekdays(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


async def _seed_history(db_session, symbol: str, n: int = 45) -> None:
    await SqlAlchemyStockRepository(db_session).upsert_universe(
        [
            StockMasterRecord(
                symbol=symbol, isin=None, name=f"{symbol} Ltd", series="EQ", listing_date=None, face_value=None
            )
        ]
    )
    await db_session.commit()
    days = _weekdays(date(2025, 1, 6), n)
    records = []
    for i, d in enumerate(days):
        c = Decimal(100 + i)  # steady uptrend so close > SMA_20 after warmup
        records.append(
            BhavcopyRecord(symbol=symbol, trade_date=d, open=c, high=c + 1, low=c - 1, close=c, volume=100000)
        )
    await SqlAlchemyHistoricalPriceRepository(db_session).bulk_upsert_bars(records)
    await db_session.commit()


async def test_auth_required(db_session):
    app = create_trading_app()

    async def _get_db():
        yield db_session

    app.dependency_overrides[get_db_session] = _get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/trading/accounts")
    assert resp.status_code in (401, 403)
    app.dependency_overrides.clear()


async def test_create_and_list_account(trading_client):
    resp = await trading_client.post("/trading/accounts", json={"mode": "PAPER", "starting_balance": "500000"})
    assert resp.status_code == 201
    account = resp.json()
    assert account["mode"] == "PAPER"
    assert Decimal(account["virtual_balance"]) == Decimal("500000")

    listed = await trading_client.get("/trading/accounts")
    assert listed.status_code == 200
    assert any(a["id"] == account["id"] for a in listed.json())


async def test_risk_profile_and_kill_switch(trading_client):
    account_id = (await trading_client.post("/trading/accounts", json={})).json()["id"]

    put = await trading_client.put(
        f"/trading/accounts/{account_id}/risk", json={"max_daily_loss": "5000", "per_trade_risk_pct": "1.5"}
    )
    assert put.status_code == 200
    assert Decimal(put.json()["max_daily_loss"]) == Decimal("5000")

    ks = await trading_client.post(f"/trading/accounts/{account_id}/kill-switch", json={"on": True})
    assert ks.status_code == 200
    assert ks.json()["kill_switch"] is True


async def test_account_ownership_404_for_unknown(trading_client):
    resp = await trading_client.get(f"/trading/accounts/{uuid.uuid4()}")
    assert resp.status_code == 404


async def test_strategy_crud_and_rule_validation(trading_client):
    created = await trading_client.post(
        "/trading/strategies",
        json={
            "name": "EMA breakout",
            "rule_tree": EMA_RULE,
            "side": "BUY",
            "product": "CNC",
            "quantity": 10,
            "target_pct": "2",
        },
    )
    assert created.status_code == 201
    sid = created.json()["id"]

    bad = await trading_client.post("/trading/strategies", json={"name": "bad", "rule_tree": {"op": "NONSENSE"}})
    assert bad.status_code == 422

    patched = await trading_client.patch(f"/trading/strategies/{sid}", json={"status": "VALIDATED"})
    assert patched.status_code == 200
    assert patched.json()["status"] == "VALIDATED"

    assert (await trading_client.delete(f"/trading/strategies/{sid}")).status_code == 204
    assert (await trading_client.get(f"/trading/strategies/{sid}")).status_code == 404


async def test_backtest_insufficient_data_returns_400(trading_client):
    resp = await trading_client.post("/trading/backtest", json={"symbol": "NOSUCH", "rule_tree": EMA_RULE})
    assert resp.status_code == 400


async def test_backtest_happy_path_runs_over_seeded_history(trading_client, db_session):
    await _seed_history(db_session, "TESTCO", n=45)
    resp = await trading_client.post(
        "/trading/backtest",
        json={
            "symbol": "TESTCO",
            "rule_tree": EMA_RULE,
            "product": "CNC",
            "quantity": 10,
            "target_pct": "1",
            "stop_loss_pct": "5",
            "starting_cash": "1000000",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["symbol"] == "TESTCO"
    assert body["bars"] == 45
    assert body["equity_points"] >= 1
    assert "win_rate" in body["metrics"]
    assert "sharpe" in body["metrics"]
    assert body["metrics"]["total_trades"] >= 1  # uptrend + 1% target -> at least one closed trade


async def test_paper_run_persists_journal_and_updates_balance(trading_client, db_session):
    await _seed_history(db_session, "PAPERCO", n=45)
    account_id = (await trading_client.post("/trading/accounts", json={"starting_balance": "1000000"})).json()["id"]
    strat = (
        await trading_client.post(
            "/trading/strategies",
            json={
                "name": "paper strat",
                "rule_tree": EMA_RULE,
                "side": "BUY",
                "product": "CNC",
                "quantity": 10,
                "target_pct": "1",
                "stop_loss_pct": "5",
            },
        )
    ).json()

    run = await trading_client.post(
        f"/trading/accounts/{account_id}/paper-run", json={"strategy_id": strat["id"], "symbol": "PAPERCO"}
    )
    assert run.status_code == 200, run.text
    body = run.json()
    assert body["trades"] >= 1
    assert body["bars"] >= 1

    # Trade journal persisted and readable
    trades = (await trading_client.get(f"/trading/accounts/{account_id}/trades")).json()
    assert len(trades) == body["trades"]
    assert all("pnl_net" in t for t in trades)

    # Virtual balance updated away from the starting 1,000,000
    account = (await trading_client.get(f"/trading/accounts/{account_id}")).json()
    assert Decimal(account["virtual_balance"]) != Decimal("1000000.00")

    # Equity curve persisted and returned in time order
    equity = (await trading_client.get(f"/trading/accounts/{account_id}/equity")).json()
    assert len(equity) == body["bars"]
    assert all("equity" in point and "ts" in point for point in equity)


async def test_paper_run_unknown_strategy_404(trading_client, db_session):
    await _seed_history(db_session, "PAPERCO2", n=40)
    account_id = (await trading_client.post("/trading/accounts", json={})).json()["id"]
    resp = await trading_client.post(
        f"/trading/accounts/{account_id}/paper-run", json={"strategy_id": str(uuid.uuid4()), "symbol": "PAPERCO2"}
    )
    assert resp.status_code == 404


async def test_broker_connect_encrypts_and_status_reflects_not_connected(trading_client, db_session):
    resp = await trading_client.post(
        "/trading/broker/connect", json={"broker": "zerodha", "api_key": "myapikey", "api_secret": "mysecret"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "api_key=myapikey" in body["login_url"]
    assert body["status"] == "DISCONNECTED"

    # credentials are stored ENCRYPTED, not in plaintext
    from app.core.config import get_settings
    from services.trading_service.persistence.repositories import BrokerSessionRepository
    from services.trading_service.security import CredentialCipher

    model = await BrokerSessionRepository(db_session).get(DEFAULT_USER_ID, "zerodha")
    assert bytes(model.api_secret_enc) != b"mysecret"
    assert CredentialCipher(get_settings()).decrypt(model.api_secret_enc) == "mysecret"

    status = await trading_client.get("/trading/broker/status")
    assert status.status_code == 200
    assert status.json()["connected"] is False


async def test_broker_connect_rejects_unknown_broker(trading_client):
    resp = await trading_client.post(
        "/trading/broker/connect", json={"broker": "robinhood", "api_key": "k", "api_secret": "s"}
    )
    assert resp.status_code == 422  # schema pattern rejects non-zerodha


async def test_broker_complete_without_kiteconnect_returns_502(trading_client):
    await trading_client.post("/trading/broker/connect", json={"api_key": "k", "api_secret": "s"})
    resp = await trading_client.post("/trading/broker/complete-login", json={"request_token": "rt123"})
    # kiteconnect isn't installed in the test env -> adapter raises BrokerError -> 502
    assert resp.status_code == 502
    assert "kiteconnect" in resp.json()["detail"].lower()


async def _seed_signal(db_session, symbol, confidence=75, signal="BUY"):
    await SqlAlchemyIntradaySignalSnapshotRepository(db_session).bulk_upsert(
        [
            IntradaySignalSnapshot(
                symbol=symbol,
                name=f"{symbol} Ltd",
                as_of=date(2026, 7, 24),
                signal=signal,
                confidence=Decimal(str(confidence)),
                entry_price=Decimal("100"),
                target_price=Decimal("103"),
                stop_loss=Decimal("98"),
                reasoning=["momentum: close above EMA20"],
            )
        ]
    )
    await db_session.commit()


async def _seed_series(db_session, symbol, closes):
    await SqlAlchemyStockRepository(db_session).upsert_universe(
        [
            StockMasterRecord(
                symbol=symbol, isin=None, name=f"{symbol} Ltd", series="EQ", listing_date=None, face_value=None
            )
        ]
    )
    await db_session.commit()
    days = _weekdays(date(2026, 1, 5), len(closes))
    recs = [
        BhavcopyRecord(
            symbol=symbol,
            trade_date=d,
            open=Decimal(str(c)),
            high=Decimal(str(c)),
            low=Decimal(str(c)),
            close=Decimal(str(c)),
            volume=100000,
        )
        for d, c in zip(days, closes, strict=False)
    ]
    await SqlAlchemyHistoricalPriceRepository(db_session).bulk_upsert_bars(recs)
    await db_session.commit()


async def test_momentum_ranking_orders_by_return(trading_client, db_session):
    flat = [100.0] * 10
    await _seed_series(db_session, "HIGHMOM", flat + [100 + 50 * i / 29 for i in range(30)])  # +50%
    await _seed_series(db_session, "MIDMOM", [100.0] * 40)  # 0%
    await _seed_series(db_session, "LOWMOM", flat + [100 - 30 * i / 29 for i in range(30)])  # -30%

    resp = await trading_client.get("/trading/momentum/ranking?lookback=30&top=3")
    assert resp.status_code == 200, resp.text
    picks = resp.json()["picks"]
    assert picks[0]["symbol"] == "HIGHMOM"
    assert picks[-1]["symbol"] == "LOWMOM"
    assert picks[0]["trailing_return_pct"] > picks[1]["trailing_return_pct"]


async def test_momentum_rebalance_and_portfolio(trading_client, db_session):
    flat = [100.0] * 10
    await _seed_series(db_session, "HIGHMOM", flat + [100 + 50 * i / 29 for i in range(30)])
    await _seed_series(db_session, "MIDMOM", flat + [100 + 20 * i / 29 for i in range(30)])
    await _seed_series(db_session, "LOWMOM", flat + [100 - 30 * i / 29 for i in range(30)])
    account_id = (await trading_client.post("/trading/accounts", json={"starting_balance": "1000000"})).json()["id"]

    rb = await trading_client.post(f"/trading/accounts/{account_id}/momentum/rebalance?top=2")
    assert rb.status_code == 200, rb.text
    body = rb.json()
    assert len(body["bought"]) == 2
    bought_syms = {b["symbol"] for b in body["bought"]}
    assert "HIGHMOM" in bought_syms and "MIDMOM" in bought_syms  # top 2, not LOWMOM
    assert 900000 < float(body["portfolio_value"]) <= 1000000  # ~fully invested

    pf = await trading_client.get(f"/trading/accounts/{account_id}/momentum/portfolio")
    assert pf.status_code == 200
    assert len(pf.json()["holdings"]) == 2

    # A second rebalance sells + rebuys without error.
    rb2 = await trading_client.post(f"/trading/accounts/{account_id}/momentum/rebalance?top=2")
    assert rb2.status_code == 200
    assert len(rb2.json()["sold"]) == 2


async def test_momentum_daily_report_notifies_user(db_session):
    from app.repositories.notification_repository import SqlAlchemyNotificationRepository
    from services.trading_service.momentum.report import generate_daily_reports
    from services.trading_service.persistence.repositories import PositionRepository, TradingAccountRepository

    await _seed_series(db_session, "MOMCO", [100.0] * 10 + [100 + 50 * i / 29 for i in range(30)])  # latest ~150
    account = await TradingAccountRepository(db_session).create(DEFAULT_USER_ID, "PAPER", Decimal("1000000"))
    await PositionRepository(db_session).upsert(account.id, "MOMCO", "CNC", 100, Decimal("140"), Decimal("0"))

    sent = await generate_daily_reports(db_session)
    assert sent >= 1  # at least our account (trading tables aren't truncated between tests)

    notifs = await SqlAlchemyNotificationRepository(db_session).list_for_user(
        DEFAULT_USER_ID, unread_only=False, limit=20, offset=0
    )
    assert any("Momentum portfolio" in n.title for n in notifs)


async def test_scanner_returns_buy_candidates(trading_client, db_session):
    await _seed_history(db_session, "SCANCO", n=40)  # creates the stock
    await _seed_signal(db_session, "SCANCO")
    resp = await trading_client.get("/trading/scanner?limit=10&min_confidence=60")
    assert resp.status_code == 200, resp.text
    symbols = [c["symbol"] for c in resp.json()["candidates"]]
    assert "SCANCO" in symbols


async def test_autopilot_run_and_report(trading_client, db_session):
    await _seed_history(db_session, "SCANCO", n=60)
    await _seed_signal(db_session, "SCANCO")
    account_id = (await trading_client.post("/trading/accounts", json={"starting_balance": "1000000"})).json()["id"]

    run = await trading_client.post(f"/trading/accounts/{account_id}/autopilot/run?top_k=5&lookback_days=1000")
    assert run.status_code == 200, run.text
    body = run.json()
    assert "summary" in body
    assert any(c["symbol"] == "SCANCO" for c in body["candidates"])

    report = await trading_client.get(f"/trading/accounts/{account_id}/report")
    assert report.status_code == 200
    assert "summary" in report.json()
    assert "total_trades" in report.json()["summary"]
