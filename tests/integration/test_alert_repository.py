import uuid
from datetime import UTC, datetime

from app.core.auth import DEFAULT_USER_ID
from app.domain.entities import AlertStatus, AlertType, StockMasterRecord
from app.repositories.alert_repository import SqlAlchemyAlertRepository
from app.repositories.stock_repository import SqlAlchemyStockRepository
from tests.conftest import requires_docker

pytestmark = requires_docker


async def _seed_stock(db_session, symbol: str) -> None:
    stock_repo = SqlAlchemyStockRepository(db_session)
    await stock_repo.upsert_universe(
        [
            StockMasterRecord(
                symbol=symbol, isin=None, name=f"{symbol} Ltd", series="EQ", listing_date=None, face_value=None
            )
        ]
    )
    await db_session.commit()


async def test_create_rejects_unknown_symbol(db_session):
    repo = SqlAlchemyAlertRepository(db_session)

    alert = await repo.create(DEFAULT_USER_ID, "DOESNOTEXIST", AlertType.PRICE_ABOVE, {"price": "100"})

    assert alert is None


async def test_create_get_and_list(db_session):
    await _seed_stock(db_session, "RELIANCE")
    repo = SqlAlchemyAlertRepository(db_session)

    created = await repo.create(DEFAULT_USER_ID, "RELIANCE", AlertType.PRICE_ABOVE, {"price": "1500"})
    assert created is not None
    assert created.symbol == "RELIANCE"
    assert created.status == AlertStatus.ACTIVE

    fetched = await repo.get(created.id, DEFAULT_USER_ID)
    assert fetched is not None
    assert fetched.condition == {"price": "1500"}

    listed = await repo.list_for_user(DEFAULT_USER_ID, None)
    assert [a.id for a in listed] == [created.id]


async def test_list_active_returns_only_active_alerts(db_session):
    await _seed_stock(db_session, "TCS")
    repo = SqlAlchemyAlertRepository(db_session)
    created = await repo.create(DEFAULT_USER_ID, "TCS", AlertType.PRICE_ABOVE, {"price": "100"})

    before = await repo.list_active()
    await repo.mark_triggered(created.id, datetime.now(UTC))
    after = await repo.list_active()

    assert len(before) == 1
    assert after == []


async def test_delete_alert(db_session):
    await _seed_stock(db_session, "INFY")
    repo = SqlAlchemyAlertRepository(db_session)
    created = await repo.create(DEFAULT_USER_ID, "INFY", AlertType.PRICE_ABOVE, {"price": "100"})

    deleted = await repo.delete(created.id, DEFAULT_USER_ID)
    deleted_again = await repo.delete(created.id, DEFAULT_USER_ID)

    assert deleted is True
    assert deleted_again is False
    assert await repo.get(created.id, DEFAULT_USER_ID) is None


async def test_get_returns_none_for_wrong_user(db_session):
    await _seed_stock(db_session, "HDFCBANK")
    repo = SqlAlchemyAlertRepository(db_session)
    created = await repo.create(DEFAULT_USER_ID, "HDFCBANK", AlertType.PRICE_ABOVE, {"price": "100"})

    result = await repo.get(created.id, uuid.uuid4())

    assert result is None
