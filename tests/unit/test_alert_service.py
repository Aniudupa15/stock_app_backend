import uuid
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.core.exceptions import AlertNotFoundError, StockNotFoundError
from app.domain.entities import AlertStatus, AlertType
from app.schemas.alert import AlertCreate
from app.services.alert_service import AlertService
from tests.conftest import FakeAlertRepository

USER_ID = uuid.uuid4()


async def test_create_alert_unknown_symbol_raises():
    repo = FakeAlertRepository(known_symbols=set())
    service = AlertService(repo)

    with pytest.raises(StockNotFoundError):
        await service.create(
            USER_ID, AlertCreate(symbol="DOESNOTEXIST", alert_type=AlertType.PRICE_ABOVE, condition={"price": "100"})
        )


async def test_create_and_list_alerts():
    repo = FakeAlertRepository(known_symbols={"RELIANCE"})
    service = AlertService(repo)

    created = await service.create(
        USER_ID, AlertCreate(symbol="reliance", alert_type=AlertType.PRICE_ABOVE, condition={"price": "1500"})
    )

    assert created.symbol == "RELIANCE"
    assert created.status == AlertStatus.ACTIVE
    assert created.condition["price"] == Decimal("1500")

    listed = await service.list(USER_ID, None)
    assert len(listed) == 1
    assert listed[0].id == created.id


async def test_list_filters_by_status():
    repo = FakeAlertRepository(known_symbols={"RELIANCE"})
    service = AlertService(repo)
    await service.create(
        USER_ID, AlertCreate(symbol="RELIANCE", alert_type=AlertType.PRICE_ABOVE, condition={"price": "1500"})
    )

    active = await service.list(USER_ID, AlertStatus.ACTIVE)
    triggered = await service.list(USER_ID, AlertStatus.TRIGGERED)

    assert len(active) == 1
    assert triggered == []


async def test_delete_unknown_alert_raises():
    repo = FakeAlertRepository()
    service = AlertService(repo)

    with pytest.raises(AlertNotFoundError):
        await service.delete(USER_ID, uuid.uuid4())


async def test_delete_alert_succeeds():
    repo = FakeAlertRepository(known_symbols={"RELIANCE"})
    service = AlertService(repo)
    created = await service.create(
        USER_ID, AlertCreate(symbol="RELIANCE", alert_type=AlertType.PRICE_ABOVE, condition={"price": "1500"})
    )

    await service.delete(USER_ID, created.id)

    assert await service.list(USER_ID, None) == []


def test_condition_missing_required_key_is_rejected():
    with pytest.raises(ValidationError):
        AlertCreate(symbol="RELIANCE", alert_type=AlertType.PRICE_ABOVE, condition={})


def test_condition_not_required_for_52_week_alert_types():
    alert = AlertCreate(symbol="RELIANCE", alert_type=AlertType.NEW_52_WEEK_HIGH, condition={})
    assert alert.condition == {}
