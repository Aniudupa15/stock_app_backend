import uuid
from datetime import date, timedelta
from decimal import Decimal

from app.domain.entities import AlertStatus, AlertType, OhlcvBar
from app.services.alert_evaluation_service import AlertEvaluationService
from tests.conftest import FakeAlertRepository, FakeHistoricalPriceRepository, FakeNotificationRepository

USER_ID = uuid.uuid4()


def _bars(closes: list[float], volumes: list[int] | None = None) -> list[OhlcvBar]:
    volumes = volumes or [1000] * len(closes)
    today = date.today()
    n = len(closes)
    return [
        OhlcvBar(
            trade_date=today - timedelta(days=(n - 1 - i)),
            open=Decimal(str(c)),
            high=Decimal(str(c)),
            low=Decimal(str(c)),
            close=Decimal(str(c)),
            volume=v,
        )
        for i, (c, v) in enumerate(zip(closes, volumes, strict=False))
    ]


async def test_price_above_alert_triggers_and_creates_notification():
    alert_repo = FakeAlertRepository(known_symbols={"RELIANCE"})
    notif_repo = FakeNotificationRepository()
    price_repo = FakeHistoricalPriceRepository({"RELIANCE": _bars([100, 105, 110])})
    alert = await alert_repo.create(USER_ID, "RELIANCE", AlertType.PRICE_ABOVE, {"price": "108"})

    service = AlertEvaluationService(alert_repo, notif_repo, price_repo)
    triggered = await service.evaluate_all()

    assert triggered == 1
    updated = await alert_repo.get(alert.id, USER_ID)
    assert updated.status == AlertStatus.TRIGGERED
    assert updated.triggered_at is not None

    notifications = await notif_repo.list_for_user(USER_ID, False, 10, 0)
    assert len(notifications) == 1
    assert "RELIANCE" in notifications[0].message


async def test_price_above_alert_does_not_trigger_below_threshold():
    alert_repo = FakeAlertRepository(known_symbols={"RELIANCE"})
    notif_repo = FakeNotificationRepository()
    price_repo = FakeHistoricalPriceRepository({"RELIANCE": _bars([100, 105, 110])})
    await alert_repo.create(USER_ID, "RELIANCE", AlertType.PRICE_ABOVE, {"price": "200"})

    service = AlertEvaluationService(alert_repo, notif_repo, price_repo)
    triggered = await service.evaluate_all()

    assert triggered == 0
    assert (await alert_repo.list_active())[0].status == AlertStatus.ACTIVE


async def test_percent_change_above_uses_day_over_day_move():
    alert_repo = FakeAlertRepository(known_symbols={"RELIANCE"})
    notif_repo = FakeNotificationRepository()
    price_repo = FakeHistoricalPriceRepository({"RELIANCE": _bars([100, 110])})  # +10%
    await alert_repo.create(USER_ID, "RELIANCE", AlertType.PERCENT_CHANGE_ABOVE, {"percent": "5"})

    service = AlertEvaluationService(alert_repo, notif_repo, price_repo)
    triggered = await service.evaluate_all()

    assert triggered == 1


async def test_rsi_above_triggers_on_strictly_rising_prices():
    closes = [100 + i for i in range(20)]  # all up-days -> RSI near 100
    alert_repo = FakeAlertRepository(known_symbols={"RELIANCE"})
    notif_repo = FakeNotificationRepository()
    price_repo = FakeHistoricalPriceRepository({"RELIANCE": _bars(closes)})
    await alert_repo.create(USER_ID, "RELIANCE", AlertType.RSI_ABOVE, {"threshold": "70"})

    service = AlertEvaluationService(alert_repo, notif_repo, price_repo)
    triggered = await service.evaluate_all()

    assert triggered == 1


async def test_rsi_below_triggers_on_strictly_falling_prices():
    closes = [120 - i for i in range(20)]  # all down-days -> RSI near 0
    alert_repo = FakeAlertRepository(known_symbols={"RELIANCE"})
    notif_repo = FakeNotificationRepository()
    price_repo = FakeHistoricalPriceRepository({"RELIANCE": _bars(closes)})
    await alert_repo.create(USER_ID, "RELIANCE", AlertType.RSI_BELOW, {"threshold": "30"})

    service = AlertEvaluationService(alert_repo, notif_repo, price_repo)
    triggered = await service.evaluate_all()

    assert triggered == 1


async def test_volume_spike_triggers_when_latest_volume_exceeds_multiplier():
    closes = [100.0] * 21
    volumes = [1000] * 20 + [5000]
    alert_repo = FakeAlertRepository(known_symbols={"RELIANCE"})
    notif_repo = FakeNotificationRepository()
    price_repo = FakeHistoricalPriceRepository({"RELIANCE": _bars(closes, volumes)})
    await alert_repo.create(USER_ID, "RELIANCE", AlertType.VOLUME_SPIKE, {"multiplier": "3"})

    service = AlertEvaluationService(alert_repo, notif_repo, price_repo)
    triggered = await service.evaluate_all()

    assert triggered == 1


async def test_new_52_week_high_triggers_when_latest_close_is_period_max():
    closes = [90, 95, 100]
    alert_repo = FakeAlertRepository(known_symbols={"RELIANCE"})
    notif_repo = FakeNotificationRepository()
    price_repo = FakeHistoricalPriceRepository({"RELIANCE": _bars(closes)})
    await alert_repo.create(USER_ID, "RELIANCE", AlertType.NEW_52_WEEK_HIGH, {})

    service = AlertEvaluationService(alert_repo, notif_repo, price_repo)
    triggered = await service.evaluate_all()

    assert triggered == 1


async def test_alert_with_no_price_history_is_skipped_not_errored():
    alert_repo = FakeAlertRepository(known_symbols={"RELIANCE"})
    notif_repo = FakeNotificationRepository()
    price_repo = FakeHistoricalPriceRepository({})  # no bars at all
    await alert_repo.create(USER_ID, "RELIANCE", AlertType.PRICE_ABOVE, {"price": "100"})

    service = AlertEvaluationService(alert_repo, notif_repo, price_repo)
    triggered = await service.evaluate_all()

    assert triggered == 0


async def test_malformed_condition_is_skipped_not_errored():
    alert_repo = FakeAlertRepository(known_symbols={"RELIANCE"})
    notif_repo = FakeNotificationRepository()
    price_repo = FakeHistoricalPriceRepository({"RELIANCE": _bars([100, 105])})
    # Simulates legacy/corrupted data bypassing AlertCreate's validation.
    await alert_repo.create(USER_ID, "RELIANCE", AlertType.PRICE_ABOVE, {})

    service = AlertEvaluationService(alert_repo, notif_repo, price_repo)
    triggered = await service.evaluate_all()

    assert triggered == 0
    assert (await alert_repo.list_active())[0].status == AlertStatus.ACTIVE
