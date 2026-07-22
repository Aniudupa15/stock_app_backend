import uuid

from app.core.exceptions import AlertNotFoundError, StockNotFoundError
from app.domain.entities import Alert, AlertStatus
from app.domain.ports import AlertRepositoryPort
from app.schemas.alert import AlertCreate, AlertOut


def _to_schema(alert: Alert) -> AlertOut:
    return AlertOut(
        id=alert.id,
        symbol=alert.symbol,
        alert_type=alert.alert_type,
        condition=alert.condition,
        status=alert.status,
        created_at=alert.created_at,
        triggered_at=alert.triggered_at,
    )


class AlertService:
    def __init__(self, repository: AlertRepositoryPort):
        self._repository = repository

    async def create(self, user_id: uuid.UUID, body: AlertCreate) -> AlertOut:
        # JSONB storage needs JSON-safe values - Decimal isn't one, so store
        # the string form; AlertOut's `dict[str, Decimal]` type coerces it
        # back on the way out, round-tripping precision exactly.
        condition = {k: str(v) for k, v in body.condition.items()}
        alert = await self._repository.create(user_id, body.symbol.strip().upper(), body.alert_type, condition)
        if alert is None:
            raise StockNotFoundError(body.symbol)
        return _to_schema(alert)

    async def list(self, user_id: uuid.UUID, status: AlertStatus | None) -> list[AlertOut]:
        alerts = await self._repository.list_for_user(user_id, status)
        return [_to_schema(a) for a in alerts]

    async def delete(self, user_id: uuid.UUID, alert_id: uuid.UUID) -> None:
        deleted = await self._repository.delete(alert_id, user_id)
        if not deleted:
            raise AlertNotFoundError(alert_id)
