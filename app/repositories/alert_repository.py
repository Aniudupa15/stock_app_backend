import uuid
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities import Alert, AlertStatus, AlertType
from app.domain.ports import AlertRepositoryPort
from app.models.alert import AlertModel
from app.models.stock import StockModel


def _to_entity(row: AlertModel, symbol: str) -> Alert:
    return Alert(
        id=row.id,
        user_id=row.user_id,
        symbol=symbol,
        alert_type=row.alert_type,
        condition=row.condition,
        status=row.status,
        created_at=row.created_at,
        triggered_at=row.triggered_at,
    )


class SqlAlchemyAlertRepository(AlertRepositoryPort):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, user_id: uuid.UUID, symbol: str, alert_type: AlertType, condition: dict) -> Alert | None:
        stock_stmt = select(StockModel.id).where(StockModel.symbol == symbol.strip().upper())
        stock_id = (await self._session.execute(stock_stmt)).scalar_one_or_none()
        if stock_id is None:
            return None

        model = AlertModel(user_id=user_id, stock_id=stock_id, alert_type=alert_type, condition=condition)
        self._session.add(model)
        await self._session.commit()
        await self._session.refresh(model)
        return _to_entity(model, symbol.strip().upper())

    async def list_for_user(self, user_id: uuid.UUID, status: AlertStatus | None) -> list[Alert]:
        stmt = (
            select(AlertModel, StockModel.symbol)
            .join(StockModel, StockModel.id == AlertModel.stock_id)
            .where(AlertModel.user_id == user_id)
            .order_by(AlertModel.created_at.desc())
        )
        if status is not None:
            stmt = stmt.where(AlertModel.status == status)
        result = await self._session.execute(stmt)
        return [_to_entity(row, symbol) for row, symbol in result]

    async def get(self, alert_id: uuid.UUID, user_id: uuid.UUID) -> Alert | None:
        stmt = (
            select(AlertModel, StockModel.symbol)
            .join(StockModel, StockModel.id == AlertModel.stock_id)
            .where(AlertModel.id == alert_id, AlertModel.user_id == user_id)
        )
        result = await self._session.execute(stmt)
        row = result.first()
        if row is None:
            return None
        return _to_entity(row[0], row[1])

    async def delete(self, alert_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        stmt = select(AlertModel).where(AlertModel.id == alert_id, AlertModel.user_id == user_id)
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        if model is None:
            return False
        await self._session.delete(model)
        await self._session.commit()
        return True

    async def list_active(self) -> list[Alert]:
        stmt = (
            select(AlertModel, StockModel.symbol)
            .join(StockModel, StockModel.id == AlertModel.stock_id)
            .where(AlertModel.status == AlertStatus.ACTIVE)
        )
        result = await self._session.execute(stmt)
        return [_to_entity(row, symbol) for row, symbol in result]

    async def mark_triggered(self, alert_id: uuid.UUID, triggered_at: datetime) -> None:
        stmt = (
            update(AlertModel)
            .where(AlertModel.id == alert_id)
            .values(status=AlertStatus.TRIGGERED, triggered_at=triggered_at)
        )
        await self._session.execute(stmt)
        await self._session.commit()
