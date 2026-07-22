import uuid

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities import Notification
from app.domain.ports import NotificationRepositoryPort
from app.models.notification import NotificationModel


def _to_entity(row: NotificationModel) -> Notification:
    return Notification(
        id=row.id,
        user_id=row.user_id,
        alert_id=row.alert_id,
        title=row.title,
        message=row.message,
        created_at=row.created_at,
        read_at=row.read_at,
    )


class SqlAlchemyNotificationRepository(NotificationRepositoryPort):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, user_id: uuid.UUID, alert_id: uuid.UUID | None, title: str, message: str) -> Notification:
        model = NotificationModel(user_id=user_id, alert_id=alert_id, title=title, message=message)
        self._session.add(model)
        await self._session.commit()
        await self._session.refresh(model)
        return _to_entity(model)

    async def list_for_user(self, user_id: uuid.UUID, unread_only: bool, limit: int, offset: int) -> list[Notification]:
        stmt = select(NotificationModel).where(NotificationModel.user_id == user_id)
        if unread_only:
            stmt = stmt.where(NotificationModel.read_at.is_(None))
        stmt = stmt.order_by(NotificationModel.created_at.desc()).offset(offset).limit(limit)
        result = await self._session.execute(stmt)
        return [_to_entity(row) for row in result.scalars().all()]

    async def mark_read(self, notification_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        stmt = (
            update(NotificationModel)
            .where(
                NotificationModel.id == notification_id,
                NotificationModel.user_id == user_id,
                NotificationModel.read_at.is_(None),
            )
            .values(read_at=func.now())
        )
        result = await self._session.execute(stmt)
        if result.rowcount == 0:
            exists_stmt = select(NotificationModel.id).where(
                NotificationModel.id == notification_id, NotificationModel.user_id == user_id
            )
            already_read = (await self._session.execute(exists_stmt)).scalar_one_or_none() is not None
            await self._session.commit()
            return already_read
        await self._session.commit()
        return True
