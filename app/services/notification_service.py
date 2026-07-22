import uuid

from app.core.exceptions import NotificationNotFoundError
from app.domain.ports import NotificationRepositoryPort
from app.schemas.notification import NotificationOut


class NotificationService:
    def __init__(self, repository: NotificationRepositoryPort):
        self._repository = repository

    async def list(self, user_id: uuid.UUID, unread_only: bool, limit: int, offset: int) -> list[NotificationOut]:
        notifications = await self._repository.list_for_user(user_id, unread_only, limit, offset)
        return [
            NotificationOut(
                id=n.id,
                alert_id=n.alert_id,
                title=n.title,
                message=n.message,
                created_at=n.created_at,
                read_at=n.read_at,
            )
            for n in notifications
        ]

    async def mark_read(self, user_id: uuid.UUID, notification_id: uuid.UUID) -> None:
        marked = await self._repository.mark_read(notification_id, user_id)
        if not marked:
            raise NotificationNotFoundError(notification_id)
