import uuid

import pytest

from app.core.exceptions import NotificationNotFoundError
from app.services.notification_service import NotificationService
from tests.conftest import FakeNotificationRepository

USER_ID = uuid.uuid4()


async def test_list_returns_notifications_for_user():
    repo = FakeNotificationRepository()
    await repo.create(USER_ID, None, "Title", "Message")
    service = NotificationService(repo)

    result = await service.list(USER_ID, unread_only=False, limit=10, offset=0)

    assert len(result) == 1
    assert result[0].title == "Title"


async def test_list_unread_only_filters_read_notifications():
    repo = FakeNotificationRepository()
    n1 = await repo.create(USER_ID, None, "Unread", "Message")
    n2 = await repo.create(USER_ID, None, "Read", "Message")
    await repo.mark_read(n2.id, USER_ID)
    service = NotificationService(repo)

    result = await service.list(USER_ID, unread_only=True, limit=10, offset=0)

    assert len(result) == 1
    assert result[0].id == n1.id


async def test_mark_read_unknown_notification_raises():
    repo = FakeNotificationRepository()
    service = NotificationService(repo)

    with pytest.raises(NotificationNotFoundError):
        await service.mark_read(USER_ID, uuid.uuid4())


async def test_mark_read_succeeds():
    repo = FakeNotificationRepository()
    n = await repo.create(USER_ID, None, "Title", "Message")
    service = NotificationService(repo)

    await service.mark_read(USER_ID, n.id)

    result = await service.list(USER_ID, unread_only=True, limit=10, offset=0)
    assert result == []
