import uuid

from app.core.auth import DEFAULT_USER_ID
from app.repositories.notification_repository import SqlAlchemyNotificationRepository
from tests.conftest import requires_docker

pytestmark = requires_docker


async def test_create_and_list(db_session):
    repo = SqlAlchemyNotificationRepository(db_session)

    created = await repo.create(DEFAULT_USER_ID, None, "Title", "Message body")

    listed = await repo.list_for_user(DEFAULT_USER_ID, unread_only=False, limit=10, offset=0)
    assert [n.id for n in listed] == [created.id]
    assert listed[0].read_at is None


async def test_mark_read_is_idempotent(db_session):
    repo = SqlAlchemyNotificationRepository(db_session)
    created = await repo.create(DEFAULT_USER_ID, None, "Title", "Message body")

    first = await repo.mark_read(created.id, DEFAULT_USER_ID)
    second = await repo.mark_read(created.id, DEFAULT_USER_ID)

    assert first is True
    assert second is True

    unread = await repo.list_for_user(DEFAULT_USER_ID, unread_only=True, limit=10, offset=0)
    assert unread == []


async def test_mark_read_unknown_notification_returns_false(db_session):
    repo = SqlAlchemyNotificationRepository(db_session)

    result = await repo.mark_read(uuid.uuid4(), DEFAULT_USER_ID)

    assert result is False


async def test_unread_only_filters_correctly(db_session):
    repo = SqlAlchemyNotificationRepository(db_session)
    n1 = await repo.create(DEFAULT_USER_ID, None, "Unread", "Message")
    n2 = await repo.create(DEFAULT_USER_ID, None, "Read", "Message")
    await repo.mark_read(n2.id, DEFAULT_USER_ID)

    unread = await repo.list_for_user(DEFAULT_USER_ID, unread_only=True, limit=10, offset=0)

    assert [n.id for n in unread] == [n1.id]
