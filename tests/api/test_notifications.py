import uuid

from app.core.auth import DEFAULT_USER_ID
from app.repositories.notification_repository import SqlAlchemyNotificationRepository


async def test_list_and_mark_read(app_client, db_session):
    client, _ = app_client
    repo = SqlAlchemyNotificationRepository(db_session)
    created = await repo.create(DEFAULT_USER_ID, None, "Title", "Message body")

    list_resp = await client.get("/api/v1/notifications")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1

    mark_resp = await client.post(f"/api/v1/notifications/{created.id}/read")
    assert mark_resp.status_code == 204

    unread_resp = await client.get("/api/v1/notifications", params={"unread_only": True})
    assert unread_resp.json() == []


async def test_mark_read_unknown_notification_returns_404(app_client):
    client, _ = app_client
    resp = await client.post(f"/api/v1/notifications/{uuid.uuid4()}/read")
    assert resp.status_code == 404
