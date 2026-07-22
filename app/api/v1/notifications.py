import uuid

from fastapi import APIRouter, Depends, Query, status

from app.api.deps import get_current_user_id, get_notification_service
from app.schemas.notification import NotificationOut
from app.services.notification_service import NotificationService

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.get("", response_model=list[NotificationOut])
async def list_notifications(
    unread_only: bool = Query(False),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user_id: uuid.UUID = Depends(get_current_user_id),
    service: NotificationService = Depends(get_notification_service),
) -> list[NotificationOut]:
    return await service.list(user_id, unread_only, limit, offset)


@router.post("/{notification_id}/read", status_code=status.HTTP_204_NO_CONTENT)
async def mark_notification_read(
    notification_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user_id),
    service: NotificationService = Depends(get_notification_service),
) -> None:
    await service.mark_read(user_id, notification_id)
