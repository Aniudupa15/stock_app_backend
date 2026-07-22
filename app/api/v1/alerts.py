import uuid

from fastapi import APIRouter, Depends, Query, status

from app.api.deps import get_alert_service, get_current_user_id
from app.domain.entities import AlertStatus
from app.schemas.alert import AlertCreate, AlertOut
from app.services.alert_service import AlertService

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.post("", response_model=AlertOut, status_code=status.HTTP_201_CREATED)
async def create_alert(
    body: AlertCreate,
    user_id: uuid.UUID = Depends(get_current_user_id),
    service: AlertService = Depends(get_alert_service),
) -> AlertOut:
    return await service.create(user_id, body)


@router.get("", response_model=list[AlertOut])
async def list_alerts(
    status_filter: AlertStatus | None = Query(None, alias="status"),
    user_id: uuid.UUID = Depends(get_current_user_id),
    service: AlertService = Depends(get_alert_service),
) -> list[AlertOut]:
    return await service.list(user_id, status_filter)


@router.delete("/{alert_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_alert(
    alert_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user_id),
    service: AlertService = Depends(get_alert_service),
) -> None:
    await service.delete(user_id, alert_id)
