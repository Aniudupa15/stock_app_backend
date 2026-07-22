from fastapi import APIRouter, Depends

from app.api.deps import get_dashboard_service
from app.schemas.dashboard import DashboardOut
from app.services.dashboard_service import DashboardService

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("", response_model=DashboardOut)
async def get_dashboard(
    service: DashboardService = Depends(get_dashboard_service),
) -> DashboardOut:
    return await service.get_dashboard()
