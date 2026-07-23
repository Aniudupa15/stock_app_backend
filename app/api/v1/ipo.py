from fastapi import APIRouter, Depends, Query

from app.api.deps import get_ipo_service
from app.schemas.ipo import IpoFilingOut
from app.services.ipo_service import IpoService

router = APIRouter(prefix="/ipo", tags=["ipo"])


@router.get("", response_model=list[IpoFilingOut])
async def list_ipos(
    status: str | None = Query(None, description="ACTIVE, LISTED, or another NSE-reported status value"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    service: IpoService = Depends(get_ipo_service),
) -> list[IpoFilingOut]:
    return await service.list(status, limit, offset)
