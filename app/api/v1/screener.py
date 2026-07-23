from fastapi import APIRouter, Depends

from app.api.deps import get_screener_service
from app.schemas.screener import ScreenerRequest, ScreenerResultOut
from app.services.screener_service import ScreenerService

router = APIRouter(prefix="/screener", tags=["screener"])


@router.post("", response_model=list[ScreenerResultOut])
async def screen_stocks(
    body: ScreenerRequest,
    service: ScreenerService = Depends(get_screener_service),
) -> list[ScreenerResultOut]:
    return await service.screen(body)
