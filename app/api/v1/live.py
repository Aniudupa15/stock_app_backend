import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.exceptions import HTTPException

from app.core.config import get_settings
from app.core.security import decode_access_token
from app.infrastructure.live.connection_manager import MARKET_CHANNEL

logger = logging.getLogger(__name__)

router = APIRouter(tags=["live"])

_UNAUTHORIZED_CLOSE_CODE = 4401


@router.websocket("/ws/live")
async def live_updates(websocket: WebSocket, token: str = Query(...)) -> None:
    """Live push for quotes (per-symbol, whatever the client subscribes to)
    and market movers (the reserved `MARKET_CHANNEL`). WebSocket auth can't
    reuse the REST `Depends(get_current_user_id)` (that's `HTTPBearer`,
    header-only) - the access token travels as a query param instead,
    decoded with the same `decode_access_token` the REST auth path uses.

    Client protocol: send `{"action": "subscribe", "symbols": [...]}` any
    time to (fully) replace this connection's subscription set - include
    `"__market__"` to also receive market-snapshot pushes. The connection
    manager and broadcaster are looked up from `app.state`, set up once in
    the lifespan alongside the existing scheduler.
    """
    settings = get_settings()
    try:
        decode_access_token(token, settings)
    except HTTPException:
        await websocket.close(code=_UNAUTHORIZED_CLOSE_CODE)
        return

    connection_manager = websocket.app.state.live_connection_manager
    await websocket.accept()
    connection_manager.connect(websocket)
    try:
        while True:
            message = await websocket.receive_json()
            if message.get("action") == "subscribe":
                symbols = message.get("symbols", [])
                channels = {s.strip().upper() if s != MARKET_CHANNEL else s for s in symbols}
                connection_manager.subscribe(websocket, channels)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.warning("Live WebSocket connection closed unexpectedly", exc_info=True)
    finally:
        connection_manager.disconnect(websocket)
