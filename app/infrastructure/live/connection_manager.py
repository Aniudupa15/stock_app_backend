import logging

from fastapi import WebSocket

logger = logging.getLogger(__name__)

MARKET_CHANNEL = "__market__"


class ConnectionManager:
    """Tracks active `/ws/live` connections and each one's currently
    subscribed symbol set (plus the reserved `MARKET_CHANNEL` for dashboard
    movers). A `subscribe` message fully replaces a connection's set rather
    than merging into it - the client (`LiveSocketService`) always sends its
    complete desired subscription, so there's nothing to reconcile here.
    """

    def __init__(self) -> None:
        self._subscriptions: dict[WebSocket, set[str]] = {}

    def connect(self, websocket: WebSocket) -> None:
        self._subscriptions[websocket] = set()

    def disconnect(self, websocket: WebSocket) -> None:
        self._subscriptions.pop(websocket, None)

    def subscribe(self, websocket: WebSocket, channels: set[str]) -> None:
        if websocket in self._subscriptions:
            self._subscriptions[websocket] = channels

    def all_subscribed_symbols(self) -> set[str]:
        symbols: set[str] = set()
        for channels in self._subscriptions.values():
            symbols |= {c for c in channels if c != MARKET_CHANNEL}
        return symbols

    def is_market_subscribed(self) -> bool:
        return any(MARKET_CHANNEL in channels for channels in self._subscriptions.values())

    def connections_for_symbol(self, symbol: str) -> list[WebSocket]:
        return [ws for ws, channels in self._subscriptions.items() if symbol in channels]

    def market_connections(self) -> list[WebSocket]:
        return [ws for ws, channels in self._subscriptions.items() if MARKET_CHANNEL in channels]

    async def send_json(self, websocket: WebSocket, message: dict) -> None:
        try:
            await websocket.send_json(message)
        except Exception:
            # A connection can die between the broadcaster reading its
            # subscription set and the send landing - the client's own
            # disconnect handler (in the WS route) is what actually removes
            # it from `_subscriptions`; a send failure here is expected and
            # harmless, not something to raise or log loudly for.
            logger.debug("Failed to send to a WebSocket - will be cleaned up on its disconnect event", exc_info=True)
