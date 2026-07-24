from unittest.mock import AsyncMock

from app.infrastructure.live.connection_manager import MARKET_CHANNEL, ConnectionManager


def _fake_websocket():
    ws = AsyncMock()
    return ws


def test_connect_starts_with_an_empty_subscription():
    manager = ConnectionManager()
    ws = _fake_websocket()

    manager.connect(ws)

    assert manager.all_subscribed_symbols() == set()
    assert manager.is_market_subscribed() is False


def test_subscribe_replaces_the_full_set_not_merges():
    manager = ConnectionManager()
    ws = _fake_websocket()
    manager.connect(ws)

    manager.subscribe(ws, {"RELIANCE", "TCS"})
    manager.subscribe(ws, {"INFY"})

    assert manager.all_subscribed_symbols() == {"INFY"}


def test_subscribe_on_unconnected_websocket_is_a_no_op():
    manager = ConnectionManager()
    ws = _fake_websocket()

    manager.subscribe(ws, {"RELIANCE"})

    assert manager.all_subscribed_symbols() == set()


def test_disconnect_removes_the_connection_and_its_subscriptions():
    manager = ConnectionManager()
    ws = _fake_websocket()
    manager.connect(ws)
    manager.subscribe(ws, {"RELIANCE"})

    manager.disconnect(ws)

    assert manager.all_subscribed_symbols() == set()
    assert manager.connections_for_symbol("RELIANCE") == []


def test_all_subscribed_symbols_unions_across_connections_and_excludes_market_channel():
    manager = ConnectionManager()
    ws_a, ws_b = _fake_websocket(), _fake_websocket()
    manager.connect(ws_a)
    manager.connect(ws_b)
    manager.subscribe(ws_a, {"RELIANCE", MARKET_CHANNEL})
    manager.subscribe(ws_b, {"TCS"})

    assert manager.all_subscribed_symbols() == {"RELIANCE", "TCS"}


def test_is_market_subscribed_true_only_if_some_connection_wants_it():
    manager = ConnectionManager()
    ws = _fake_websocket()
    manager.connect(ws)

    assert manager.is_market_subscribed() is False

    manager.subscribe(ws, {MARKET_CHANNEL})

    assert manager.is_market_subscribed() is True
    assert manager.market_connections() == [ws]


def test_connections_for_symbol_returns_only_matching_connections():
    manager = ConnectionManager()
    ws_a, ws_b = _fake_websocket(), _fake_websocket()
    manager.connect(ws_a)
    manager.connect(ws_b)
    manager.subscribe(ws_a, {"RELIANCE"})
    manager.subscribe(ws_b, {"TCS"})

    assert manager.connections_for_symbol("RELIANCE") == [ws_a]
    assert manager.connections_for_symbol("TCS") == [ws_b]
    assert manager.connections_for_symbol("INFY") == []


async def test_send_json_swallows_errors_from_a_dead_connection():
    manager = ConnectionManager()
    ws = _fake_websocket()
    ws.send_json.side_effect = RuntimeError("connection closed")

    await manager.send_json(ws, {"type": "quote"})  # must not raise
