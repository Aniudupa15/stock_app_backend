"""Broker connectivity endpoints (Zerodha first).

connect  -> store encrypted api_key/api_secret, return the Kite login URL
complete -> exchange the request_token for an access token (stored encrypted)
status   -> is there a valid, unexpired session?

Credentials are encrypted at rest (CredentialCipher); the api_secret and
access_token are never returned to the client.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from libs.broker.base import BrokerError
from libs.broker.zerodha import create_live_adapter
from libs.trading_calendar.calendar import IST
from services.trading_service.api.deps import get_cipher, get_current_user_id, get_db_session
from services.trading_service.api.schemas import BrokerCompleteIn, BrokerConnectIn, BrokerConnectOut, BrokerStatusOut
from services.trading_service.persistence.repositories import BrokerSessionRepository
from services.trading_service.security import CredentialCipher

router = APIRouter(prefix="/trading/broker", tags=["broker"])


def _login_url(api_key: str) -> str:
    return f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"


@router.post("/connect", response_model=BrokerConnectOut)
async def connect(
    body: BrokerConnectIn,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
    cipher: CredentialCipher = Depends(get_cipher),
) -> BrokerConnectOut:
    model = await BrokerSessionRepository(session).upsert_credentials(
        user_id, body.broker, cipher.encrypt(body.api_key), cipher.encrypt(body.api_secret)
    )
    return BrokerConnectOut(broker=body.broker, login_url=_login_url(body.api_key), status=model.status)


@router.get("/status", response_model=BrokerStatusOut)
async def broker_status(
    broker: str = "zerodha",
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
) -> BrokerStatusOut:
    model = await BrokerSessionRepository(session).get(user_id, broker)
    if model is None:
        return BrokerStatusOut(broker=broker, connected=False, status="DISCONNECTED")
    now = datetime.now(IST)
    valid = model.status == "CONNECTED" and (
        model.access_token_expires_at is None or now < model.access_token_expires_at
    )
    return BrokerStatusOut(
        broker=broker,
        connected=valid,
        status=model.status if valid else "EXPIRED" if model.status == "CONNECTED" else model.status,
        expires_at=model.access_token_expires_at,
    )


@router.post("/complete-login", response_model=BrokerStatusOut)
async def complete_login(
    body: BrokerCompleteIn,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
    cipher: CredentialCipher = Depends(get_cipher),
) -> BrokerStatusOut:
    repo = BrokerSessionRepository(session)
    model = await repo.get(user_id, body.broker)
    if model is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="connect the broker first")

    api_key = cipher.decrypt(model.api_key_enc)
    api_secret = cipher.decrypt(model.api_secret_enc)
    try:
        adapter = create_live_adapter(api_key, api_secret, user_id)
        broker_session = await adapter.complete_login({"request_token": body.request_token})
    except BrokerError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    await repo.set_access_token(
        model.id, cipher.encrypt(broker_session.access_token), broker_session.expires_at, "CONNECTED"
    )
    return BrokerStatusOut(
        broker=body.broker, connected=True, status="CONNECTED", expires_at=broker_session.expires_at
    )
