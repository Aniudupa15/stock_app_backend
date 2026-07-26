"""Shared dependencies for the trading API - reuses the data-service's JWT
auth (`get_current_user_id`) and DB session (`get_db_session`) so both
services speak the same tokens against the same database."""

from __future__ import annotations

import uuid

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user_id
from app.core.config import Settings, get_settings
from app.infrastructure.db.session import get_db_session
from services.trading_service.persistence.models import TradingAccountModel
from services.trading_service.persistence.repositories import TradingAccountRepository
from services.trading_service.security import CredentialCipher

__all__ = ["get_current_user_id", "get_db_session", "get_owned_account", "get_cipher"]


def get_cipher(settings: Settings = Depends(get_settings)) -> CredentialCipher:
    return CredentialCipher(settings)


async def get_owned_account(
    account_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
) -> TradingAccountModel:
    """Fetch an account and 404 unless it belongs to the caller (never leak
    the existence of another user's account)."""
    account = await TradingAccountRepository(session).get(account_id)
    if account is None or account.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="account not found")
    return account
