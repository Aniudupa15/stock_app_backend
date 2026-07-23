import uuid

from fastapi import APIRouter, Depends, status

from app.api.deps import get_auth_service, get_current_user_id
from app.schemas.auth import (
    LoginRequest,
    LogoutRequest,
    RefreshRequest,
    RegisterRequest,
    TokenPairOut,
    UpdateProfileRequest,
    UserProfileOut,
)
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenPairOut, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    service: AuthService = Depends(get_auth_service),
) -> TokenPairOut:
    return await service.register(body.email, body.password, body.display_name)


@router.post("/login", response_model=TokenPairOut)
async def login(
    body: LoginRequest,
    service: AuthService = Depends(get_auth_service),
) -> TokenPairOut:
    return await service.login(body.email, body.password)


@router.post("/refresh", response_model=TokenPairOut)
async def refresh(
    body: RefreshRequest,
    service: AuthService = Depends(get_auth_service),
) -> TokenPairOut:
    return await service.refresh(body.refresh_token)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    body: LogoutRequest,
    service: AuthService = Depends(get_auth_service),
) -> None:
    await service.logout(body.refresh_token)


@router.get("/me", response_model=UserProfileOut)
async def get_me(
    user_id: uuid.UUID = Depends(get_current_user_id),
    service: AuthService = Depends(get_auth_service),
) -> UserProfileOut:
    return await service.get_profile(user_id)


@router.patch("/me", response_model=UserProfileOut)
async def update_me(
    body: UpdateProfileRequest,
    user_id: uuid.UUID = Depends(get_current_user_id),
    service: AuthService = Depends(get_auth_service),
) -> UserProfileOut:
    return await service.update_profile(user_id, body.display_name, body.email)
