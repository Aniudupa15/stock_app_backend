from pydantic import BaseModel, EmailStr, Field

from app.core.security import MAX_PASSWORD_LENGTH


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=MAX_PASSWORD_LENGTH)
    display_name: str = Field(..., min_length=1, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=MAX_PASSWORD_LENGTH)


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str


class TokenPairOut(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
