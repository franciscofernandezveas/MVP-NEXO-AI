from fastapi import APIRouter, Depends, Request, Response, status
from pydantic import BaseModel
from typing import Optional

from .dependencies import get_current_user
from .models import User
from .service import (
    change_password,
    login_user,
    logout_user,
    refresh_session,
    register_user,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None
    role: str = "user"


class LoginRequest(BaseModel):
    email: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class AuthResponse(BaseModel):
    success: bool
    user: dict | None = None
    message: str | None = None


@router.post("/register", response_model=AuthResponse)
async def register(payload: RegisterRequest):
    user = register_user(
        email=payload.email,
        password=payload.password,
        full_name=payload.full_name,
        role=payload.role,
    )
    return AuthResponse(
        success=True,
        user={
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
        },
        message="Cuenta creada exitosamente.",
    )


@router.post("/login", response_model=AuthResponse)
async def login(payload: LoginRequest, request: Request, response: Response):
    return login_user(request, response, payload.email, payload.password)


@router.post("/logout")
async def logout(request: Request, response: Response):
    return logout_user(request, response)


@router.get("/me")
async def me(user: User = Depends(get_current_user)):
    return {"success": True, "user": user.model_dump()}


@router.post("/refresh")
async def refresh(request: Request, response: Response):
    return refresh_session(request, response)


@router.post("/change-password")
async def change_password_endpoint(
    payload: ChangePasswordRequest,
    user: User = Depends(get_current_user),
):
    return change_password(user.id, payload.current_password, payload.new_password)
