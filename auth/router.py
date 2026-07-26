from fastapi import APIRouter, Depends
from pydantic import BaseModel

from .dependencies import get_current_user
from .models import User

router = APIRouter(prefix="/api/auth", tags=["auth"])


class ProfileResponse(BaseModel):
    success: bool
    user: dict


@router.get("/me", response_model=ProfileResponse)
async def me(user: User = Depends(get_current_user)):
    """
    Devuelve el perfil local sincronizado con Supabase.
    Si el usuario no existe localmente, se crea automáticamente.
    """
    return ProfileResponse(success=True, user=user.model_dump())


@router.post("/logout")
async def logout():
    """
    Con Supabase Auth los tokens son stateless.
    El logout real se hace en el frontend con supabase.auth.signOut().
    """
    return {"success": True, "message": "Sesión cerrada"}
