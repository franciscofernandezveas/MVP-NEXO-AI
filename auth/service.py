import logging
from typing import Optional

from fastapi import HTTPException, status

from .config import settings
from .db import get_user_by_supabase_uid, upsert_user_from_supabase
from .models import User
from .security import decode_supabase_token

logger = logging.getLogger("auth_service")


def get_user_from_token(token: str) -> User:
    logger.debug("Validando token de Supabase")
    payload = decode_supabase_token(token)

    supabase_uid = payload.get("sub")
    email = payload.get("email")

    if not supabase_uid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido: falta identificador de usuario",
        )

    full_name = payload.get("user_metadata", {}).get("full_name")
    email_verified = payload.get("email_confirmed_at") is not None

    row = get_user_by_supabase_uid(supabase_uid)

    # Primer login: sincronizamos perfil local
    if not row:
        role = "admin" if email and email.lower() in settings.admin_email_set else "user"
        logger.info(f"Sincronizando usuario {email} con rol {role}")
        row = upsert_user_from_supabase(
            supabase_uid=supabase_uid,
            email=email or "",
            full_name=full_name,
            role=role,
            email_verified=email_verified,
        )

    if not row.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Usuario inactivo"
        )

    return User(**row)


def sync_user(supabase_uid: str, email: str, full_name: Optional[str] = None) -> User:
    row = get_user_by_supabase_uid(supabase_uid)
    if not row:
        role = "admin" if email.lower() in settings.admin_email_set else "user"
        row = upsert_user_from_supabase(supabase_uid, email, full_name, role)
    return User(**row)
