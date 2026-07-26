import time
from typing import Optional

import jwt
from fastapi import HTTPException, Request, status

from .config import settings

_FAILED_ATTEMPTS: dict[str, list[float]] = {}
MAX_ATTEMPTS = 5
WINDOW_SECONDS = 900


def _rate_limit_key(request: Request, email: str) -> str:
    client_ip = request.client.host if request.client else "unknown"
    return f"{client_ip}:{email.lower()}"


def check_rate_limit(request: Request, email: str):
    key = _rate_limit_key(request, email)
    now = time.time()
    attempts = [t for t in _FAILED_ATTEMPTS.get(key, []) if now - t < WINDOW_SECONDS]
    if len(attempts) >= MAX_ATTEMPTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Demasiados intentos fallidos. Intente más tarde.",
        )
    _FAILED_ATTEMPTS[key] = attempts


def record_failed_attempt(request: Request, email: str):
    key = _rate_limit_key(request, email)
    now = time.time()
    _FAILED_ATTEMPTS.setdefault(key, []).append(now)


def get_client_ip(request: Request) -> Optional[str]:
    return request.client.host if request.client else None


def decode_supabase_token(token: str) -> dict:
    """
    Decodifica un JWT de acceso emitido por Supabase Auth.
    """
    try:
        payload = jwt.decode(
            token,
            settings.SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            options={"verify_aud": False},
        )
        if payload.get("type") not in (None, "access"):
            raise jwt.InvalidTokenError("Tipo de token inválido")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expirado"
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token inválido: {str(e)}",
        )
