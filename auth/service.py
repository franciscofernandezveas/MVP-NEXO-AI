import logging
from fastapi import HTTPException, Request, Response, status

from .config import settings
from .db import (
    create_user_db,
    get_user_by_email,
    get_user_by_id,
    revoke_all_user_refresh_tokens_db,
    revoke_refresh_token_db,
    store_refresh_token_db,
    update_user_password,
    verify_refresh_token_db,
)
from .models import User
from .security import (
    check_rate_limit,
    clear_auth_cookies,
    create_access_token,
    create_refresh_token,
    decode_access_token,
    get_client_ip,
    hash_password,
    hash_token,
    record_failed_attempt,
    set_auth_cookies,
    verify_password,
)

logger = logging.getLogger("auth_service")


def _user_from_row(row: dict) -> User:
    return User(
        id=str(row["id"]),
        email=row["email"],
        full_name=row.get("full_name"),
        role=row.get("role", "user"),
        is_active=row.get("is_active", True),
        email_verified=row.get("email_verified", False),
    )


def register_user(email: str, password: str, full_name=None, role="user") -> User:
    logger.debug(f"REGISTER: intentando registrar {email}")
    try:
        password_hash = hash_password(password)
    except ValueError as e:
        logger.warning(f"REGISTER: password débil - {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    existing = get_user_by_email(email)
    if existing:
        logger.warning(f"REGISTER: email ya existe {email}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ya existe una cuenta con este email",
        )

    row = create_user_db(email, password_hash, full_name, role)
    logger.info(f"REGISTER: usuario creado {email} id={row['id']}")
    return _user_from_row(row)


def login_user(request: Request, response: Response, email: str, password: str) -> dict:
    logger.debug(f"LOGIN: intento para {email} desde IP {get_client_ip(request)}")
    check_rate_limit(request, email)

    row = get_user_by_email(email)
    if not row:
        logger.warning(f"LOGIN: usuario no encontrado {email}")
        record_failed_attempt(request, email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Credenciales inválidas"
        )

    user = _user_from_row(row)
    logger.debug(f"LOGIN: usuario encontrado {email} id={user.id} active={user.is_active}")

    if not user.is_active:
        logger.warning(f"LOGIN: usuario inactivo {email}")
        record_failed_attempt(request, email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Usuario inactivo"
        )

    if not verify_password(password, row["password_hash"]):
        logger.warning(f"LOGIN: password incorrecta {email}")
        record_failed_attempt(request, email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Credenciales inválidas"
        )

    logger.info(f"LOGIN: credenciales válidas {email}")
    access_token = create_access_token(user.id, user.email, user.role)
    refresh_token = create_refresh_token()

    store_refresh_token_db(
        user.id,
        hash_token(refresh_token),
        ip=get_client_ip(request),
        user_agent=request.headers.get("user-agent"),
    )

    set_auth_cookies(response, access_token, refresh_token)
    logger.debug(f"LOGIN: cookies seteadas para {email}")

    return {
        "success": True,
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
        },
    }


def logout_user(request: Request, response: Response) -> dict:
    logger.debug("LOGOUT: cerrando sesión")
    refresh_token = request.cookies.get(settings.REFRESH_TOKEN_COOKIE_NAME)
    if refresh_token:
        revoke_refresh_token_db(hash_token(refresh_token))
    clear_auth_cookies(response)
    return {"success": True}


def refresh_session(request: Request, response: Response) -> dict:
    logger.debug("REFRESH: intentando refresh")
    refresh_token = request.cookies.get(settings.REFRESH_TOKEN_COOKIE_NAME)
    if not refresh_token:
        logger.warning("REFRESH: no hay refresh token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="No hay refresh token"
        )

    row = verify_refresh_token_db(hash_token(refresh_token))
    if not row:
        logger.warning("REFRESH: token inválido o expirado")
        clear_auth_cookies(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token inválido o expirado",
        )

    user = _user_from_row(row)
    logger.debug(f"REFRESH: usuario {user.email}")

    access_token = create_access_token(user.id, user.email, user.role)
    new_refresh_token = create_refresh_token()

    store_refresh_token_db(
        user.id,
        hash_token(new_refresh_token),
        ip=get_client_ip(request),
        user_agent=request.headers.get("user-agent"),
    )

    set_auth_cookies(response, access_token, new_refresh_token)
    return {
        "success": True,
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
        },
    }


def change_password(user_id: str, current_password: str, new_password: str) -> dict:
    logger.debug(f"CHANGE_PASSWORD: user_id={user_id}")
    row = get_user_by_id(user_id)
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado"
        )

    if not verify_password(current_password, row["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Contraseña actual incorrecta"
        )

    new_hash = hash_password(new_password)
    update_user_password(user_id, new_hash)
    revoke_all_user_refresh_tokens_db(user_id)
    return {
        "success": True,
        "message": "Contraseña actualizada. Inicia sesión de nuevo.",
    }


def get_user_from_token(token: str) -> User:
    logger.debug(f"GET_USER_FROM_TOKEN: token recibido")
    payload = decode_access_token(token)
    logger.debug(f"GET_USER_FROM_TOKEN: payload sub={payload.get('sub')}")
    row = get_user_by_id(payload.get("sub"))
    if not row:
        logger.warning("GET_USER_FROM_TOKEN: usuario no encontrado")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Usuario no encontrado"
        )
    user = _user_from_row(row)
    if not user.is_active:
        logger.warning(f"GET_USER_FROM_TOKEN: usuario inactivo {user.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Usuario inactivo"
        )
    logger.debug(f"GET_USER_FROM_TOKEN: OK {user.email}")
    return user
