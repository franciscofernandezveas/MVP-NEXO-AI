from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone
from uuid import UUID
import logging

from core.supabase_client import get_supabase_client
from utils.analytics import emit_event

router = APIRouter()
logger = logging.getLogger("routes.onboarding.sessions")


class CreateSessionRequest(BaseModel):
    user_id: UUID
    source: Optional[str] = "web"


class SessionResponse(BaseModel):
    session_id: str
    ok: bool


def _safe_execute(query):
    """Ejecuta query y devuelve (data, error). Nunca lanza excepción."""
    try:
        result = query.execute()
        if result is None:
            return None, "Supabase devolvió None"
        return getattr(result, "data", None), None
    except Exception as e:
        logger.warning(f"[supabase] query error: {e}")
        return None, str(e)


async def _get_first_active_company(supabase):
    data, _ = _safe_execute(
        supabase.table("companies")
        .select("id")
        .eq("is_active", True)
        .limit(1)
        .maybe_single()
    )
    return data.get("id") if data else None


async def _get_or_create_user_profile(supabase, user_id: str):
    """
    Busca el perfil en public.user_profiles. Si no existe, lo crea
    asociado a la primera empresa piloto activa.
    El user_id debe existir previamente en auth.users.
    """
    # 1. Buscar perfil existente
    data, err = _safe_execute(
        supabase.table("user_profiles")
        .select("user_id, company_id")
        .eq("user_id", user_id)
        .maybe_single()
    )
    if data:
        return data

    logger.info(f"[sessions] perfil no encontrado para {user_id}, creando uno...")

    # 2. Buscar primera empresa piloto activa
    company_id = await _get_first_active_company(supabase)
    if not company_id:
        logger.error("[sessions] no hay empresa piloto activa")
        return None

    # 3. Crear perfil mínimo
    profile_data, profile_err = _safe_execute(
        supabase.table("user_profiles").insert({
            "user_id": user_id,
            "company_id": company_id,
            "email": "piloto@nexobi.cl",
            "name": "Usuario Piloto",
            "profile": "cafe_admin",
            "role": "operator",
            "is_active": True,
        })
    )

    if profile_data and isinstance(profile_data, list) and len(profile_data) > 0:
        logger.info(f"[sessions] perfil creado para {user_id}")
        return profile_data[0]

    logger.error(f"[sessions] no se pudo crear perfil: {profile_err}")
    return None


@router.post("/create", response_model=SessionResponse)
async def create_session(body: CreateSessionRequest):
    supabase = get_supabase_client()
    user_id_str = str(body.user_id)

    profile = await _get_or_create_user_profile(supabase, user_id_str)
    company_id = profile.get("company_id") if profile else None

    # Fallback: usar primera empresa activa
    if not company_id:
        company_id = await _get_first_active_company(supabase)

    session_data, session_err = _safe_execute(
        supabase.table("sessions").insert({
            "user_id": user_id_str,
            "company_id": company_id,
            "source": body.source or "web",
            "started_at": datetime.now(timezone.utc).isoformat(),
        })
    )

    if not session_data:
        logger.error(f"[sessions] error insertando sesión: {session_err}")
        raise HTTPException(status_code=500, detail=f"Error creando sesión: {session_err}")

    session_id = (
        session_data[0]["id"]
        if isinstance(session_data, list) and len(session_data) > 0
        else session_data["id"]
    )

    # Crear credenciales demo para que el agente pueda operar
    _safe_execute(
        supabase.table("session_credentials").upsert({
            "session_id": session_id,
            "is_demo": True,
            "db_type": "postgresql",
        })
    )

    await emit_event(
        "session.started",
        user_id=user_id_str,
        company_id=company_id,
        session_id=session_id,
        payload={"source": body.source or "web", "auto_created_profile": profile is None},
    )

    await emit_event(
        "feature.used",
        user_id=user_id_str,
        company_id=company_id,
        session_id=session_id,
        payload={"feature_name": "chat"},
    )

    return SessionResponse(session_id=session_id, ok=True)


@router.post("/{session_id}/heartbeat")
async def session_heartbeat(session_id: str):
    supabase = get_supabase_client()

    data, err = _safe_execute(
        supabase.table("sessions")
        .select("id, user_id, company_id")
        .eq("id", session_id)
        .maybe_single()
    )

    if not data:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    await emit_event(
        "session.heartbeat",
        user_id=data.get("user_id"),
        company_id=data.get("company_id"),
        session_id=session_id,
    )
    return {"ok": True}


@router.post("/{session_id}/end")
async def end_session(session_id: str):
    supabase = get_supabase_client()

    data, err = _safe_execute(
        supabase.table("sessions")
        .select("id, user_id, company_id, started_at")
        .eq("id", session_id)
        .maybe_single()
    )

    if not data:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    try:
        started = datetime.fromisoformat(data["started_at"].replace("Z", "+00:00"))
    except Exception:
        started = datetime.now(timezone.utc)

    ended = datetime.now(timezone.utc)
    duration = max(0, int((ended - started).total_seconds()))

    _safe_execute(
        supabase.table("sessions").update({
            "ended_at": ended.isoformat(),
            "duration_seconds": duration,
        }).eq("id", session_id)
    )

    await emit_event(
        "session.ended",
        user_id=data.get("user_id"),
        company_id=data.get("company_id"),
        session_id=session_id,
        payload={"duration_seconds": duration},
    )
    return {"ok": True, "duration_seconds": duration}
