from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone
from uuid import UUID

from core.supabase_client import get_supabase_client
from utils.analytics import emit_event

router = APIRouter()

logger = __import__("logging").getLogger("routes.onboarding.sessions")


class CreateSessionRequest(BaseModel):
    user_id: UUID
    source: Optional[str] = "web"


class SessionResponse(BaseModel):
    session_id: str
    ok: bool


@router.post("/create", response_model=SessionResponse)
async def create_session(body: CreateSessionRequest):
    supabase = get_supabase_client()

    user = supabase.table("users").select("id, company_id").eq("id", str(body.user_id)).maybe_single().execute()
    if not user.data:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    user_id = user.data["id"]
    company_id = user.data.get("company_id")

    result = supabase.table("sessions").insert({
        "user_id": user_id,
        "company_id": company_id,
        "source": body.source or "web",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }).execute()

    session_id = result.data[0]["id"]

    await emit_event(
        "session.started",
        user_id=user_id,
        company_id=company_id,
        session_id=session_id,
        payload={"source": body.source or "web"},
    )

    await emit_event(
        "feature.used",
        user_id=user_id,
        company_id=company_id,
        session_id=session_id,
        payload={"feature_name": "chat"},
    )

    return SessionResponse(session_id=session_id, ok=True)


@router.post("/{session_id}/heartbeat")
async def session_heartbeat(session_id: str):
    supabase = get_supabase_client()

    session = supabase.table("sessions").select("id, user_id, company_id").eq("id", session_id).maybe_single().execute()
    if not session.data:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    await emit_event(
        "session.heartbeat",
        user_id=session.data.get("user_id"),
        company_id=session.data.get("company_id"),
        session_id=session_id,
    )
    return {"ok": True}


@router.post("/{session_id}/end")
async def end_session(session_id: str):
    supabase = get_supabase_client()

    session = supabase.table("sessions").select("id, user_id, company_id, started_at").eq("id", session_id).maybe_single().execute()
    if not session.data:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    started = datetime.fromisoformat(session.data["started_at"].replace("Z", "+00:00"))
    ended = datetime.now(timezone.utc)
    duration = int((ended - started).total_seconds())

    supabase.table("sessions").update({
        "ended_at": ended.isoformat(),
        "duration_seconds": duration,
    }).eq("id", session_id).execute()

    await emit_event(
        "session.ended",
        user_id=session.data.get("user_id"),
        company_id=session.data.get("company_id"),
        session_id=session_id,
        payload={"duration_seconds": duration},
    )
    return {"ok": True, "duration_seconds": duration}
