from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone

from core.supabase_client import get_supabase_client
from utils.analytics import emit_event

router = APIRouter()


class DemoStartResponse(BaseModel):
    session_id: str
    ok: bool


@router.post("/start", response_model=DemoStartResponse)
async def start_demo():
    supabase = get_supabase_client()
    slug = "demo-main"

    existing = supabase.table("demo_sessions")\
        .select("id")\
        .eq("slug", slug)\
        .maybe_single()\
        .execute()

    demo_user_id = "dddddddd-dddd-dddd-dddd-dddddddddddd"  # adulto mayor piloto por defecto
    demo_company_id = "11111111-1111-1111-1111-111111111111"

    if existing.data:
        session_id = existing.data["id"]
        return DemoStartResponse(session_id=session_id, ok=True)

    result = supabase.table("demo_sessions")\
        .insert({"slug": slug, "profile": {}, "is_demo": True})\
        .execute()

    session_id = result.data[0]["id"]

    supabase.table("session_credentials")\
        .insert({"session_id": session_id, "is_demo": True})\
        .execute()

    # Registrar también como sesión real de adopción
    supabase.table("sessions").insert({
        "id": session_id,
        "user_id": demo_user_id,
        "company_id": demo_company_id,
        "source": "demo",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }).execute()

    await emit_event(
        "session.started",
        user_id=demo_user_id,
        company_id=demo_company_id,
        session_id=session_id,
        payload={"source": "demo", "is_demo": True},
    )

    await emit_event(
        "feature.used",
        user_id=demo_user_id,
        company_id=demo_company_id,
        session_id=session_id,
        payload={"feature_name": "chat", "source": "demo"},
    )

    return DemoStartResponse(session_id=session_id, ok=True)
