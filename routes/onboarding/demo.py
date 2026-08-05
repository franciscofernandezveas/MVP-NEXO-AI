from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime, timezone
from core.supabase_client import get_supabase_client
from utils.analytics import emit_event

router = APIRouter()


class DemoStartResponse(BaseModel):
    session_id: str
    ok: bool


@router.post("/start", response_model=DemoStartResponse)
async def start_demo(user_id: str = None, company_id: str = None):
    supabase = get_supabase_client()
    slug = "demo-main"

    default_user_id = "dddddddd-dddd-dddd-dddd-dddddddddddd"
    default_company_id = "11111111-1111-1111-1111-111111111111"

    # Si no vienen parámetros, buscar sesión demo existente
    existing = supabase.table("demo_sessions")\
        .select("id")\
        .eq("slug", slug)\
        .maybe_single()\
        .execute()

    if existing and existing.data:
        session_id = existing.data["id"]
    else:
        result = supabase.table("demo_sessions")\
            .insert({"slug": slug, "profile": {}, "is_demo": True})\
            .execute()

        session_id = result.data[0]["id"]

        supabase.table("session_credentials")\
            .insert({"session_id": session_id, "is_demo": True})\
            .execute()

        # Registrar como sesión real de adopción
        supabase.table("sessions").insert({
            "id": session_id,
            "user_id": user_id or default_user_id,
            "company_id": company_id or default_company_id,
            "source": "demo",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }).execute()

        await emit_event(
            "session.started",
            user_id=user_id or default_user_id,
            company_id=company_id or default_company_id,
            session_id=session_id,
            payload={"source": "demo", "is_demo": True},
        )

    return DemoStartResponse(session_id=session_id, ok=True)
