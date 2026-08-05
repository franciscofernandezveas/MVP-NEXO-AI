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
async def start_demo():
    supabase = get_supabase_client()
    slug = "demo-main"
    default_company_id = "11111111-1111-1111-1111-111111111111"

    existing = supabase.table("demo_sessions")\
        .select("id")\
        .eq("slug", slug)\
        .maybe_single()\
        .execute()

    if existing and existing.data:
        return DemoStartResponse(session_id=existing.data["id"], ok=True)

    result = supabase.table("demo_sessions")\
        .insert({"slug": slug, "profile": {}, "is_demo": True})\
        .execute()

    session_id = result.data[0]["id"]

    # Credenciales demo
    supabase.table("session_credentials")\
        .insert({"session_id": session_id, "is_demo": True})\
        .execute()

    # Sesión de analytics demo (sin user_id real)
    supabase.table("sessions").insert({
        "id": session_id,
        "user_id": None,
        "company_id": default_company_id,
        "source": "demo",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }).execute()

    await emit_event(
        "session.started",
        user_id=None,
        company_id=default_company_id,
        session_id=session_id,
        payload={"source": "demo", "is_demo": True},
    )

    return DemoStartResponse(session_id=session_id, ok=True)
