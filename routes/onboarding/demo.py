from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.supabase_client import get_supabase_client

router = APIRouter()


class DemoStartResponse(BaseModel):
    session_id: str
    ok: bool


@router.post("/start", response_model=DemoStartResponse)
def start_demo():
    supabase = get_supabase_client()
    slug = "demo-main"

    existing = supabase.table("demo_sessions")\
        .select("id")\
        .eq("slug", slug)\
        .maybe_single()\
        .execute()

    if existing.data:
        return DemoStartResponse(session_id=existing.data["id"], ok=True)

    result = supabase.table("demo_sessions")\
        .insert({"slug": slug, "profile": {}, "is_demo": True})\
        .execute()

    session_id = result.data[0]["id"]

    supabase.table("session_credentials")\
        .insert({"session_id": session_id, "is_demo": True})\
        .execute()

    return DemoStartResponse(session_id=session_id, ok=True)
