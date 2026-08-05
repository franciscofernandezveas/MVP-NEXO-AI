from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from core.supabase_client import get_supabase_client
from utils.analytics import emit_event

router = APIRouter()


class CredentialsRequest(BaseModel):
    db_type: str = "postgresql"
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None


@router.post("/{session_id}/credentials")
async def save_credentials(session_id: str, body: CredentialsRequest):
    supabase = get_supabase_client()

    session = supabase.table("sessions").select("id, user_id, company_id").eq("id", session_id).maybe_single().execute()
    if not session.data:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    user_id = session.data.get("user_id")
    company_id = session.data.get("company_id")

    supabase.table("session_credentials").upsert({
        "session_id": session_id,
        "db_type": body.db_type,
        "is_demo": False,
    }).execute()

    await emit_event(
        "company.connected",
        user_id=user_id,
        company_id=company_id,
        session_id=session_id,
        payload={"db_type": body.db_type},
    )

    return {"ok": True}
