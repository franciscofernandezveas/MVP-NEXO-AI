from fastapi import APIRouter, HTTPException

from core.supabase_client import get_supabase_client
from utils.analytics import emit_event

router = APIRouter()


@router.post("/{session_id}/db/test")
async def test_db(session_id: str):
    supabase = get_supabase_client()

    session = supabase.table("sessions").select("id, user_id, company_id").eq("id", session_id).maybe_single().execute()
    if not session.data:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    user_id = session.data.get("user_id")
    company_id = session.data.get("company_id")

    success = True  # lógica real de conexión se integra después

    await emit_event(
        "db.test.success" if success else "db.test.error",
        user_id=user_id,
        company_id=company_id,
        session_id=session_id,
        payload={"success": success},
    )

    return {"ok": True, "success": success}
