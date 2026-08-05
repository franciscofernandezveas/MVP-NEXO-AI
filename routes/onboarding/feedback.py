from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone
from uuid import UUID

from core.supabase_client import get_supabase_client
from utils.analytics import emit_event

router = APIRouter()


class FeedbackRequest(BaseModel):
    response_id: UUID
    message_id: Optional[UUID] = None
    feedback_type: str = Field(..., pattern="^(like|dislike)$")
    comment: Optional[str] = None


def _safe_execute(query, silent: bool = True):
    try:
        result = query.execute()
        if result is None:
            return None, "Supabase devolvió None"
        return getattr(result, "data", None), None
    except Exception as e:
        if not silent:
            print(f"[feedback] error supabase: {e}")
        return None, str(e)


@router.post("/{session_id}/feedback")
async def feedback(session_id: str, body: FeedbackRequest):
    supabase = get_supabase_client()

    # Recuperar user_id y company_id desde la respuesta del agente
    response_data, response_err = _safe_execute(
        supabase.table("agent_responses")
        .select("id, user_id, session_id")
        .eq("id", str(body.response_id))
        .maybe_single()
    )

    user_id = response_data.get("user_id") if response_data else None

    # Insertar feedback
    fb_data, fb_err = _safe_execute(
        supabase.table("feedback").insert({
            "response_id": str(body.response_id),
            "message_id": str(body.message_id) if body.message_id else None,
            "user_id": user_id,
            "feedback_type": body.feedback_type,
            "comment": body.comment,
        })
    )

    if not fb_data:
        raise HTTPException(status_code=500, detail=f"No se pudo guardar feedback: {fb_err}")

    await emit_event(
        "feedback.given",
        user_id=user_id,
        company_id=None,
        session_id=session_id,
        payload={
            "feedback_type": body.feedback_type,
            "response_id": str(body.response_id),
            "has_comment": bool(body.comment),
        },
    )

    return {"ok": True}
