from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

from core.supabase_client import get_supabase_client
from utils.analytics import emit_event

router = APIRouter()


class FeedbackType(str, Enum):
    like = "like"
    dislike = "dislike"


class FeedbackRequest(BaseModel):
    response_id: str = Field(..., description="ID de agent_responses")
    message_id: Optional[str] = None
    feedback_type: FeedbackType
    comment: Optional[str] = None


@router.post("/{session_id}/feedback")
async def feedback(session_id: str, body: FeedbackRequest):
    supabase = get_supabase_client()

    response = supabase.table("agent_responses")\
        .select("id, user_id, company_id, session_id")\
        .eq("id", body.response_id)\
        .maybe_single()\
        .execute()

    if not response.data:
        raise HTTPException(status_code=404, detail="Respuesta no encontrada")

    user_id = response.data.get("user_id")
    company_id = response.data.get("company_id")

    supabase.table("feedback").insert({
        "response_id": body.response_id,
        "message_id": body.message_id,
        "user_id": user_id,
        "feedback_type": body.feedback_type.value,
        "comment": body.comment,
    }).execute()

    await emit_event(
        "feedback.given",
        user_id=user_id,
        company_id=company_id,
        session_id=session_id,
        payload={
            "feedback_type": body.feedback_type.value,
            "has_comment": bool(body.comment),
        },
    )

    return {"ok": True}
