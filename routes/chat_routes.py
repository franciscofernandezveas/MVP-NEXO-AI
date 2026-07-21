# routes/chat_routes.py
from fastapi import APIRouter, Depends, Query, Body
from pydantic import BaseModel
from typing import Optional
from auth import get_current_user, User

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ConversationCreate(BaseModel):
    user_id: str


class ConversationOut(BaseModel):
    id: str
    title: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@router.get("/conversations")
async def get_conversations(
    user: User = Depends(get_current_user),
    user_id: str = Query(...),
    limit: int = Query(50)
):
    """
    Devuelve las conversaciones del usuario.
    """
    return {"conversations": [], "total": 0, "limit": limit}


@router.post("/conversations")
async def create_conversation(
    body: ConversationCreate,
    user: User = Depends(get_current_user)
):
    """
    Crea una nueva conversación para el usuario.
    """
    import uuid
    from datetime import datetime

    return {
        "success": True,
        "conversation": {
            "id": str(uuid.uuid4()),
            "user_id": body.user_id,
            "title": "Nueva conversación",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
    }
