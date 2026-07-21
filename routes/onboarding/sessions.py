from fastapi import APIRouter

router = APIRouter()

@router.post("/create")
def create_session():
    return {"session_id": "session-placeholder", "ok": True}
