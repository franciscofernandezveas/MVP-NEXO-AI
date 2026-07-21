from fastapi import APIRouter

router = APIRouter()

@router.post("/{session_id}/feedback")
def feedback(session_id: str):
    return {"ok": True}
