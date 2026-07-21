from fastapi import APIRouter

router = APIRouter()

@router.post("/{session_id}/credentials")
def save_credentials(session_id: str):
    return {"ok": True}
