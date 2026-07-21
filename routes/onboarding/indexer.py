from fastapi import APIRouter

router = APIRouter()

@router.post("/{session_id}/index")
def index(session_id: str):
    return {"ok": True, "chunks": 0}
