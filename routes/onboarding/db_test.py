from fastapi import APIRouter

router = APIRouter()

@router.post("/{session_id}/db/test")
def test_db(session_id: str):
    return {"ok": True}
