from fastapi import APIRouter

router = APIRouter()

@router.post("/{session_id}/agents-md/save")
def save_agents_md(session_id: str):
    return {"ok": True}

@router.post("/{session_id}/agents-md/build")
def build_agents_md(session_id: str):
    return {"ok": True}
