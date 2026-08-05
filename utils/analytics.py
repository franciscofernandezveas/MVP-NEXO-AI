import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from supabase import create_client, Client

logger = logging.getLogger("analytics")

_supabase: Optional[Client] = None


def get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        url = os.getenv("NEXO_SUPABASE_URL")
        key = os.getenv("NEXO_SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise RuntimeError("Faltan NEXO_SUPABASE_URL o NEXO_SUPABASE_SERVICE_KEY")
        _supabase = create_client(url, key)
    return _supabase


def emit_event_sync(
    event_type: str,
    user_id: Optional[str] = None,
    company_id: Optional[str] = None,
    session_id: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
):
    try:
        get_supabase().table("events").insert({
            "event_type": event_type,
            "user_id": user_id,
            "company_id": company_id,
            "session_id": session_id,
            "payload": payload or {},
            "client_timestamp": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        logger.warning(f"[analytics] no se pudo guardar {event_type}: {e}")


async def emit_event(
    event_type: str,
    user_id: Optional[str] = None,
    company_id: Optional[str] = None,
    session_id: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
):
    """Emite evento en background sin bloquear."""
    asyncio.create_task(
        asyncio.to_thread(emit_event_sync, event_type, user_id, company_id, session_id, payload)
    )
