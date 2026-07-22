import os
from supabase import create_client

_supabase = None


def _normalize_supabase_url(url: str | None) -> str | None:
    if url and url.endswith('.supabase.com'):
        return url.replace('.supabase.com', '.supabase.co')
    return url


def get_supabase_client():
    global _supabase

    if _supabase is not None:
        return _supabase

    url = _normalize_supabase_url(os.getenv("NEXO_SUPABASE_URL"))
    key = os.getenv("NEXO_SUPABASE_SERVICE_KEY")

    print(f"DEBUG NEXO_SUPABASE_URL: {url}")
    print(f"DEBUG NEXO_SUPABASE_SERVICE_KEY prefix: {key[:15] + '...' if key else 'NO KEY'}")

    if not url or not key:
        raise RuntimeError(
            "NEXO_SUPABASE_URL o NEXO_SUPABASE_SERVICE_KEY no configuradas. "
            "Verifica tu archivo .env o variables de Railway"
        )

    try:
        _supabase = create_client(url, key)
        print(f"✅ Supabase app client creado: {url}")
    except Exception as e:
        print(f"❌ Error creando Supabase client: {e}")
        print(f"   URL: {url}")
        print(f"   KEY prefix: {key[:20]}...")
        raise

    return _supabase


def get_session_credentials(session_id: str):
    supabase = get_supabase_client()
    result = supabase.table("session_credentials")\
        .select("*")\
        .eq("session_id", session_id)\
        .maybe_single()\
        .execute()
    return result.data


def get_session(session_id: str):
    supabase = get_supabase_client()
    result = supabase.table("demo_sessions")\
        .select("*")\
        .eq("id", session_id)\
        .maybe_single()\
        .execute()
    return result.data
