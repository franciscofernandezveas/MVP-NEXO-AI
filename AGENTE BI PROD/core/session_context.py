import os
import json
import asyncio
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

from cryptography.fernet import Fernet

from core.supabase_client import get_session_credentials

_session_locks: Dict[str, asyncio.Lock] = {}

AGENTS_DIR = Path(os.getenv("AGENTS_DIR", "/data/agents"))
AGENTS_DIR.mkdir(parents=True, exist_ok=True)

fernet = Fernet(os.environ["ENCRYPTION_KEY"])


def get_session_lock(session_id: str) -> asyncio.Lock:
    if session_id not in _session_locks:
        _session_locks[session_id] = asyncio.Lock()
    return _session_locks[session_id]


def _prepare_agents_md_path(session_id: str, content: str | None = None) -> str:
    session_dir = AGENTS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    agents_md_path = session_dir / "AGENTS.md"

    if content:
        agents_md_path.write_text(content, encoding="utf-8")

    return str(agents_md_path)


def _clear_db_caches():
    try:
        from core.database import get_engine, get_database
        get_engine.cache_clear()
        get_database.cache_clear()
        print("✅ Cachés de DB limpiadas")
    except Exception as e:
        print(f"⚠️ No se pudieron limpiar cachés de DB: {e}")


@contextmanager
def session_scope(session_id: str):
    credentials = get_session_credentials(session_id)
    if not credentials:
        raise ValueError(f"Sesión no encontrada o sin credenciales: {session_id}")

    original_db_uri = os.environ.get("SUPABASE_DB_URI")
    original_agents_path = os.environ.get("AGENTS_MD_PATH")
    original_default_path = os.environ.get("DEFAULT_AGENTS_MD_PATH")

    try:
        # Solo cambiar la DB. La OpenAI API key es global.
        if credentials.get("is_demo"):
            os.environ["SUPABASE_DB_URI"] = os.getenv("DEMO_DATABASE_URL", "")
        else:
            db_json = fernet.decrypt(credentials["db_config_encrypted"].encode()).decode()
            try:
                db_config = json.loads(db_json)
                database_url = db_config.get("database_url") or db_config.get("SUPABASE_DB_URI")
            except json.JSONDecodeError:
                database_url = db_json

            os.environ["SUPABASE_DB_URI"] = database_url

        # Preparar AGENTS.md por sesión
        agents_content = None
        if not credentials.get("is_demo"):
            from core.supabase_client import get_supabase_client
            supabase = get_supabase_client()
            doc = supabase.table("agent_docs")\
                .select("content")\
                .eq("session_id", session_id)\
                .order("version", desc=True)\
                .limit(1)\
                .maybe_single()\
                .execute()
            if doc.data:
                agents_content = doc.data["content"]

        global_agents = Path(__file__).resolve().parent.parent / "AGENTS.md"
        if agents_content is None and global_agents.exists():
            agents_content = global_agents.read_text(encoding="utf-8")

        agents_md_path = _prepare_agents_md_path(session_id, agents_content)
        os.environ["AGENTS_MD_PATH"] = agents_md_path

        # Limpiar cachés de DB
        _clear_db_caches()

        yield

    finally:
        if original_db_uri is not None:
            os.environ["SUPABASE_DB_URI"] = original_db_uri
        elif "SUPABASE_DB_URI" in os.environ:
            del os.environ["SUPABASE_DB_URI"]

        if original_agents_path is not None:
            os.environ["AGENTS_MD_PATH"] = original_agents_path
        elif "AGENTS_MD_PATH" in os.environ:
            del os.environ["AGENTS_MD_PATH"]

        if original_default_path is not None:
            os.environ["DEFAULT_AGENTS_MD_PATH"] = original_default_path
        elif "DEFAULT_AGENTS_MD_PATH" in os.environ:
            del os.environ["DEFAULT_AGENTS_MD_PATH"]
