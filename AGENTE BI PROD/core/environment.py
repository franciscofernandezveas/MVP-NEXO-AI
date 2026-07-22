import os
import logging
import urllib.parse
from pathlib import Path
from dotenv import load_dotenv
import requests

logger = logging.getLogger(__name__)


def setup_environment() -> str | None:
    """
    Carga variables de entorno y normaliza la conexión a base de datos.
    Soporta:
      - SUPABASE_DB_URI explícita
      - DATABASE_URL (fallback)
      - Variables por partes (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD)
    """
    # 1. Limpiar variables conflictivas previas
    problematic_vars = [
        'OPENAI_API_KEY', 'OPENAI_API_BASE',
        'AZURE_OPENAI_ENDPOINT', 'OPENAI_API_VERSION',
        'OPENAI_CHAT_MODEL'
    ]
    for var in problematic_vars:
        if var in os.environ:
            del os.environ[var]

    # 2. Cargar .env desde la raíz del proyecto
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=True)
        logger.info(f"✅ .env cargado desde: {env_path}")
    else:
        load_dotenv(override=True)
        logger.warning("⚠️  No se encontró .env en la raíz. Usando variables del sistema.")

    # 3. Normalizar la URI de base de datos (SUPABASE_DB_URI es la estándar del proyecto)
    db_uri = _resolve_database_uri()
    if db_uri:
        os.environ["SUPABASE_DB_URI"] = db_uri
        logger.info("✅ SUPABASE_DB_URI configurada correctamente.")
    else:
        logger.error("❌ No se pudo construir SUPABASE_DB_URI. Verifica tu .env")

    # 4. Devolver API key de OpenAI para validación inmediata
    return os.getenv("OPENAI_API_KEY")


def _resolve_database_uri() -> str | None:
    """
    Intenta obtener la URI de PostgreSQL de múltiples fuentes.
    Prioridad:
      1. SUPABASE_DB_URI (ya completa)
      2. DATABASE_URL (ya completa)
      3. Construir desde DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    """
    # Prioridad 1: Variable directa
    uri = os.getenv("SUPABASE_DB_URI", "").strip()
    if uri and "host" not in uri.lower():
        return uri

    # Prioridad 2: DATABASE_URL
    database_url = os.getenv("DATABASE_URL", "").strip()
    if database_url and "host" not in database_url.lower():
        return database_url

    # Prioridad 3: Construir desde partes
    host = os.getenv("DB_HOST", "").strip()
    port = os.getenv("DB_PORT", "").strip()
    db_name = os.getenv("DB_NAME", "").strip()
    user = os.getenv("DB_USER", "").strip()
    password = os.getenv("DB_PASSWORD", "")

    if all([host, port, db_name, user]):
        # Escapar la contraseña para que sea válida en URL
        # Los caracteres especiales como $, @, #, etc. deben escaparse
        safe_password = urllib.parse.quote(str(password), safe='')
        
        # Detectar si es el Pooler de Supabase (usa puerto 6543)
        pooler_suffix = "?pgbouncer=true&connection_limit=1" if port == "6543" else ""
        
        uri = f"postgresql://{user}:{safe_password}@{host}:{port}/{db_name}{pooler_suffix}"
        return uri

    return None


def verify_openai_connection(api_key: str) -> bool:
    """Verifica conexión con OpenAI API (opcional)."""
    if not api_key or not api_key.startswith('sk-'):
        logger.error("API key no válida")
        return False

    try:
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        success = response.status_code == 200
        if success:
            logger.info("✅ Conexión con OpenAI verificada")
        else:
            logger.error(f"Error en API OpenAI: {response.status_code}")
        return success
    except Exception as e:
        logger.error(f"Error de conexión con OpenAI: {e}")
        return False
