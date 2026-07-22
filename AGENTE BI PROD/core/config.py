import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("bi_orchestrator")

# Defaults para desarrollo (sobrescribir con variables de entorno reales)
os.environ.setdefault("OPENAI_API_KEY", "sk-...")
os.environ.setdefault("SUPABASE_DB_URI", "postgresql://user:pass@host:5432/db")
