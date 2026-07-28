import os
import sys
import logging
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

# ============================================================================
# 1. CONFIGURAR PYTHONPATH - CRÍTICO PARA DOCKER
# ============================================================================

BACKEND_DIR = Path(__file__).parent.absolute()

# Limpiar cualquier instancia previa
for p in list(sys.path):
    if p in [str(BACKEND_DIR), str(BACKEND_DIR / "AGENTE BI PROD")]:
        sys.path.remove(p)

# AGENTE BI PROD primero: contiene core/, agents/, etc.
AGENTE_BI_PATH = BACKEND_DIR / "AGENTE BI PROD"
if AGENTE_BI_PATH.exists():
    sys.path.insert(0, str(AGENTE_BI_PATH))
    print(f"✅ sys.path[0] = {sys.path[0]}")
else:
    print(f"❌ No existe AGENTE BI PROD en: {AGENTE_BI_PATH}")

# Backend después: routes/, utils/, auth/
sys.path.insert(1, str(BACKEND_DIR))
print(f"✅ sys.path[1] = {sys.path[1]}")

# ============================================================================
# 2. CARGAR .ENV
# ============================================================================

from dotenv import load_dotenv

env_file = BACKEND_DIR / ".env"
if env_file.exists():
    load_dotenv(str(env_file), override=True)

# OpenAI API key global para todo el agente BI
os.environ["OPENAI_API_KEY"] = os.getenv("DEMO_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

# Desactivar LangSmith
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

# Variables críticas del MVP
required_vars = [
    "ENCRYPTION_KEY",
    "NEXO_SUPABASE_URL",
    "NEXO_SUPABASE_SERVICE_KEY",
    "DEMO_OPENAI_API_KEY",
    "DEMO_DATABASE_URL",
]
missing = [v for v in required_vars if not os.getenv(v)]
if missing:
    print(f"❌ Variables de entorno faltantes: {missing}")
else:
    print("✅ Variables de entorno del MVP verificadas")

# ============================================================================
# 3. LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("main")

# ============================================================================
# 4. FASTAPI APP
# ============================================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Nexo AI - API",
    version="6.1.0-mvp",
    description="Backend del agente BI con onboarding por sesión",
)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

ALLOWED_ORIGINS = [
    FRONTEND_URL,
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

if os.getenv("ALLOWED_ORIGINS"):
    extra = [o.strip() for o in os.getenv("ALLOWED_ORIGINS").split(",") if o.strip()]
    ALLOWED_ORIGINS.extend(extra)

ALLOWED_ORIGINS = list(set(ALLOWED_ORIGINS))

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Access-Control-Allow-Origin"],
)

@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"➡️  {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"⬅️  {response.status_code} {request.method} {request.url.path}")
    return response

# ============================================================================
# 5. ARCHIVOS ESTÁTICOS
# ============================================================================

FILES_DIR = Path(os.getenv("FILES_DIR", str(BACKEND_DIR / "files"))).resolve()
VIZ_DIR = Path(os.getenv("VIZ_DIR", str(BACKEND_DIR / "visualizations"))).resolve()

for d in [FILES_DIR, VIZ_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logger.info(f"📁 FILES_DIR: {FILES_DIR}")
logger.info(f"📁 VIZ_DIR: {VIZ_DIR}")

app.mount("/files", StaticFiles(directory=str(FILES_DIR)), name="files")
app.mount("/visualizations", StaticFiles(directory=str(VIZ_DIR)), name="visualizations")


# ============================================================================
# 5.5 DEBUG CHARTS
# ============================================================================

@app.get("/debug/charts")
def debug_charts():
    charts_dir = FILES_DIR / "charts"
    return {
        "files_dir": str(FILES_DIR),
        "files_dir_exists": FILES_DIR.exists(),
        "charts_dir": str(charts_dir),
        "charts_dir_exists": charts_dir.exists(),
        "charts": [
            {
                "name": p.name,
                "size": p.stat().st_size,
                "path": str(p.relative_to(FILES_DIR)),
            }
            for p in charts_dir.glob("**/*") if p.is_file()
        ] if charts_dir.exists() else [],
    }


# ============================================================================
# 6. ROUTERS
# ============================================================================

def safe_include_router(import_path: str, prefix: str = "", tags: list = None):
    try:
        module_path, router_name = import_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[router_name])
        router = getattr(module, router_name)
        app.include_router(router, prefix=prefix, tags=tags or [])
        logger.info(f"✅ Router cargado: {import_path}")
    except Exception as e:
        logger.error(f"❌ Error cargando router {import_path}: {e}")
        traceback.print_exc()

# Onboarding MVP (solo estos son críticos para la demo)
safe_include_router("routes.onboarding.demo.router", "/onboarding/demo", ["onboarding"])
safe_include_router("routes.onboarding.sessions.router", "/onboarding/sessions", ["onboarding"])
safe_include_router("routes.onboarding.credentials.router", "/onboarding/sessions", ["onboarding"])
safe_include_router("routes.onboarding.db_test.router", "/onboarding/sessions", ["onboarding"])
safe_include_router("routes.onboarding.agents_md.router", "/onboarding/sessions", ["onboarding"])
safe_include_router("routes.onboarding.indexer.router", "/onboarding/sessions", ["onboarding"])
safe_include_router("routes.onboarding.feedback.router", "/onboarding/sessions", ["onboarding"])
safe_include_router("routes.onboarding.chat.router", "/onboarding/sessions", ["onboarding"])

# Reportes / KPIs
safe_include_router("routes.report_routes.router", "", ["reports"])

# Legacy auth/chat deshabilitados temporalmente para MVP
# safe_include_router("auth.router.router")
# safe_include_router("routes.chat_routes.router")

# ============================================================================
# 7. HEALTH
# ============================================================================

@app.get("/health")
def health():
    return {"status": "ok", "version": "6.1.0-mvp"}

# ============================================================================
# 8. LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🟢 Nexo API iniciada")
    yield
    logger.info("🛑 Nexo API finalizada")

app.router.lifespan_context = lifespan

# ============================================================================
# 9. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
