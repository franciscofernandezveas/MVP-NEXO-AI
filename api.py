#!/usr/bin/env python3
# api.py - FastAPI Server con Agente BI integrado + Auth propio + Streaming
# v6.0.0 - Agente BI embebido, warm-up de DB, streaming real del grafo

import os
import sys
import logging
import shutil
import re
import json
import asyncio
import time
import traceback
from datetime import datetime, date, time as dt_time
from typing import Dict, Optional, Any, List, AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4, UUID
from decimal import Decimal
from urllib.parse import quote_plus

BACKEND_DIR = Path(__file__).parent.absolute()
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# PYTHONPATH para submódulos
PROJECT_ROOT = BACKEND_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

AGENTE_BI_PATH = BACKEND_DIR / "AGENTE BI PROD"
if AGENTE_BI_PATH.exists():
    sys.path.insert(0, str(AGENTE_BI_PATH))

try:
    from dotenv import load_dotenv
    env_file = BACKEND_DIR / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)
        print("✅ .env cargado")
except Exception as e:
    print(f"⚠️ .env error: {e}")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("api")

logger.info("=" * 70)
logger.info("🚀 INICIANDO API.PY v6.0.0 - AGENTE BI INTEGRADO")
logger.info("=" * 70)

# ============================================================================
# 1. ENTORNO Y BASE DE DATOS
# ============================================================================

from core.environment import setup_environment

api_key = setup_environment()
if not api_key:
    print("❌ ERROR: OPENAI_API_KEY no está definida. Verifica tu archivo .env")
    sys.exit(1)

SUPABASE_DB_URI = os.getenv("SUPABASE_DB_URI") or os.getenv("DATABASE_URL")
if not SUPABASE_DB_URI:
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT", "6543")
    db_name = os.getenv("DB_NAME", "postgres")
    if all([db_user, db_password, db_host, db_port, db_name]):
        safe_password = quote_plus(db_password)
        SUPABASE_DB_URI = f"postgresql://{db_user}:{safe_password}@{db_host}:{db_port}/{db_name}?sslmode=require"

if not SUPABASE_DB_URI:
    print("❌ ERROR: No se pudo construir SUPABASE_DB_URI.")
    sys.exit(1)

os.environ["SUPABASE_DB_URI"] = SUPABASE_DB_URI
safe_uri = re.sub(r':([^:@]+)@', ':****@', SUPABASE_DB_URI)
logger.info(f"✅ DB URI configurada: {safe_uri}")
logger.info(f"✅ Configuración OK. DB_HOST detectado: {SUPABASE_DB_URI.split('@')[-1].split('/')[0]}")

# ============================================================================
# 2. WARM-UP DE BASE DE DATOS E IMPORTACIÓN DEL ORQUESTADOR
# ============================================================================

print("🔌 Iniciando warm-up de base de datos...")
from core.database import warmup_db

try:
    warmup_db()
    logger.info("✅ Warm-up de base de datos completado")
except Exception as e:
    logger.warning(f"⚠️ Warm-up de base de datos falló: {e}")

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from core.orchestrator import BI_ORCHESTRATOR
from core.config import logger as bi_logger

AGENTE_BI_AVAILABLE = BI_ORCHESTRATOR is not None
logger.info(f"✅ Agente BI cargado: {AGENTE_BI_AVAILABLE}")

# ============================================================================
# 3. FASTAPI, CORS Y ROUTERS
# ============================================================================

from fastapi import FastAPI, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# AUTH PROPIO
from auth import get_current_user, User, router as auth_router
logger.info("✅ Auth import OK")

# RUTAS DE NEGOCIO
from routes.sales_routes import router as sales_router
logger.info("✅ sales_routes import OK")

reports_router = None
monthly_reports_router = None
chat_router = None

try:
    from routes.reports_routes import router as reports_router
    logger.info("✅ reports_routes import OK")
except Exception as e:
    logger.warning(f"⚠️ reports_routes no disponible: {e}")

try:
    from routes.monthly_reports_routes import router as monthly_reports_router
    logger.info("✅ monthly_reports_routes import OK")
except Exception as e:
    logger.warning(f"⚠️ monthly_reports_routes no disponible: {e}")

try:
    from backend.routes.onboarding.chat_routes import router as chat_router
    logger.info("✅ chat_routes import OK")
except Exception as e:
    logger.warning(f"⚠️ chat_routes no disponible: {e}")

# SUPABASE (solo para storage/RAG, no auth)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

if SUPABASE_URL and SUPABASE_URL.endswith('.supabase.com'):
    SUPABASE_URL = SUPABASE_URL.replace('.supabase.com', '.supabase.co')

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("✅ Supabase client creado")
    except Exception as e:
        logger.error(f"❌ Supabase init error: {e}")

# DIRECTORIOS
INVOICES_DIR = BACKEND_DIR / "files" / "invoices"
REPORTS_DIR = BACKEND_DIR / "files" / "reports"
CHARTS_DIR = BACKEND_DIR / "files" / "charts"
VIZ_DIR = BACKEND_DIR / "visualizations"

for d in [INVOICES_DIR, REPORTS_DIR, CHARTS_DIR, VIZ_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SESSIONS: Dict[str, Any] = {}
PENDING_ACTIONS: Dict[str, Dict[str, Any]] = {}

# APP + CORS
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3001")

ALLOWED_ORIGINS = [
    FRONTEND_URL,
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

logger.info(f"FRONTEND_URL CORS: {FRONTEND_URL}")
logger.info(f"ALLOWED_ORIGINS CORS: {ALLOWED_ORIGINS}")

app = FastAPI(
    title="Nexo AI - API",
    version="6.0.0",
    description="API con Agente BI integrado + Auth propio + Streaming real del grafo",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/files", StaticFiles(directory=str(BACKEND_DIR / "files")), name="files")
if VIZ_DIR.exists():
    app.mount("/visualizations", StaticFiles(directory=str(VIZ_DIR)), name="visualizations")

# MIDDLEWARE LOG
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"➡️ {request.method} {request.url.path} | Origin: {request.headers.get('origin')} | Cookie: {request.headers.get('cookie', 'NO COOKIE')[:60]}")
    response = await call_next(request)
    logger.info(f"⬅️ {response.status_code} {request.method} {request.url.path}")
    return response

# REGISTRO DE RUTAS
logger.info("Registrando routers...")
app.include_router(auth_router)
app.include_router(sales_router)

if reports_router:
    app.include_router(reports_router)
if monthly_reports_router:
    app.include_router(monthly_reports_router)
if chat_router:
    app.include_router(chat_router)

# ============================================================================
# 4. SERIALIZADOR JSON Y VISUALIZACIÓN DE EVENTOS (main.py)
# ============================================================================

def _json_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (datetime, date, dt_time)):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def print_header(text: str):
    print(f"\n{'='*70}")
    print(f" {text}")
    print(f"{'='*70}")


def print_event(agent, iteration: int, event: dict):
    """Renderiza el estado de cada nodo durante el streaming."""
    if agent == "planner":
        plan = event.get("plan")
        if plan:
            print(f"\n📋 [Ciclo {iteration}] PLANNER")
            print(f"   ├─ Intención : {plan.intent}")
            print(f"   ├─ Tipo      : {plan.question_type}")
            print(f"   ├─ Métricas  : {plan.metrics}")
            print(f"   ├─ Dimension : {plan.dimensions}")

            if hasattr(plan, 'tasks') and plan.tasks:
                tasks_summary = " | ".join([f"T{t.task_id}:{t.task[:40]}" for t in plan.tasks])
                print(f"   ├─ Tasks     : {len(plan.tasks)} ({tasks_summary})")
            else:
                print(f"   ├─ Tasks     : Ninguna")

            print(f"   └─ Confianza : {plan.confidence:.2f}")

    elif agent == "researcher":
        findings = event.get("research_findings", "")
        sql_results = event.get("sql_results", [])
        print(f"\n🔬 [Ciclo {iteration}] RESEARCHER")
        print(f"   ├─ Queries ejecutadas : {len(sql_results)}")
        print(f"   ├─ Findings preview   : {findings[:180]}{'...' if len(findings) > 180 else ''}")
        print(f"   └─ Informe generado   : {'Sí' if findings else 'No'}")

    elif agent == "forecaster":
        forecasts = event.get("forecast_results", [])
        forecast_error = event.get("forecast_error")
        print(f"\n🏭 [Ciclo {iteration}] FORECASTER")
        if forecast_error:
            print(f"   └─ Error: {forecast_error}")
        else:
            print(f"   └─ Pronóstico generado: {len(forecasts)} días")

    elif agent == "sql_agent":
        results = event.get("sql_results", [])
        print(f"\n🔍 [Ciclo {iteration}] SQL AGENT ({len(results)} tarea(s))")
        for contract in results:
            sql_clean = (contract.generated_sql or "N/A").replace("\n", " ").strip()
            print(f"   ├─ Tarea {contract.task_id}")
            print(f"   │  ├─ Status: {contract.status}")
            print(f"   │  ├─ SQL   : {sql_clean}")
            print(f"   │  ├─ Filas : {contract.row_count}")
            print(f"   │  └─ CanAns: {contract.can_answer}")
            if contract.rows:
                for i, row in enumerate(contract.rows[:3]):
                    print(f"   │     📊 Fila {i+1}: {json.dumps(row, ensure_ascii=False, default=_json_default)}")
                if len(contract.rows) > 3:
                    print(f"   │     ... y {len(contract.rows) - 3} filas más")
            else:
                print(f"   │     ⚠️ Sin filas")
        print(f"   └─ Todas las tareas evaluadas.")

    elif agent == "viz_agent":
        viz = event.get("viz_result")
        print(f"\n📈 [Ciclo {iteration}] VIZ AGENT")
        if viz:
            print(f"   └─ Chart: {getattr(viz, 'chart_type', 'N/A')} | "
                  f"Título: {getattr(viz, 'title', 'N/A')}")

    elif agent == "render_plotly":
        print(f"\n🎨 [Ciclo {iteration}] RENDER PLOTLY → figura renderizada")

    elif agent == "viz_approval":
        print(f"\n✅ [Ciclo {iteration}] VIZ APPROVAL → esperando confirmación")

    elif agent == "analyst":
        answer = event.get("final_answer", "")
        print(f"\n📊 [Ciclo {iteration}] ANALYST")
        preview = answer[:150] + "..." if len(answer) > 150 else answer
        print(f"   └─ Respuesta : {preview}")

    elif agent == "supervisor":
        print(f"\n🧠 [Ciclo {iteration}] SUPERVISOR → reevaluando...")


# ============================================================================
# 5. EJECUCIÓN DEL ORQUESTADOR (main.py adaptado)
# ============================================================================

def run_bi_query(question: str, thread_id: str = "cli-session-001", silent: bool = False) -> Optional[str]:
    initial_state = {
        "question": question,
        "messages": [HumanMessage(content=question)],
        "plan": None,
        "sql_results": [],
        "viz_result": None,
        "viz_approved": None,
        "viz_rendered": False,
        "final_answer": None,
        "iteration_count": 0,
        "last_agent": None,
        "harness_context": None,
        "semantic_context": "",
        "allowed_views": [],
        "preferred_view": None,
        "schema_info": "",
        "research_findings": None,
        "forecast_request": None,
        "forecast_results": None,
        "forecast_error": None,
    }

    config = RunnableConfig(
        configurable={"thread_id": thread_id},
        recursion_limit=100
    )

    if not silent:
        print_header(f"🚀 EJECUTANDO: {question[:60]}{'...' if len(question) > 60 else ''}")

    final_state = None
    events_seen = 0

    try:
        for event in BI_ORCHESTRATOR.stream(initial_state, config, stream_mode="values"):
            events_seen += 1
            agent = event.get("last_agent")
            iteration = event.get("iteration_count", 0)
            bi_logger.debug(f"Evento {events_seen} | Agente: {agent} | Iter: {iteration}")
            if not silent:
                print_event(agent, iteration, event)
            final_state = event

    except Exception as e:
        bi_logger.error(f"Error durante la ejecución del grafo: {e}", exc_info=True)
        if not silent:
            print(f"\n❌ Error en la ejecución: {e}")
        return None

    if final_state is None:
        if not silent:
            print("\n❌ No se obtuvo estado final del orquestador.")
        return None

    final_answer = final_state.get("final_answer")
    total_iterations = final_state.get("iteration_count", 0)

    if not silent:
        print_header("RESPUESTA FINAL")
        if final_answer:
            print(final_answer)
        else:
            print("No se pudo generar una respuesta.")
        print(f"\n📈 Estadísticas: {total_iterations} iteraciones | {events_seen} eventos")

    return final_answer


# ============================================================================
# 6. AUXILIARES DE SESIÓN, ESTADO Y SSE
# ============================================================================

class QueryRequest(BaseModel):
    question: str = Field(..., description="Pregunta del usuario")


def get_or_create_thread_id(user_id: str) -> str:
    if user_id not in SESSIONS:
        SESSIONS[user_id] = {"thread_id": f"api-{user_id}-{uuid4().hex[:8]}"}
    return SESSIONS[user_id]["thread_id"]


def reset_thread_id(user_id: str) -> str:
    SESSIONS[user_id] = {"thread_id": f"api-{user_id}-{uuid4().hex[:8]}"}
    return SESSIONS[user_id]["thread_id"]


def _build_initial_state(question: str) -> dict:
    return {
        "question": question,
        "messages": [HumanMessage(content=question)],
        "plan": None,
        "sql_results": [],
        "viz_result": None,
        "viz_approved": None,
        "viz_rendered": False,
        "final_answer": None,
        "iteration_count": 0,
        "last_agent": None,
        "harness_context": None,
        "semantic_context": "",
        "allowed_views": [],
        "preferred_view": None,
        "schema_info": "",
        "research_findings": None,
        "forecast_request": None,
        "forecast_results": None,
        "forecast_error": None,
    }


def sse_event(event_type: str, **payload) -> str:
    return f"data: {json.dumps({'type': event_type, **payload}, ensure_ascii=False)}\n\n"


def _node_message(node: Optional[str]) -> str:
    return {
        "planner": "Analizando tu pregunta...",
        "researcher": "Buscando contexto de negocio...",
        "sql_agent": "Consultando la base de datos...",
        "viz_agent": "Diseñando visualización...",
        "render_plotly": "Generando gráfico...",
        "analyst": "Redactando la respuesta...",
        "supervisor": "Revisando la respuesta...",
    }.get(node, f"Procesando ({node})...")


async def sse_stream_text(text: Optional[str], sleep_time: float = 0.003):
    if not text:
        return
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    for i, sentence in enumerate(sentences):
        sep = " " if i < len(sentences) - 1 else ""
        yield sse_event("chunk", content=sentence + sep)
        await asyncio.sleep(sleep_time)


def detect_yes_no_response(text: str) -> Optional[bool]:
    text_lower = text.lower().strip()
    yes_words = ['sí', 'si', 'yes', 'y', 'ok', 'dale', 'claro', 'por supuesto',
                 'obvio', 'desde luego', 'adelante', 'confirmo', 'afirmativo']
    no_words = ['no', 'nop', 'nope', 'negativo', 'mejor no', 'no gracias',
                'paso', 'cancel', 'cancelar']
    if any(text_lower == w or text_lower.startswith(w + " ") for w in yes_words):
        return True
    if any(text_lower == w or text_lower.startswith(w + " ") for w in no_words):
        return False
    return None


def publish_generated_chart(user_id: str) -> Optional[str]:
    possible_paths = [
        AGENTE_BI_PATH / "chart.png",
        BACKEND_DIR / "chart.png",
        Path(os.getcwd()) / "chart.png",
    ]
    for src in possible_paths:
        if src.exists() and src.stat().st_size > 0:
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_uid = re.sub(r'\W+', '', str(user_id))[:20]
                fname = f"chart_{safe_uid}_{ts}_{uuid4().hex[:4]}.png"
                dest = CHARTS_DIR / fname
                shutil.copy2(str(src), str(dest))
                time.sleep(0.05)
                if not dest.exists() or dest.stat().st_size == 0:
                    return None
                src.unlink(missing_ok=True)
                return f"/files/charts/{fname}"
            except Exception as e:
                logger.error(f"Error publicando chart: {e}")
    return None


async def _publish_chart(user_id: str, base_url: str) -> Optional[str]:
    loop = asyncio.get_event_loop()
    path = await loop.run_in_executor(None, publish_generated_chart, user_id)
    if path:
        return f"{base_url}{path}"
    return None


# ============================================================================
# 7. ENDPOINTS
# ============================================================================

@app.get("/api/debug/routes")
async def debug_routes():
    routes = []
    for r in app.routes:
        routes.append({
            "path": r.path,
            "name": r.name,
            "methods": list(r.methods) if hasattr(r, "methods") else []
        })
    return {"total": len(routes), "routes": [r for r in routes if r["path"].startswith("/api")]}


@app.get("/api/debug/config")
async def debug_config(user: User = Depends(get_current_user)):
    return {
        "frontend_url": FRONTEND_URL,
        "user": user.model_dump(),
        "db_configured": bool(SUPABASE_DB_URI),
        "agente_bi": AGENTE_BI_AVAILABLE,
        "thread_id": SESSIONS.get(user.id, {}).get("thread_id"),
    }


@app.get("/api/debug/ping")
async def debug_ping():
    return {"status": "ok"}


@app.get("/api/v1/system/status")
async def get_system_status(user: User = Depends(get_current_user)):
    return JSONResponse(content={
        "status": "ok",
        "capabilities": {
            "extractor_available": False,
            "supabase_available": bool(supabase),
            "multiagent_available": AGENTE_BI_AVAILABLE,
            "version": "6.0.0"
        }
    })


@app.post("/api/v1/session/clear")
async def clear_session(user: User = Depends(get_current_user)):
    cleared = []
    if user.id in PENDING_ACTIONS:
        del PENDING_ACTIONS[user.id]
        cleared.append("pending_actions")
    if user.id in SESSIONS:
        reset_thread_id(user.id)
        cleared.append("session_data")
    return {
        "success": True,
        "cleared_items": cleared,
        "thread_id": SESSIONS.get(user.id, {}).get("thread_id"),
    }


@app.post("/api/v1/chat/stream")
async def stream_chat(
    request: Request,
    body: QueryRequest,
    user: User = Depends(get_current_user)
):
    if not AGENTE_BI_AVAILABLE or BI_ORCHESTRATOR is None:
        raise HTTPException(status_code=503, detail="Agente BI no disponible")

    base_url = str(request.base_url).rstrip('/')
    thread_id = get_or_create_thread_id(user.id)

    async def async_graph_stream() -> AsyncGenerator[str, None]:
        initial_state = _build_initial_state(body.question)
        config = {"configurable": {"thread_id": thread_id}}

        yield sse_event("start")

        last_agent = None
        final_answer: Optional[str] = None
        chart_url: Optional[str] = None
        chart_emitted = False
        final_answer_emitted = False

        async for state in BI_ORCHESTRATOR.astream(
            initial_state, config, stream_mode="values"
        ):
            agent = state.get("last_agent")
            iteration = state.get("iteration_count", 0)

            if agent and agent != last_agent:
                logger.debug(f"Graph node → {agent} (iter {iteration})")
                yield sse_event(
                    "progress",
                    node=agent,
                    iteration=iteration,
                    message=_node_message(agent),
                )
                last_agent = agent

            if not chart_emitted and (
                state.get("viz_rendered") or agent in ("render_plotly", "viz_agent")
            ):
                chart_url = await _publish_chart(user.id, base_url)
                if chart_url:
                    yield sse_event("chart", url=chart_url, format="png")
                    chart_emitted = True

            answer = state.get("final_answer")
            if answer and not final_answer_emitted:
                final_answer = answer
                async for chunk in sse_stream_text(final_answer, sleep_time=0.003):
                    yield chunk
                final_answer_emitted = True

        if not chart_emitted:
            chart_url = await _publish_chart(user.id, base_url)
            if chart_url:
                yield sse_event("chart", url=chart_url, format="png")
                chart_emitted = True

        follow_up = "¿Quieres que te proporcione información más detallada sobre esto?"
        async for chunk in sse_stream_text(follow_up, sleep_time=0.003):
            yield chunk

        PENDING_ACTIONS[user.id] = {
            "action_type": "ask_detailed",
            "result": {
                "response": final_answer,
                "chart_url": chart_url,
            },
            "timestamp": datetime.now().isoformat(),
        }

        yield sse_event("end", intent="BI_QUERY", success=True)


    async def sync_fallback_stream() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(
            None, lambda: run_bi_query(body.question, thread_id=thread_id, silent=True)
        )

        yield sse_event("start")
        yield sse_event("progress", node="agent", message="Procesando...")

        if response_text:
            async for chunk in sse_stream_text(response_text, sleep_time=0.003):
                yield chunk

        chart_url = await _publish_chart(user.id, base_url)
        if chart_url:
            yield sse_event("chart", url=chart_url, format="png")

        follow_up = "¿Quieres que te proporcione información más detallada sobre esto?"
        async for chunk in sse_stream_text(follow_up, sleep_time=0.003):
            yield chunk

        PENDING_ACTIONS[user.id] = {
            "action_type": "ask_detailed",
            "result": {
                "response": response_text,
                "chart_url": chart_url,
            },
            "timestamp": datetime.now().isoformat(),
        }

        yield sse_event("end", intent="BI_QUERY", success=True)


    async def event_generator() -> AsyncGenerator[str, None]:
        t_start = time.time()
        try:
            if user.id in PENDING_ACTIONS:
                pending = PENDING_ACTIONS[user.id]
                yes_no = detect_yes_no_response(body.question)

                if yes_no is not None:
                    action_type = pending["action_type"]
                    if action_type == "ask_detailed":
                        if yes_no:
                            async for chunk in sse_stream_text(
                                "📄 Generando información detallada... ✅ Listo.",
                                sleep_time=0.003,
                            ):
                                yield chunk
                            del PENDING_ACTIONS[user.id]
                            yield sse_event("end", intent="DETAILED_INFO", success=True)
                            return
                        else:
                            async for chunk in sse_stream_text(
                                "👍 Entendido. ¿Necesitas algo más?",
                                sleep_time=0.003,
                            ):
                                yield chunk
                            del PENDING_ACTIONS[user.id]
                            yield sse_event("end", intent="CONVERSATION_END", success=True)
                            return
                else:
                    del PENDING_ACTIONS[user.id]

            if hasattr(BI_ORCHESTRATOR, "astream"):
                logger.info(f"▶️ Chat stream (astream) | user={user.id} | thread={thread_id}")
                async for payload in async_graph_stream():
                    yield payload
            else:
                logger.info(f"▶️ Chat stream (fallback sync) | user={user.id} | thread={thread_id}")
                async for payload in sync_fallback_stream():
                    yield payload

        except asyncio.CancelledError:
            logger.info(f"Cliente desconectado: {user.id}")
        except Exception as e:
            logger.error(f"Error streaming: {e}", exc_info=True)
            yield sse_event("error", content=str(e))
            yield sse_event("end", intent="ERROR", success=False)
        finally:
            logger.info(f"⏱️ Chat stream finalizado en {time.time() - t_start:.2f}s | user={user.id}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ============================================================================
# 8. MANEJO DE ERRORES Y LIFESPAN
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.now().isoformat()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Error no manejado: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "timestamp": datetime.now().isoformat()}
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🟢 API iniciada")
    if AGENTE_BI_AVAILABLE:
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, warmup_db)
            logger.info("✅ DB warm-up completado")
        except Exception as e:
            logger.warning(f"⚠️ DB warm-up no disponible: {e}")
    yield
    logger.info("🛑 API finalizada")

app.router.lifespan_context = lifespan

# MAIN
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
