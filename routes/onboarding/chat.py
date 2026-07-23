import os
import json
import time
import asyncio
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from core.session_context import session_scope, get_session_lock
from utils.sse_helpers import sse_event, sse_stream_text, detect_yes_no_response

router = APIRouter()

PENDING_ACTIONS: dict = {}


class ChatRequest(BaseModel):
    question: str
    messages: list[dict] = []


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
        "next": None,
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


def _node_message(node: str) -> str:
    return {
        "planner": "Analizando tu pregunta...",
        "build_harness": "Preparando contexto del negocio...",
        "researcher": "Buscando contexto de negocio...",
        "sql_agent": "Consultando la base de datos...",
        "viz_agent": "Diseñando visualización...",
        "render_plotly": "Generando gráfico...",
        "analyst": "Redactando la respuesta...",
        "supervisor": "Revisando la respuesta...",
        "forecaster": "Calculando pronóstico...",
    }.get(node, f"Procesando ({node})...")


async def _publish_chart(session_id: str, base_url: str) -> str | None:
    """Copia el chart generado a un archivo público y devuelve su URL."""
    backend_dir = Path(os.getenv("BACKEND_DIR", Path(__file__).resolve().parent.parent.parent))
    charts_dir = backend_dir / "files" / "charts"
    src = charts_dir / "chart.png"

    charts_dir.mkdir(parents=True, exist_ok=True)

    if src.exists() and src.stat().st_size > 0:
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_id = "".join(c for c in session_id if c.isalnum())[:20]
            fname = f"chart_{safe_id}_{ts}_{uuid4().hex[:4]}.png"
            dest = charts_dir / fname
            shutil.copy2(str(src), str(dest))
            public_url = os.getenv("BACKEND_PUBLIC_URL", base_url).rstrip('/')
            return f"{public_url}/files/charts/{fname}"
        except Exception:
            traceback.print_exc()
    return None


@router.post("/{session_id}/chat")
async def stream_chat(request: Request, session_id: str, body: ChatRequest):
    from core.orchestrator import BI_ORCHESTRATOR

    if BI_ORCHESTRATOR is None:
        raise HTTPException(status_code=503, detail="Agente BI no disponible")

    # ✅ Usar BACKEND_PUBLIC_URL si existe, sino request.base_url
    base_url = os.getenv("BACKEND_PUBLIC_URL", str(request.base_url)).rstrip('/')
    lock = get_session_lock(session_id)

    async def event_generator():
        async with lock:
            with session_scope(session_id):
                t_start = time.time()
                try:
                    question = body.question
                    yes_no = detect_yes_no_response(question)

                    if session_id in PENDING_ACTIONS and yes_no is not None:
                        if yes_no:
                            async for chunk in sse_stream_text("Generando información detallada..."):
                                yield chunk
                        else:
                            async for chunk in sse_stream_text("Entendido. ¿Necesitas algo más?"):
                                yield chunk
                        del PENDING_ACTIONS[session_id]
                        yield sse_event("end", intent="CONVERSATION_END", success=True)
                        return

                    if session_id in PENDING_ACTIONS:
                        del PENDING_ACTIONS[session_id]

                    initial_state = _build_initial_state(question)
                    config = {
                        "configurable": {"thread_id": session_id},
                        "recursion_limit": 50
                    }

                    yield sse_event("start")

                    last_agent = None
                    final_answer = None
                    chart_url = None
                    chart_emitted = False
                    final_answer_emitted = False

                    async for state in BI_ORCHESTRATOR.astream(
                        initial_state, config, stream_mode="values"
                    ):
                        agent = state.get("last_agent")
                        iteration = state.get("iteration_count", 0)

                        if agent and agent != last_agent:
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
                            chart_url = await _publish_chart(session_id, base_url)
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
                        chart_url = await _publish_chart(session_id, base_url)
                        if chart_url:
                            yield sse_event("chart", url=chart_url, format="png")
                            chart_emitted = True

                    follow_up = "¿Quieres que te proporcione información más detallada sobre esto?"
                    async for chunk in sse_stream_text(follow_up, sleep_time=0.003):
                        yield chunk

                    PENDING_ACTIONS[session_id] = {
                        "action_type": "ask_detailed",
                        "result": {"response": final_answer, "chart_url": chart_url},
                        "timestamp": time.time(),
                    }

                    yield sse_event("end", intent="BI_QUERY", success=True)

                except asyncio.CancelledError:
                    return
                except Exception as e:
                    traceback.print_exc()
                    yield sse_event("error", content=str(e))
                    yield sse_event("end", intent="ERROR", success=False)
                finally:
                    print(f"Chat stream finalizado en {time.time() - t_start:.2f}s | session={session_id}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
