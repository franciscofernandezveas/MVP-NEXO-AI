import os
import json
import time
import asyncio
import shutil
import traceback
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Any, Optional

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from core.session_context import session_scope, get_session_lock
from core.supabase_client import get_supabase_client
from utils.sse_helpers import sse_event, sse_stream_text, detect_yes_no_response
from utils.analytics import emit_event

router = APIRouter()

logger = logging.getLogger("routes.onboarding.chat")

PENDING_ACTIONS: dict = {}


class ChatRequest(BaseModel):
    question: str
    messages: list[dict] = []


def _get(obj: Any, attr: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


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
        "render_attempts": 0,
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


def _cleanup_residual_charts() -> None:
    files_dir = Path(os.getenv("FILES_DIR", "/app/files")).resolve()
    charts_dir = files_dir / "charts"
    if not charts_dir.exists():
        return
    for filename in ["chart.png", "chart.html"]:
        fp = charts_dir / filename
        if fp.exists():
            try:
                fp.unlink()
            except Exception as e:
                logger.warning(f"No se pudo eliminar residual {fp}: {e}")


async def _publish_chart(session_id: str, base_url: str, viz_result: Optional[Any] = None) -> str | None:
    chart_type = _get(viz_result, "chart_type")
    if not chart_type or chart_type == "null":
        return None

    files_dir = Path(os.getenv("FILES_DIR", "/app/files")).resolve()
    charts_dir = files_dir / "charts"
    src = charts_dir / "chart.png"
    charts_dir.mkdir(parents=True, exist_ok=True)

    if not src.exists() or src.stat().st_size == 0:
        return None

    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_id = "".join(c for c in session_id if c.isalnum())[:20]
        fname = f"chart_{safe_id}_{ts}_{uuid4().hex[:4]}.png"
        dest = charts_dir / fname
        shutil.copy2(str(src), str(dest))
        public_url = os.getenv("BACKEND_PUBLIC_URL", base_url).rstrip('/')
        return f"{public_url}/files/charts/{fname}"
    except Exception:
        logger.exception("Error copiando chart")
        return None


@router.post("/{session_id}/chat")
async def stream_chat(request: Request, session_id: str, body: ChatRequest):
    from core.orchestrator import BI_ORCHESTRATOR

    if BI_ORCHESTRATOR is None:
        raise HTTPException(status_code=503, detail="Agente BI no disponible")

    supabase = get_supabase_client()

    # Recuperar user_id/company_id de la sesión
    session_row = supabase.table("sessions")\
        .select("id, user_id, company_id, started_at")\
        .eq("id", session_id)\
        .maybe_single()\
        .execute()

    if not session_row.data:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    user_id = session_row.data.get("user_id")
    company_id = session_row.data.get("company_id")
    first_query_at = datetime.now(timezone.utc)

    # Guardar mensaje del usuario
    try:
        msg_user = supabase.table("messages").insert({
            "session_id": session_id,
            "user_id": user_id,
            "role": "user",
            "content": body.question,
            "message_type": "question",
        }).execute()
        user_message_id = msg_user.data[0]["id"] if msg_user.data else None
    except Exception as e:
        logger.warning(f"[chat] no se pudo guardar mensaje usuario: {e}")
        user_message_id = None

    await emit_event(
        "chat.message.sent",
        user_id=user_id,
        company_id=company_id,
        session_id=session_id,
        payload={"message_type": "question", "content_length": len(body.question)},
    )

    base_url = os.getenv("BACKEND_PUBLIC_URL", str(request.base_url)).rstrip('/')
    lock = get_session_lock(session_id)

    async def event_generator():
        async with lock:
            with session_scope(session_id):
                t_start = time.time()
                response_time_ms = 0
                final_answer = None
                chart_url = None
                chart_emitted = False
                final_answer_emitted = False
                final_state = None
                assistant_message_id = None
                response_id = None
                sql_generated = None
                sql_success = None
                error_type = None
                has_viz = False

                try:
                    _cleanup_residual_charts()
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

                    async for state in BI_ORCHESTRATOR.astream(
                        initial_state, config, stream_mode="values"
                    ):
                        final_state = state
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

                        answer = state.get("final_answer")
                        if answer and not final_answer_emitted:
                            final_answer = answer
                            async for chunk in sse_stream_text(final_answer, sleep_time=0.003):
                                yield chunk
                            final_answer_emitted = True

                    response_time_ms = int((time.time() - t_start) * 1000)

                    if final_state:
                        viz_rendered = final_state.get("viz_rendered", False)
                        viz_result = final_state.get("viz_result") or {}
                        sql_results = final_state.get("sql_results") or []
                        has_viz = viz_rendered and bool(_get(viz_result, "chart_type"))

                        if has_viz:
                            chart_url = await _publish_chart(session_id, base_url, viz_result)
                            if chart_url:
                                yield sse_event("chart", url=chart_url, format="png")
                                chart_emitted = True

                        # Detectar SQL generado / éxito
                        if sql_results:
                            sql_generated = json.dumps(sql_results) if isinstance(sql_results, (list, dict)) else str(sql_results)
                            sql_success = not any(r.get("error") for r in sql_results if isinstance(r, dict))

                    # Guardar mensaje del asistente
                    try:
                        if final_answer:
                            msg_assistant = supabase.table("messages").insert({
                                "session_id": session_id,
                                "user_id": user_id,
                                "role": "assistant",
                                "content": final_answer[:4000],
                                "message_type": "answer",
                            }).execute()
                            assistant_message_id = msg_assistant.data[0]["id"] if msg_assistant.data else None

                            # Guardar respuesta del agente
                            ar_result = supabase.table("agent_responses").insert({
                                "message_id": assistant_message_id,
                                "session_id": session_id,
                                "user_id": user_id,
                                "response_time_ms": response_time_ms,
                                "sql_generated": sql_generated,
                                "sql_executed": sql_generated is not None,
                                "sql_success": sql_success,
                                "error_type": error_type,
                                "has_visualization": chart_emitted,
                                "has_dashboard": False,
                            }).execute()
                            response_id = ar_result.data[0]["id"] if ar_result.data else None
                    except Exception as e:
                        logger.warning(f"[chat] no se pudo guardar respuesta agente: {e}")

                    # Eventos de calidad
                    await emit_event(
                        "agent.response.generated",
                        user_id=user_id,
                        company_id=company_id,
                        session_id=session_id,
                        payload={
                            "response_time_ms": response_time_ms,
                            "sql_generated": sql_generated is not None,
                            "has_visualization": chart_emitted,
                            "has_dashboard": False,
                        },
                    )

                    if sql_generated:
                        await emit_event(
                            "agent.sql.executed",
                            user_id=user_id,
                            company_id=company_id,
                            session_id=session_id,
                            payload={
                                "success": bool(sql_success),
                                "error_type": error_type,
                                "rows_returned": len(sql_results) if isinstance(sql_results, list) else 0,
                            },
                        )

                    # Insight: cuando hay respuesta final con datos/visualización
                    if final_answer and (chart_emitted or sql_generated):
                        try:
                            query_count = 1  # en este MVP contamos la pregunta actual
                            reached_at = datetime.now(timezone.utc)
                            seconds_to_insight = int((reached_at - first_query_at).total_seconds())

                            supabase.table("insights").insert({
                                "session_id": session_id,
                                "user_id": user_id,
                                "company_id": company_id,
                                "first_query_at": first_query_at.isoformat(),
                                "insight_reached_at": reached_at.isoformat(),
                                "query_count": query_count,
                                "successful": True,
                                "self_served": True,
                                "description": final_answer[:500],
                            }).execute()

                            await emit_event(
                                "insight.reached",
                                user_id=user_id,
                                company_id=company_id,
                                session_id=session_id,
                                payload={
                                    "seconds_to_insight": seconds_to_insight,
                                    "query_count": query_count,
                                    "self_served": True,
                                },
                            )
                        except Exception as e:
                            logger.warning(f"[chat] no se pudo guardar insight: {e}")

                    if not chart_emitted:
                        logger.info("[Chat] No se generó visualización en esta ejecución.")

                    follow_up = "¿Quieres que te proporcione información más detallada sobre esto?"
                    async for chunk in sse_stream_text(follow_up, sleep_time=0.003):
                        yield chunk

                    PENDING_ACTIONS[session_id] = {
                        "action_type": "ask_detailed",
                        "result": {"response": final_answer, "chart_url": chart_url},
                        "timestamp": time.time(),
                    }

                    yield sse_event(
                        "end",
                        intent="BI_QUERY",
                        success=True,
                        response_id=response_id,
                        assistant_message_id=assistant_message_id,
                    )

                except asyncio.CancelledError:
                    return
                except Exception as e:
                    traceback.print_exc()
                    error_type = type(e).__name__
                    await emit_event(
                        "agent.response.error",
                        user_id=user_id,
                        company_id=company_id,
                        session_id=session_id,
                        payload={"error_type": error_type, "error_message": str(e)},
                    )
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
