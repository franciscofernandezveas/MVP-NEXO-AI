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


def _safe_execute(query, silent: bool = True):
    """Ejecuta una query de supabase y devuelve (data, error)."""
    try:
        result = query.execute()
        if result is None:
            return None, "Supabase devolvió None"
        return getattr(result, "data", None), None
    except Exception as e:
        if not silent:
            logger.warning(f"[supabase] query error: {e}")
        return None, str(e)


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


def _sql_result_to_dict(result: Any) -> Any:
    """
    Convierte un SQLContract (u objeto Pydantic) a diccionario serializable.
    Si ya es dict/list, lo devuelve tal cual. Si falla, devuelve string.
    """
    if result is None:
        return None
    if isinstance(result, dict):
        return result
    if isinstance(result, list):
        return [_sql_result_to_dict(item) for item in result]

    # Pydantic v2 / v1 compatible
    if hasattr(result, "model_dump"):
        try:
            return result.model_dump(mode="json")
        except Exception:
            pass
    if hasattr(result, "dict"):
        try:
            return result.dict()
        except Exception:
            pass

    # Atributos públicos como dict
    try:
        return {
            k: _sql_result_to_dict(v)
            for k, v in result.__dict__.items()
            if not k.startswith("_")
        }
    except Exception:
        return str(result)


def _is_sql_success(sql_results: list) -> Optional[bool]:
    """
    Determina si la ejecución SQL fue exitosa inspeccionando los contratos.
    """
    if not sql_results:
        return None
    for r in sql_results:
        if isinstance(r, dict):
            if r.get("error_message") or r.get("error"):
                return False
        else:
            err = _get(r, "error_message") or _get(r, "error")
            if err:
                return False
    return True


def _sql_rows_returned(sql_results: list) -> int:
    if not sql_results:
        return 0
    first = sql_results[0]
    if isinstance(first, dict):
        rows = first.get("rows") or first.get("row_count") or first.get("data") or []
        return len(rows) if isinstance(rows, list) else 0
    rows = _get(first, "rows") or _get(first, "row_count") or _get(first, "data") or []
    return len(rows) if isinstance(rows, list) else 0


def _extract_token(chunk: Any) -> str:
    """
    Extrae texto visible de un AIMessageChunk / ToolMessageChunk / dict / objeto.
    Ignora tool_call_chunks sin contenido textual.
    """
    if chunk is None:
        return ""

    if isinstance(chunk, dict):
        return chunk.get("content") or ""

    if hasattr(chunk, "content"):
        content = chunk.content
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        # A veces content es una lista de dicts (tool_call_chunks); filtrar texto
        texts = []
        for item in content if isinstance(content, list) else []:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    texts.append(text)
            elif isinstance(item, str):
                texts.append(item)
        return "".join(texts)

    return str(chunk) if chunk else ""


def _is_answer_node(node: Optional[str]) -> bool:
    """
    Heurística para decidir si los tokens de un nodo son la respuesta final
    visible para el usuario. Ajusta según los nombres reales de tu grafo.
    """
    if not node:
        return True  # Si no sabemos el nodo, no filtramos por seguridad
    visible = {
        "analyst",
        "supervisor",
        "generate",
        "final",
        "responder",
        "answer",
        "answer_agent",
        "response_agent",
    }
    return node in visible or "answer" in node.lower() or "final" in node.lower()


@router.post("/{session_id}/chat")
async def stream_chat(request: Request, session_id: str, body: ChatRequest):
    from core.orchestrator import BI_ORCHESTRATOR

    if BI_ORCHESTRATOR is None:
        raise HTTPException(status_code=503, detail="Agente BI no disponible")

    supabase = get_supabase_client()

    # Recuperar user_id/company_id de la sesión (robusto a respuestas None)
    session_data, session_err = _safe_execute(
        supabase.table("sessions")
        .select("id, user_id, company_id, started_at")
        .eq("id", session_id)
        .maybe_single()
    )

    if not session_data:
        logger.error(f"[chat] sesión no encontrada: {session_id} | error: {session_err}")
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    user_id = session_data.get("user_id")
    company_id = session_data.get("company_id")
    first_query_at = datetime.now(timezone.utc)

    # Guardar mensaje del usuario
    user_message_id = None
    try:
        msg_data, _ = _safe_execute(
            supabase.table("messages").insert({
                "session_id": session_id,
                "user_id": user_id,
                "role": "user",
                "content": body.question,
                "message_type": "question",
            })
        )
        if msg_data and isinstance(msg_data, list) and len(msg_data) > 0:
            user_message_id = msg_data[0]["id"]
    except Exception as e:
        logger.warning(f"[chat] no se pudo guardar mensaje usuario: {e}")

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
                final_state = None
                assistant_message_id = None
                response_id = None
                sql_generated = None
                sql_success = None
                error_type = None
                sql_results_serializable = None
                streamed_tokens: list[str] = []

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
                        "recursion_limit": 100
                    }

                    yield sse_event("start")
                    last_agent = None

                    # ═══════════════════════════════════════════════════════
                    # STREAMING REAL DE TOKENS (LangGraph 0.1.x)
                    # ═══════════════════════════════════════════════════════
                    events_supported = True
                    try:
                        # Verificamos que el orquestador soporte astream_events
                        _ = BI_ORCHESTRATOR.astream_events
                    except AttributeError:
                        events_supported = False
                        logger.warning(
                            "[chat] BI_ORCHESTRATOR no expone astream_events; "
                            "fallback a stream_mode='values'"
                        )

                    if events_supported:
                        async for event in BI_ORCHESTRATOR.astream_events(
                            initial_state, config, version="v2"
                        ):
                            # Cliente cerró la conexión: cancelar inmediatamente
                            if await request.is_disconnected():
                                logger.info(f"[chat] cliente desconectado durante stream: {session_id}")
                                break

                            kind = event.get("event")
                            data = event.get("data", {})
                            metadata = event.get("metadata", {})
                            node = metadata.get("langgraph_node")

                            # ── Progreso de nodos ──
                            if kind == "on_chain_start" and node and node != last_agent:
                                yield sse_event(
                                    "progress",
                                    node=node,
                                    iteration=0,
                                    message=_node_message(node),
                                )
                                last_agent = node

                            # ── Tokens reales del modelo de lenguaje ──
                            if kind == "on_chat_model_stream":
                                token = _extract_token(data.get("chunk"))
                                if token and _is_answer_node(node):
                                    yield sse_event("chunk", content=token)
                                    streamed_tokens.append(token)

                            # ── Capturar estado final si está disponible ──
                            if (
                                kind == "on_chain_end"
                                and event.get("name") in ("LangGraph", "__root__", None, "")
                            ):
                                output = data.get("output")
                                if isinstance(output, dict):
                                    final_state = output

                        # Si astream_events no devolvió estado final, fallback a values
                        if final_state is None:
                            async for state in BI_ORCHESTRATOR.astream(
                                initial_state, config, stream_mode="values"
                            ):
                                final_state = state
                                if await request.is_disconnected():
                                    break

                    else:
                        # Fallback: solo snapshots de estado + final_answer
                        async for state in BI_ORCHESTRATOR.astream(
                            initial_state, config, stream_mode="values"
                        ):
                            if await request.is_disconnected():
                                break
                            final_state = state
                            agent = state.get("last_agent")
                            if agent and agent != last_agent:
                                yield sse_event(
                                    "progress",
                                    node=agent,
                                    iteration=0,
                                    message=_node_message(agent),
                                )
                                last_agent = agent

                    response_time_ms = int((time.time() - t_start) * 1000)

                    # Recuperar respuesta final del estado
                    if final_state:
                        final_answer = final_state.get("final_answer")
                        viz_rendered = final_state.get("viz_rendered", False)
                        viz_result = final_state.get("viz_result") or {}
                        sql_results = final_state.get("sql_results") or []

                        sql_results_serializable = _sql_result_to_dict(sql_results)

                        if sql_results:
                            try:
                                sql_generated = json.dumps(sql_results_serializable)
                            except Exception:
                                sql_generated = str(sql_results_serializable)
                            sql_success = _is_sql_success(sql_results)
                            error_type = None

                        has_viz = viz_rendered and bool(_get(viz_result, "chart_type"))

                        if has_viz:
                            chart_url = await _publish_chart(session_id, base_url, viz_result)
                            if chart_url:
                                yield sse_event("chart", url=chart_url, format="png")
                                chart_emitted = True

                    # Si NO logramos stream de tokens reales, enviar final_answer como typing
                    # (para no romper el fallback y mantener compatibilidad)
                    if not streamed_tokens and final_answer:
                        async for chunk in sse_stream_text(final_answer, sleep_time=0.003):
                            yield chunk

                    # Si hubo tokens reales pero el final_answer difiere (p.ej. prefijo/sufijo),
                    # podemos enviar la diferencia. Por ahora confiamos en los tokens.

                    # Guardar mensaje del asistente
                    answer_to_save = final_answer
                    if not answer_to_save and streamed_tokens:
                        answer_to_save = "".join(streamed_tokens)

                    if answer_to_save:
                        try:
                            msg_data, _ = _safe_execute(
                                supabase.table("messages").insert({
                                    "session_id": session_id,
                                    "user_id": user_id,
                                    "role": "assistant",
                                    "content": answer_to_save[:4000],
                                    "message_type": "answer",
                                })
                            )
                            if msg_data and isinstance(msg_data, list) and len(msg_data) > 0:
                                assistant_message_id = msg_data[0]["id"]

                            ar_data, _ = _safe_execute(
                                supabase.table("agent_responses").insert({
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
                                })
                            )
                            if ar_data and isinstance(ar_data, list) and len(ar_data) > 0:
                                response_id = ar_data[0]["id"]
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
                                "rows_returned": _sql_rows_returned(sql_results or []),
                            },
                        )

                    # Insight
                    if answer_to_save and (chart_emitted or sql_generated):
                        try:
                            query_count = 1
                            reached_at = datetime.now(timezone.utc)
                            seconds_to_insight = int((reached_at - first_query_at).total_seconds())

                            _safe_execute(
                                supabase.table("insights").insert({
                                    "session_id": session_id,
                                    "user_id": user_id,
                                    "company_id": company_id,
                                    "first_query_at": first_query_at.isoformat(),
                                    "insight_reached_at": reached_at.isoformat(),
                                    "query_count": query_count,
                                    "successful": True,
                                    "self_served": True,
                                    "description": answer_to_save[:500],
                                })
                            )

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
                        "result": {"response": answer_to_save, "chart_url": chart_url},
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
                    logger.info(f"Chat stream finalizado en {time.time() - t_start:.2f}s | session={session_id}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
