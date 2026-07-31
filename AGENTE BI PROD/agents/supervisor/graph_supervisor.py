# agents/supervisor/graph_supervisor.py
# -------------------------------------------------

# agents/supervisor/graph_supervisor.py
# -------------------------------------------------

import json
from typing import Any, Dict, List, Literal, Optional
from langchain_core.messages import AIMessage, HumanMessage
from core.llm import LLM
from core.contracts import SupervisorDecision
from core.config import logger

from langsmith import traceable


MAX_ITERATIONS = 30


def _get(obj: Any, attr: str, default: Any = None) -> Any:
    """
    Lee atributos de objetos Pydantic o claves de diccionarios.
    Esencial porque LangGraph serializa el estado a dicts entre nodos.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _is_new_topic(question: str, previous_question: Optional[str]) -> bool:
    """
    Heurística simple para detectar si el usuario quiere cambiar de tema.
    Útil para reiniciar el plan en el mismo thread cuando el contexto previo
    ya no es relevante.
    """
    if not previous_question:
        return False

    new_topic_markers = [
        "ahora", "en cambio", "otra cosa", "diferente", "nueva pregunta",
        "cambia de tema", "hablamos de otra cosa", "olvidate de eso", "olvida eso"
    ]
    return any(marker in question.lower() for marker in new_topic_markers)


def _extract_last_user_question(messages: List[Any]) -> Optional[str]:
    """Extrae la última pregunta del usuario del historial."""
    if not messages:
        return None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return getattr(msg, "content", None)
    return None


@traceable(name="Supervisor: Route Next Agent")
def supervisor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    current_iter = state.get("iteration_count", 0)
    next_iter = current_iter + 1
    render_attempts = state.get("render_attempts", 0)

    def make_update(next_node: str, **extra) -> Dict[str, Any]:
        base = {
            "next": next_node,
            "last_agent": "supervisor",
            "iteration_count": next_iter,
            "render_attempts": render_attempts,
        }
        base.update(extra)
        return base

    # 1. Límite de iteraciones
    if current_iter >= MAX_ITERATIONS:
        logger.warning(f"[Supervisor] Límite de iteraciones ({MAX_ITERATIONS}) alcanzado.")
        return make_update(
            "__end__",
            final_answer=state.get("final_answer") or "No pude completar la consulta tras varios intentos.",
            messages=[AIMessage(content=f"[Supervisor] Iteración límite ({MAX_ITERATIONS}) alcanzado. Cerrando.")]
        )

    # Lectura de estado
    plan = state.get("plan")
    sql_results = state.get("sql_results")
    viz_result = state.get("viz_result")
    viz_approved = state.get("viz_approved")
    viz_rendered = state.get("viz_rendered", False)
    final_answer = state.get("final_answer")
    last_agent = state.get("last_agent")
    forecast_results = state.get("forecast_results")
    forecast_error = state.get("forecast_error")
    question = state.get("question", "")
    messages = state.get("messages", [])

    # NUEVO: Detectar cambio de tema y reiniciar plan si aplica
    previous_question = _extract_last_user_question(messages[:-1]) if messages else None
    if plan and previous_question and _is_new_topic(question, previous_question):
        logger.info(
            f"[Supervisor] Detectado cambio de tema ('{question}' vs '{previous_question}'). "
            f"Reiniciando planificación."
        )
        return make_update(
            "planner",
            plan=None,
            sql_results=[],
            viz_result=None,
            viz_approved=None,
            viz_rendered=False,
            final_answer=None,
            research_findings=None,
            forecast_request=None,
            forecast_results=None,
            forecast_error=None,
            messages=messages + [
                AIMessage(content="[Supervisor] Nuevo tema detectado; reiniciando planificación.")
            ]
        )

    # === ANTI-LOOP: Validación de viz_result ===
    viz_is_valid = False
    if viz_result:
        has_chart_type = _get(viz_result, "chart_type") is not None
        is_success = _get(viz_result, "status") == "success"
        suitable = _get(viz_result, "suitable_for_visualization", False)
        viz_is_valid = has_chart_type and is_success and suitable

    if viz_result and not viz_is_valid and not viz_rendered:
        logger.info(
            f"[Supervisor] Viz result inválido "
            f"(chart_type={_get(viz_result, 'chart_type')}, "
            f"status={_get(viz_result, 'status')}, "
            f"suitable={_get(viz_result, 'suitable_for_visualization')}). "
            f"Saltando renderización."
        )
        return make_update(
            "analyst",
            viz_rendered=True,
            messages=[AIMessage(content="[Supervisor] Spec de visualización inválida, se omite render y se continúa.")]
        )

    # === ANTI-LOOP RENDER ===
    if last_agent == "render_plotly":
        if not viz_rendered:
            if render_attempts >= 1:
                logger.warning("[Supervisor] Render ya falló una vez. Forzando avance a analyst.")
                return make_update(
                    "analyst",
                    viz_rendered=True,
                    messages=[AIMessage(content="[Supervisor] Render no disponible, se omite visualización.")]
                )
            logger.info(f"[Supervisor] Reintentando render (intento {render_attempts + 1})")
            return make_update(
                "render_plotly",
                render_attempts=render_attempts + 1,
                messages=[AIMessage(content="[Supervisor] Reintentando render...")]
            )
        else:
            logger.info("[Supervisor] Render completado exitosamente.")

    if last_agent == "render_plotly" and not viz_is_valid:
        logger.info("[Supervisor] Prevención de loop post-render fallido. Forzando avance.")
        return make_update(
            "analyst",
            viz_rendered=True,
            messages=[AIMessage(content="[Supervisor] Render previo no aplicable, continuando.")]
        )

    # === RUTAS DE FLUJO ===

    # 1. Sin plan → Planner
    if not plan:
        logger.info("[Supervisor] Ruteando a Planner - no hay plan")
        return make_update("planner")

    # ------------------------------------------------------------------
    # 2. Ruta de predicción de demanda (PRIORIDAD MÁXIMA)
    # ------------------------------------------------------------------
    if _get(plan, "question_type") == "demand_forecast":
        if not forecast_results and not forecast_error:
            logger.info("[Supervisor] Ruteando a Forecaster - predicción de demanda solicitada")
            return make_update("forecaster")

        if forecast_error and not final_answer:
            logger.info("[Supervisor] Forecast falló, ruteando a Analyst para explicar el error")
            return make_update("analyst")

        if forecast_results and not final_answer:
            logger.info("[Supervisor] Forecast completado, ruteando a Analyst para redactar respuesta")
            return make_update("analyst")

        logger.info("[Supervisor] Forecast y respuesta final listos, finalizando")
        return make_update("__end__")

    # ------------------------------------------------------------------
    # 3. Ruta de informe profundo
    # ------------------------------------------------------------------
    if (
        _get(plan, "question_type") == "deep_research"
        and state.get("research_findings") is None
    ):
        logger.info("[Supervisor] Ruteando a Researcher - informe profundo solicitado")
        return make_update("researcher")

    # 4. Con plan pero sin resultados SQL → SQL Agent
    if not sql_results:
        if last_agent == "sql_agent":
            logger.warning("[Supervisor] SQL Agent ya ejecutó pero no hay resultados. Forzando analyst.")
            return make_update(
                "analyst",
                messages=[AIMessage(
                    content="[Supervisor] SQL Agent no devolvió resultados tras intentar. "
                            "Se genera respuesta explicativa."
                )]
            )

        logger.info("[Supervisor] Ruteando a SQL Agent - no hay resultados")
        return make_update("sql_agent")

    # 5. Plan pide visualización y aún no hemos corrido el viz_agent
    if _get(plan, "visualization_candidate", False) and viz_result is None:
        logger.info("[Supervisor] Ruteando a Viz Agent - visualización solicitada (sin spec previa)")
        return make_update("viz_agent")

    # 6. Hay spec VÁLIDA y aún no renderizada → Render Plotly
    if viz_is_valid and not viz_rendered:
        logger.info("[Supervisor] Ruteando a Render Plotly")
        return make_update("render_plotly")

    # 7. Sin respuesta final → Analyst
    if not final_answer:
        logger.info("[Supervisor] Ruteando a Analyst - no hay respuesta final")
        return make_update("analyst")

    # 8. Viz ya resuelto y hay datos → Viz Approval (si aplica)
    if viz_is_valid and viz_rendered and viz_approved is None:
        suitable_results = [
            r for r in sql_results
            if _get(r, "can_answer", False) and len(_get(r, "rows", []) or []) > 0
        ]
        if suitable_results:
            logger.info("[Supervisor] Ruteando a Viz Approval")
            return make_update("viz_approval")

    # 9. Todo listo → Fin
    logger.info("[Supervisor] Todo completado, finalizando")
    return make_update("__end__")
