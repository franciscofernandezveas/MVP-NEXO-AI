# agents/supervisor/graph_supervisor.py
# -------------------------------------------------

from typing import Any, Dict, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from core.config import logger

from langsmith import traceable


def _get(obj: Any, attr: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _is_new_topic(question: str, previous_question: Optional[str]) -> bool:
    if not previous_question:
        return False

    new_topic_markers = [
        "ahora", "en cambio", "otra cosa", "diferente", "nueva pregunta",
        "cambia de tema", "hablamos de otra cosa", "olvidate de eso", "olvida eso"
    ]
    return any(marker in question.lower() for marker in new_topic_markers)


def _extract_last_user_question(messages: List[Any]) -> Optional[str]:
    if not messages:
        return None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return getattr(msg, "content", None)
    return None


@traceable(name="Supervisor: Route Next Agent")
def supervisor_node(state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    render_attempts = state.get("render_attempts", 0)

    def make_update(next_node: str, **extra) -> Dict[str, Any]:
        base = {
            "next": next_node,
            "last_agent": "supervisor",
            "render_attempts": render_attempts,
        }
        base.update(extra)
        return base

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

    # Detectar cambio de tema y reiniciar plan si aplica
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
            next_agent_instruction=(
                "El usuario cambió de tema. Genera un plan completamente nuevo "
                "para la nueva pregunta, descartando el contexto anterior."
            ),
            messages=[
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
            next_agent_instruction=(
                "La especificación de visualización no fue viable. "
                "Genera una respuesta basada únicamente en los datos SQL disponibles."
            ),
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
                    next_agent_instruction=(
                        "La renderización de la visualización falló. "
                        "Continúa con la respuesta textual basada en los datos SQL."
                    ),
                    messages=[AIMessage(content="[Supervisor] Render no disponible, se omite visualización.")]
                )
            logger.info(f"[Supervisor] Reintentando render (intento {render_attempts + 1})")
            return make_update(
                "render_plotly",
                render_attempts=render_attempts + 1,
                next_agent_instruction="Reintentar la renderización de la visualización en Plotly.",
                messages=[AIMessage(content="[Supervisor] Reintentando render...")]
            )
        else:
            logger.info("[Supervisor] Render completado exitosamente.")

    if last_agent == "render_plotly" and not viz_is_valid:
        logger.info("[Supervisor] Prevención de loop post-render fallido. Forzando avance.")
        return make_update(
            "analyst",
            viz_rendered=True,
            next_agent_instruction="Continúa al analista tras fallo de render previo.",
            messages=[AIMessage(content="[Supervisor] Render previo no aplicable, continuando.")]
        )

    # === RUTAS DE FLUJO ===

    # 1. Sin plan → Planner
    if not plan:
        logger.info("[Supervisor] Ruteando a Planner - no hay plan")
        return make_update(
            "planner",
            next_agent_instruction="Genera un plan de ejecución para responder la pregunta del usuario."
        )

    # ------------------------------------------------------------------
    # 2. Ruta de predicción de demanda (PRIORIDAD MÁXIMA)
    # ------------------------------------------------------------------
    if _get(plan, "question_type") == "demand_forecast":
        if not forecast_results and not forecast_error:
            logger.info("[Supervisor] Ruteando a Forecaster - predicción de demanda solicitada")
            return make_update(
                "forecaster",
                next_agent_instruction="Ejecuta el pronóstico de demanda con los parámetros estructurados del plan."
            )

        if forecast_error and not final_answer:
            logger.info("[Supervisor] Forecast falló, ruteando a Analyst para explicar el error")
            return make_update(
                "analyst",
                next_agent_instruction="Explica el error del pronóstico de demanda al usuario."
            )

        if forecast_results and not final_answer:
            logger.info("[Supervisor] Forecast completado, ruteando a Analyst para redactar respuesta")
            return make_update(
                "analyst",
                next_agent_instruction="Resume los resultados del pronóstico de demanda en lenguaje claro y accionable."
            )

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
        return make_update(
            "researcher",
            next_agent_instruction="Realiza una exploración profunda con múltiples queries SQL autocontenidas y genera un informe ejecutivo completo."
        )

    # ------------------------------------------------------------------
    # 4. Con plan: ejecutar subtareas SQL pendientes
    # ------------------------------------------------------------------
    plan_tasks = _get(plan, "tasks", []) or []

    if plan_tasks:
        # Índice de resultados existentes por task_id
        results_by_task: Dict[str, Any] = {}
        for r in (sql_results or []):
            tid = str(_get(r, "task_id", ""))
            if tid:
                results_by_task[tid] = r

        pending_tasks = []
        for t in plan_tasks:
            tid = str(_get(t, "task_id", ""))
            res = results_by_task.get(tid)

            # Si no hay resultado o no es útil, la tarea está pendiente
            if res is None:
                pending_tasks.append(t)
                continue

            status_ok = _get(res, "status") in ("success", "partial")
            useful = _get(res, "can_answer", False)
            if not (status_ok and useful):
                pending_tasks.append(t)

        if pending_tasks:
            pending_ids = [str(_get(t, "task_id")) for t in pending_tasks]
            logger.info(
                f"[Supervisor] Ruteando a SQL Agent - "
                f"Faltan {len(pending_tasks)} tareas por ejecutar/validar de {len(plan_tasks)} "
                f"(ids: {pending_ids})"
            )
            return make_update(
                "sql_agent",
                next_agent_instruction=(
                    f"Ejecuta las tareas pendientes del plan. "
                    f"Quedan {len(pending_tasks)} subtareas por procesar: {', '.join(pending_ids)}."
                )
            )

    # Fallback legacy: plan sin tasks explícitas
    if not sql_results:
        if last_agent == "sql_agent":
            logger.warning("[Supervisor] SQL Agent ya ejecutó pero no hay resultados. Forzando analyst.")
            return make_update(
                "analyst",
                next_agent_instruction="El SQL Agent no devolvió resultados tras intentar. Genera una respuesta explicativa.",
                messages=[AIMessage(
                    content="[Supervisor] SQL Agent no devolvió resultados tras intentar. "
                            "Se genera respuesta explicativa."
                )]
            )

        logger.info("[Supervisor] Ruteando a SQL Agent - no hay resultados")
        return make_update(
            "sql_agent",
            next_agent_instruction="Ejecuta las tareas SQL del plan actual y devuelve un contrato por cada una."
        )

    # 5. Plan pide visualización y aún no hemos corrido el viz_agent
    if _get(plan, "visualization_candidate", False) and viz_result is None:
        logger.info("[Supervisor] Ruteando a Viz Agent - visualización solicitada (sin spec previa)")
        return make_update(
            "viz_agent",
            next_agent_instruction="Genera una especificación de visualización adecuada para los datos SQL obtenidos."
        )

    # 6. Hay spec VÁLIDA y aún no renderizada → Render Plotly
    if viz_is_valid and not viz_rendered:
        logger.info("[Supervisor] Ruteando a Render Plotly")
        return make_update(
            "render_plotly",
            next_agent_instruction="Renderiza la especificación de visualización en una figura Plotly."
        )

    # 7. Sin respuesta final → Analyst
    if not final_answer:
        logger.info("[Supervisor] Ruteando a Analyst - no hay respuesta final")
        return make_update(
            "analyst",
            next_agent_instruction="Genera la respuesta final en lenguaje natural a partir de los resultados disponibles."
        )

    # 8. Viz ya resuelto y hay datos → Viz Approval (si aplica)
    if viz_is_valid and viz_rendered and viz_approved is None:
        suitable_results = [
            r for r in sql_results
            if _get(r, "can_answer", False) and len(_get(r, "rows", []) or []) > 0
        ]
        if suitable_results:
            logger.info("[Supervisor] Ruteando a Viz Approval")
            return make_update(
                "viz_approval",
                next_agent_instruction="Presenta la visualización al usuario para aprobación o rechazo."
            )

    # 9. Todo listo → Fin
    logger.info("[Supervisor] Todo completado, finalizando")
    return make_update("__end__")
