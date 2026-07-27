import json
from typing import Any, Dict, List, Optional
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from core.llm import LLM
from core.contracts import SupervisorDecision, PlannerContract, SQLContract
from core.config import logger
from core.harness import BusinessMemory
from langsmith import traceable

MAX_ITERATIONS = 10

_biz_mem_supervisor = BusinessMemory.from_file()


def _build_view_catalog_from_allowed(allowed_views: List[str]) -> Dict[str, Any]:
    """Construye catálogo semántico fallback si no viene en el estado."""
    catalog: Dict[str, Any] = {}
    for view_full_name in allowed_views:
        view_name = view_full_name.replace("semantic.", "").strip()
        view_info = _biz_mem_supervisor.get_view(view_name)
        if not view_info:
            continue
        catalog[view_full_name] = {
            "tipo": view_info.tipo,
            "descripcion": view_info.descripcion,
            "granularidad": view_info.granularidad,
            "filtro_fecha": view_info.filtro_fecha,
            "metricas": list(view_info.metricas.keys()),
            "columnas_fecha": view_info.columnas_fecha,
            "notas": view_info.notas,
        }
    return catalog


def _validate_plan_against_catalog(
    plan: PlannerContract,
    view_catalog: Dict[str, Any]
) -> tuple[bool, Optional[str]]:
    """Valida que el plan sea técnicamente ejecutable contra el catálogo de vistas."""
    if not plan or not plan.tasks:
        return False, "El plan no contiene tareas."

    for task in plan.tasks:
        pv = task.preferred_view
        if not pv:
            return False, f"Tarea {task.task_id}: no tiene preferred_view asignada."

        if pv not in task.candidate_views:
            return False, (
                f"Tarea {task.task_id}: preferred_view '{pv}' no está en candidate_views "
                f"({task.candidate_views})."
            )

        view_def = view_catalog.get(pv) or view_catalog.get(pv.replace("semantic.", ""))
        if not view_def:
            return False, f"Tarea {task.task_id}: la vista '{pv}' no existe en el catálogo."

        available_cols = set(
            view_def.get("metricas", [])
            + view_def.get("dimensiones", [])
            + view_def.get("columnas_fecha", [])
            + view_def.get("columnas", [])
        )

        required_cols = set(
            task.metrics
            + task.dimensions
            + [f.column for f in task.filters]
        )
        missing = required_cols - available_cols
        if missing:
            return False, (
                f"Tarea {task.task_id}: columnas requeridas no disponibles en '{pv}': {sorted(missing)}. "
                f"Columnas disponibles: {sorted(available_cols)}."
            )

    return True, None


def _extract_statuses(sql_results: List[SQLContract]) -> Dict[str, List[SQLContract]]:
    grouped = {
        "success": [],
        "error": [],
        "partial": [],
        "needs_clarification": [],
        "unrecoverable": [],
    }
    for r in sql_results or []:
        status = getattr(r, "status", None)
        if status in grouped:
            grouped[status].append(r)
    return grouped


def _summarize_sql_results(sql_results: List[SQLContract]) -> str:
    if not sql_results:
        return "No hay resultados SQL aún."
    grouped = _extract_statuses(sql_results)
    lines = []
    for status, items in grouped.items():
        if items:
            lines.append(f"- {status}: {len(items)} tarea(s)")
            for r in items:
                task_id = getattr(r, "task_id", "?")
                reason = getattr(r, "reason_for_view_choice", None) or getattr(r, "error_message", "")
                lines.append(f"  - tarea {task_id}: {reason[:120]}")
    return "\n".join(lines)


def _build_state_summary(state: Dict[str, Any]) -> str:
    plan = state.get("plan")
    sql_results = state.get("sql_results", [])
    viz_result = state.get("viz_result")
    last_agent = state.get("last_agent", "ninguno")
    iteration_count = state.get("iteration_count", 0)
    final_answer = state.get("final_answer")

    summary_lines = [
        f"iteration_count: {iteration_count}",
        f"last_agent: {last_agent}",
        f"question_type: {getattr(plan, 'question_type', 'sin plan')}",
        f"needs_followup: {getattr(plan, 'needs_followup', False)}",
        f"visualization_candidate: {getattr(plan, 'visualization_candidate', False)}",
        f"viz_result status: {getattr(viz_result, 'status', None)}",
        f"viz_result chart_type: {getattr(viz_result, 'chart_type', None)}",
        f"viz_rendered: {state.get('viz_rendered', False)}",
        f"forecast_results presentes: {state.get('forecast_results') is not None}",
        f"forecast_error presente: {state.get('forecast_error') is not None}",
        f"research_findings presentes: {state.get('research_findings') is not None}",
        f"final_answer presente: {final_answer is not None}",
        "",
        "=== RESUMEN SQL ===",
        _summarize_sql_results(sql_results),
    ]
    return "\n".join(summary_lines)


# ============================================================
# FEW-SHOT EXAMPLES
# ============================================================

def _build_few_shot_messages() -> List[Any]:
    return [
        # 1. Sin plan
        HumanMessage(content="""
=== ESTADO ACTUAL ===
iteration_count: 0
last_agent: ninguno
question_type: sin plan
needs_followup: false
visualization_candidate: false
viz_result status: None
viz_result chart_type: None
viz_rendered: false
forecast_results presentes: false
forecast_error presente: false
research_findings presentes: false
final_answer presente: false

=== RESUMEN SQL ===
No hay resultados SQL aún.
        """.strip()),
        AIMessage(content=json.dumps({
            "reasoning": "No existe plan. Se requiere decomponer la pregunta del usuario.",
            "next_agent": "planner",
            "feedback_to_planner": None,
            "feedback_to_sql_agent": None
        }, ensure_ascii=False)),

        # 2. Plan válido, sin SQL
        HumanMessage(content="""
=== ESTADO ACTUAL ===
iteration_count: 1
last_agent: planner
question_type: single_kpi
needs_followup: false
visualization_candidate: false
viz_result status: None
viz_result chart_type: None
viz_rendered: false
forecast_results presentes: false
forecast_error presente: false
research_findings presentes: false
final_answer presente: false

=== RESUMEN SQL ===
No hay resultados SQL aún.
        """.strip()),
        AIMessage(content=json.dumps({
            "reasoning": "Plan válido con una tarea de ventas por sede. Se debe generar el SQL.",
            "next_agent": "sql_agent",
            "feedback_to_planner": None,
            "feedback_to_sql_agent": None
        }, ensure_ascii=False)),

        # 3. Plan inválido
        HumanMessage(content="""
=== ESTADO ACTUAL ===
iteration_count: 1
last_agent: planner
question_type: single_kpi
needs_followup: false
visualization_candidate: false
viz_result status: None
viz_result chart_type: None
viz_rendered: false
forecast_results presentes: false
forecast_error presente: false
research_findings presentes: false
final_answer presente: false

=== RESUMEN SQL ===
No hay resultados SQL aún.
        """.strip()),
        AIMessage(content=json.dumps({
            "reasoning": "El planner eligió una vista que no contiene la columna 'vendedor' requerida. Se solicita replanificación.",
            "next_agent": "planner",
            "feedback_to_planner": "Tarea t1: columna 'vendedor' no está disponible en semantic.vw_ventas_sede. Elige una vista que contenga vendedor o reformula la pregunta.",
            "feedback_to_sql_agent": None
        }, ensure_ascii=False)),

        # 4. SQL error corregible
        HumanMessage(content="""
=== ESTADO ACTUAL ===
iteration_count: 2
last_agent: sql_agent
question_type: single_kpi
needs_followup: false
visualization_candidate: false
viz_result status: None
viz_result chart_type: None
viz_rendered: false
forecast_results presentes: false
forecast_error presente: false
research_findings presentes: false
final_answer presente: false

=== RESUMEN SQL ===
- error: 1 tarea(s)
  - tarea t1: column "fecha_venta" does not exist
        """.strip()),
        AIMessage(content=json.dumps({
            "reasoning": "La query SQL falló por una columna mal escrita. Se reintenta con el mensaje de error como feedback.",
            "next_agent": "sql_agent",
            "feedback_to_planner": None,
            "feedback_to_sql_agent": "Tarea t1: Error 'column \"fecha_venta\" does not exist'. Revisa el catálogo y usa el nombre exacto de la columna de fecha disponible en la vista."
        }, ensure_ascii=False)),

        # 5. SQL insalvable
        HumanMessage(content="""
=== ESTADO ACTUAL ===
iteration_count: 2
last_agent: sql_agent
question_type: single_kpi
needs_followup: false
visualization_candidate: false
viz_result status: None
viz_result chart_type: None
viz_rendered: false
forecast_results presentes: false
forecast_error presente: false
research_findings presentes: false
final_answer presente: false

=== RESUMEN SQL ===
- unrecoverable: 1 tarea(s)
  - tarea t1: Columna nombre_vendedor no existe en semantic.vw_ventas_sede
        """.strip()),
        AIMessage(content=json.dumps({
            "reasoning": "La tarea requiere una columna que no existe. Se debe replanificar.",
            "next_agent": "planner",
            "feedback_to_planner": "Tarea t1: la columna 'nombre_vendedor' no existe en semantic.vw_ventas_sede. No inventes columnas. Reformula la tarea sin desglose por vendedor.",
            "feedback_to_sql_agent": None
        }, ensure_ascii=False)),

        # 6. Needs clarification
        HumanMessage(content="""
=== ESTADO ACTUAL ===
iteration_count: 2
last_agent: sql_agent
question_type: single_kpi
needs_followup: false
visualization_candidate: false
viz_result status: None
viz_result chart_type: None
viz_rendered: false
forecast_results presentes: false
forecast_error presente: false
research_findings presentes: false
final_answer presente: false

=== RESUMEN SQL ===
- needs_clarification: 1 tarea(s)
  - tarea t1: Ambigüedad: la pregunta menciona 'local' pero no se detecta si es nombre_sede, comuna o región
        """.strip()),
        AIMessage(content=json.dumps({
            "reasoning": "El SQL agent no puede resolver una ambigüedad semántica sin aclaración del usuario. Se finaliza con pregunta.",
            "next_agent": "FINISH",
            "feedback_to_planner": None,
            "feedback_to_sql_agent": None
        }, ensure_ascii=False)),

        # 7. demand_forecast
        HumanMessage(content="""
=== ESTADO ACTUAL ===
iteration_count: 1
last_agent: planner
question_type: demand_forecast
needs_followup: false
visualization_candidate: false
viz_result status: None
viz_result chart_type: None
viz_rendered: false
forecast_results presentes: false
forecast_error presente: false
research_findings presentes: false
final_answer presente: false

=== RESUMEN SQL ===
No hay resultados SQL aún.
        """.strip()),
        AIMessage(content=json.dumps({
            "reasoning": "La pregunta es una predicción de demanda. Se debe ejecutar el forecaster.",
            "next_agent": "forecaster",
            "feedback_to_planner": None,
            "feedback_to_sql_agent": None
        }, ensure_ascii=False)),

        # 8. deep_research
        HumanMessage(content="""
=== ESTADO ACTUAL ===
iteration_count: 1
last_agent: planner
question_type: deep_research
needs_followup: false
visualization_candidate: false
viz_result status: None
viz_result chart_type: None
viz_rendered: false
forecast_results presentes: false
forecast_error presente: false
research_findings presentes: false
final_answer presente: false

=== RESUMEN SQL ===
No hay resultados SQL aún.
        """.strip()),
        AIMessage(content=json.dumps({
            "reasoning": "Se solicitó un informe profundo. Se delega al researcher.",
            "next_agent": "researcher",
            "feedback_to_planner": None,
            "feedback_to_sql_agent": None
        }, ensure_ascii=False)),

        # 9. visualización
        HumanMessage(content="""
=== ESTADO ACTUAL ===
iteration_count: 3
last_agent: sql_agent
question_type: single_kpi
needs_followup: false
visualization_candidate: true
viz_result status: None
viz_result chart_type: None
viz_rendered: false
forecast_results presentes: false
forecast_error presente: false
research_findings presentes: false
final_answer presente: false

=== RESUMEN SQL ===
- success: 1 tarea(s)
  - tarea t1: Query ejecutada correctamente
        """.strip()),
        AIMessage(content=json.dumps({
            "reasoning": "SQL exitoso y el plan indica visualización. Se genera la especificación del gráfico.",
            "next_agent": "viz_agent",
            "feedback_to_planner": None,
            "feedback_to_sql_agent": None
        }, ensure_ascii=False)),

        # 10. render
        HumanMessage(content="""
=== ESTADO ACTUAL ===
iteration_count: 4
last_agent: viz_agent
question_type: single_kpi
needs_followup: false
visualization_candidate: true
viz_result status: success
viz_result chart_type: bar
viz_rendered: false
forecast_results presentes: false
forecast_error presente: false
research_findings presentes: false
final_answer presente: false

=== RESUMEN SQL ===
- success: 1 tarea(s)
  - tarea t1: Query ejecutada correctamente
        """.strip()),
        AIMessage(content=json.dumps({
            "reasoning": "Spec de visualización válida (bar chart). Se procede a renderizar.",
            "next_agent": "render_plotly",
            "feedback_to_planner": None,
            "feedback_to_sql_agent": None
        }, ensure_ascii=False)),

        # 11. analyst
        HumanMessage(content="""
=== ESTADO ACTUAL ===
iteration_count: 3
last_agent: sql_agent
question_type: single_kpi
needs_followup: false
visualization_candidate: false
viz_result status: None
viz_result chart_type: None
viz_rendered: false
forecast_results presentes: false
forecast_error presente: false
research_findings presentes: false
final_answer presente: false

=== RESUMEN SQL ===
- success: 1 tarea(s)
  - tarea t1: Query ejecutada correctamente, 45 filas
        """.strip()),
        AIMessage(content=json.dumps({
            "reasoning": "SQL ejecutado con éxito y no se requiere visualización. Se redacta respuesta final.",
            "next_agent": "analyst",
            "feedback_to_planner": None,
            "feedback_to_sql_agent": None
        }, ensure_ascii=False)),

        # 12. FINISH
        HumanMessage(content="""
=== ESTADO ACTUAL ===
iteration_count: 5
last_agent: analyst
question_type: single_kpi
needs_followup: false
visualization_candidate: false
viz_result status: None
viz_result chart_type: None
viz_rendered: false
forecast_results presentes: false
forecast_error presente: false
research_findings presentes: false
final_answer presente: true

=== RESUMEN SQL ===
- success: 1 tarea(s)
  - tarea t1: Query ejecutada correctamente
        """.strip()),
        AIMessage(content=json.dumps({
            "reasoning": "La respuesta final ya está lista. Se finaliza el flujo.",
            "next_agent": "FINISH",
            "feedback_to_planner": None,
            "feedback_to_sql_agent": None
        }, ensure_ascii=False)),
    ]


@traceable(name="Supervisor LLM: Route Decision")
def supervisor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    current_iter = state.get("iteration_count", 0)
    next_iter = current_iter + 1

    # 1. Límite de iteraciones
    if current_iter >= MAX_ITERATIONS:
        logger.warning(f"[Supervisor] Límite de iteraciones ({MAX_ITERATIONS}) alcanzado.")
        return {
            "next": "__end__",
            "last_agent": "supervisor",
            "iteration_count": next_iter,
            "final_answer": state.get("final_answer") or "No pude completar la consulta tras varios intentos.",
            "messages": [AIMessage(content="[Supervisor] Iteración límite alcanzada. Cerrando.")],
        }

    plan = state.get("plan")
    sql_results = state.get("sql_results") or []
    view_catalog = state.get("view_catalog", {})
    allowed_views = state.get("allowed_views", [])

    # Fallback: construir catálogo si no viene en estado
    if not view_catalog and allowed_views:
        logger.info("[Supervisor] view_catalog no encontrado en estado. Construyendo desde allowed_views.")
        view_catalog = _build_view_catalog_from_allowed(allowed_views)

    # 2. Sin plan → Planner
    if not plan:
        return {
            "next": "planner",
            "last_agent": "supervisor",
            "iteration_count": next_iter,
            "messages": [AIMessage(content="[Supervisor] Sin plan. Ruteando a planner.")],
        }

    # 3. Plan pide aclaración al usuario
    if getattr(plan, "needs_followup", False):
        followup = plan.followup_question or "Necesito más información para responder tu consulta."
        return {
            "next": "__end__",
            "last_agent": "supervisor",
            "iteration_count": next_iter,
            "final_answer": followup,
            "messages": [AIMessage(content=f"[Supervisor] Aclaración requerida: {followup}")],
        }

    # 4. Validación del plan contra catálogo
    plan_valid, plan_feedback = _validate_plan_against_catalog(plan, view_catalog)
    if not plan_valid:
        last_agent = state.get("last_agent")
        if last_agent == "planner":
            logger.warning("[Supervisor] Plan inválido tras replanificación. Forzando analyst.")
            return {
                "next": "analyst",
                "last_agent": "supervisor",
                "iteration_count": next_iter,
                "planner_validation_error": plan_feedback,
                "messages": [
                    AIMessage(
                        content=f"[Supervisor] Plan sigue siendo inválido: {plan_feedback}. "
                                "Se redacta respuesta explicativa."
                    )
                ],
            }

        logger.warning(f"[Supervisor] Plan inválido: {plan_feedback}. Replanificando.")
        return {
            "next": "planner",
            "last_agent": "supervisor",
            "iteration_count": next_iter,
            "feedback_to_planner": plan_feedback,
            "messages": [
                AIMessage(
                    content=f"[Supervisor] Plan inválido detectado: {plan_feedback}. Replanificando."
                )
            ],
        }

    # 5. Pre-dispatch hardcodeado para casos de alto riesgo
    statuses = _extract_statuses(sql_results)

    # needs_clarification → terminar con pregunta al usuario
    if statuses["needs_clarification"] and not state.get("final_answer"):
        questions = "\n".join(
            f"- {getattr(r, 'error_message', 'Necesito aclaración')}"
            for r in statuses["needs_clarification"]
        )
        return {
            "next": "__end__",
            "last_agent": "supervisor",
            "iteration_count": next_iter,
            "final_answer": f"Necesito aclaración para continuar:\n{questions}",
            "messages": [AIMessage(content=f"[Supervisor] Aclaración requerida:\n{questions}")],
        }

    # unrecoverable → replanificar con feedback (una sola vez)
    if statuses["unrecoverable"] and state.get("last_agent") != "planner":
        feedback = "\n".join(
            f"Tarea {getattr(r, 'task_id', '?')}: "
            f"{getattr(r, 'reason_for_view_choice', getattr(r, 'error_message', 'Error insalvable'))}"
            for r in statuses["unrecoverable"]
        )
        return {
            "next": "planner",
            "last_agent": "supervisor",
            "iteration_count": next_iter,
            "feedback_to_planner": feedback,
            "messages": [AIMessage(content=f"[Supervisor] Errores insalvables. Replanificando: {feedback}")],
        }

    # 6. Delegar decisión al LLM
    state_summary = _build_state_summary(state)

    system_prompt = f"""
Eres el Supervisor Orchestrator de un sistema BI multi-agente construido en LangGraph.
Tu trabajo es decidir cuál es el siguiente nodo a ejecutar, basándote en el estado actual.

=== AGENTES DISPONIBLES ===
- planner: replanifica cuando el plan es inválido, incompleto o insuficiente.
- sql_agent: genera o corrige queries SQL.
- researcher: investigación profunda (solo si question_type == "deep_research").
- forecaster: predicción de demanda (solo si question_type == "demand_forecast").
- viz_agent: genera especificación de gráfico.
- render_plotly: renderiza visualización.
- analyst: redacta respuesta final.
- FINISH: termina y devuelve respuesta al usuario.

=== ESTADO ACTUAL ===
{state_summary}

=== REGLAS DE RUTEO ===
1. demand_forecast sin resultados/error → forecaster.
2. demand_forecast con error/results y sin final_answer → analyst.
3. deep_research sin research_findings → researcher.
4. visualization_candidate=true, sin viz_result, sin errores bloqueantes → viz_agent.
5. viz_result válido y no renderizado → render_plotly.
6. viz_result inválido o render fallido → analyst.
7. sql_results con status="error" y no venimos de sql_agent → sql_agent (con feedback).
8. No hay sql_results y plan válido → sql_agent.
9. Hay resultados de éxito o research_findings y no hay final_answer → analyst.
10. Ya existe final_answer → FINISH.

=== ANTI-LOOP ===
- No envíes dos veces seguidas a sql_agent sin feedback_to_sql_agent.
- No envíes dos veces seguidas a planner sin cambio real en el estado.
- Si iteration_count > 8 y hay errores persistentes, termina con FINISH.

=== FORMATO DE SALIDA ===
Devuelve SupervisorDecision con:
- reasoning
- next_agent
- feedback_to_planner (solo si next_agent == planner)
- feedback_to_sql_agent (solo si next_agent == sql_agent)
"""

    few_shot_messages = _build_few_shot_messages()

    messages = [
        SystemMessage(content=system_prompt),
        *few_shot_messages,
        HumanMessage(content="Decide el siguiente nodo a ejecutar para el estado actual."),
    ]

    try:
        llm_low_temp = LLM.bind(temperature=0.0)
        structured_llm = llm_low_temp.with_structured_output(SupervisorDecision, include_raw=False)
        decision_raw = structured_llm.invoke(messages)

        # ============================================================
        # FIX CRÍTICO: langchain puede devolver dict en lugar de objeto
        # ============================================================
        if isinstance(decision_raw, dict):
            logger.info(f"[Supervisor] Decisión recibida como dict: {decision_raw}")
            decision = SupervisorDecision(**decision_raw)
        else:
            decision = decision_raw

    except Exception as e:
        logger.error(f"[Supervisor] Error llamando al LLM: {e}")
        decision = SupervisorDecision(
            reasoning=f"Fallback por error del LLM: {str(e)}",
            next_agent="analyst",
        )

    # 7. Guardrails post-LLM
    valid_agents = [
        "planner", "sql_agent", "researcher", "forecaster",
        "viz_agent", "render_plotly", "analyst", "FINISH"
    ]

    # Si por alguna razón next_agent no existe o es inválido
    if not hasattr(decision, "next_agent") or decision.next_agent not in valid_agents:
        logger.warning(f"[Supervisor] next_agent inválido '{getattr(decision, 'next_agent', None)}'. Forzando analyst.")
        decision = SupervisorDecision(
            reasoning=f"next_agent inválido '{getattr(decision, 'next_agent', None)}' corregido a analyst",
            next_agent="analyst",
        )

    # Anti-loop: no planner dos veces seguidas
    if decision.next_agent == "planner" and state.get("last_agent") == "planner":
        logger.warning("[Supervisor] Loop planner→planner. Forzando analyst.")
        decision = SupervisorDecision(
            reasoning="Loop detectado: planner fue el último agente. Se redacta respuesta explicativa.",
            next_agent="analyst",
        )

    # Anti-loop: no sql_agent dos veces seguidas sin feedback
    if decision.next_agent == "sql_agent" and state.get("last_agent") == "sql_agent":
        if not decision.feedback_to_sql_agent:
            error_feedback = "\n".join(
                f"Tarea {getattr(r, 'task_id', '?')}: {getattr(r, 'error_message', 'error desconocido')}"
                for r in statuses["error"]
            ) or "Reintento solicitado sin feedback explícito."
            decision.feedback_to_sql_agent = error_feedback

    # FINISH solo si hay final_answer o es caso de terminación forzada
    if decision.next_agent == "FINISH" and not state.get("final_answer"):
        if not statuses["needs_clarification"]:
            logger.warning("[Supervisor] FINISH elegido sin respuesta final. Forzando analyst.")
            decision = SupervisorDecision(
                reasoning="FINISH elegido sin respuesta final disponible. Se redacta primero.",
                next_agent="analyst",
            )

    # Limpiar feedbacks que no corresponden
    if decision.next_agent != "planner":
        decision.feedback_to_planner = None
    if decision.next_agent != "sql_agent":
        decision.feedback_to_sql_agent = None

    # 8. Construir update de estado
    update = {
        "next": decision.next_agent,
        "last_agent": "supervisor",
        "iteration_count": next_iter,
        "supervisor_reasoning": decision.reasoning,
        "messages": [
            AIMessage(
                content=f"[Supervisor] Ruta seleccionada: {decision.next_agent}. "
                        f"Razón: {decision.reasoning}"
            )
        ],
    }
    if decision.feedback_to_planner:
        update["feedback_to_planner"] = decision.feedback_to_planner
    if decision.feedback_to_sql_agent:
        update["feedback_to_sql_agent"] = decision.feedback_to_sql_agent

    logger.info(f"[Supervisor] Decisión final: {decision.next_agent} | {decision.reasoning}")
    return update
