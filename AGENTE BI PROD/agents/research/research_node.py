# core/research_node.py
import logging
from typing import Any, Callable, Dict, List, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph

from core.llm import LLM
from core.contracts import SQLContract, SQLPayload, ResearchPlan
from core.database import get_semantic_schema_for_views

logger = logging.getLogger("research_node")


def _normalize_view_name(view: Optional[str]) -> Optional[str]:
    if view and not view.startswith("semantic."):
        return f"semantic.{view}"
    return view


def _ensure_sql_payload(task: Any) -> SQLPayload:
    """Convierte una tarea a SQLPayload si llegó como dict."""
    if isinstance(task, dict):
        return SQLPayload(**task)
    return task


def _ensure_research_plan(plan: Any) -> ResearchPlan:
    """Convierte el plan a ResearchPlan si llegó como dict."""
    if isinstance(plan, dict):
        raw_tasks = plan.get("tasks", [])
        normalized_tasks = []
        for t in raw_tasks:
            if isinstance(t, dict):
                normalized_tasks.append(SQLPayload(**t))
            else:
                normalized_tasks.append(t)
        plan["tasks"] = normalized_tasks
        return ResearchPlan(**plan)
    return plan


def _validate_tasks(tasks: List[Any], allowed_views: List[str]) -> List[str]:
    """Asegura que candidate_views/preferred_view estén dentro de allowed_views."""
    warnings = []
    allowed_set = set(allowed_views)

    for raw_task in tasks:
        task = _ensure_sql_payload(raw_task)
        task.preferred_view = _normalize_view_name(task.preferred_view)

        if task.preferred_view and task.preferred_view not in allowed_set:
            valid_candidate = None
            for cv in task.candidate_views:
                cvn = _normalize_view_name(cv)
                if cvn in allowed_set:
                    valid_candidate = cvn
                    break
            if valid_candidate:
                task.preferred_view = valid_candidate
            else:
                warnings.append(
                    f"Tarea {task.task_id}: preferred_view "
                    f"'{task.preferred_view}' no está en allowed_views"
                )

        task.candidate_views = [
            _normalize_view_name(cv) for cv in task.candidate_views
            if _normalize_view_name(cv) in allowed_set
        ]

    return warnings


def make_research_node(
    sql_subgraph: StateGraph,
    llm=LLM,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    structured_llm = llm.with_structured_output(ResearchPlan, method="function_calling")

    def research_node(state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        question = state["question"]
        harness = state.get("harness_context", {})
        allowed_views = harness.get("allowed_views", [])
        preferred_view = harness.get("preferred_view")
        semantic_context = state.get("semantic_context", harness.get("semantic_context", ""))
        existing_sql_results = state.get("sql_results", []) or []

        # NUEVO: leer instrucción del supervisor y contexto de ledger
        instruction = state.get("next_agent_instruction") or state.get("supervisor_instruction")
        progress_ledger = state.get("progress_ledger", {}) or {}
        stall_count = progress_ledger.get("stall_count", 0)
        is_replan = stall_count > 0 or progress_ledger.get("unproductive_loop_detected", False)

        if instruction:
            logger.info(f"[Researcher] Instrucción recibida: {instruction[:200]}...")

        replan_hint = ""
        if is_replan:
            replan_hint = (
                "\nIMPORTANTE: Esta es una REPLANIFICACIÓN. Los intentos previos no generaron progreso. "
                "Diseña un plan de investigación ALTERNO: cambia el enfoque, las dimensiones, "
                "las métricas o las temporalidades. NO repitas el plan anterior."
            )

        system_prompt = f"""
Eres un Researcher BI senior. El usuario pide un informe profundo o análisis completo del negocio.

CONTEXTO:
- Vistas semánticas disponibles: {allowed_views}
- Vista preferida por el harness: {preferred_view}
- Contexto semántico: {semantic_context}

INSTRUCCIÓN DEL SUPERVISOR:
{instruction or "Ninguna instrucción adicional. Genera un informe completo y profundo."}{replan_hint}

OBJETIVO:
Genera un plan de exploración interno con múltiples queries SQL autocontenidas que cubran las métricas, dimensiones y temporalidades relevantes para responder: "{question}"

REGLAS:
1. Usa ÚNICAMENTE vistas de la lista 'allowed_views'.
2. Cada tarea debe ser un SQLPayload con: task_id, task, metrics, dimensions, filters_description, time_window, execution_strategy, candidate_views, preferred_view.
3. Incluye variedad analítica: tendencias históricas, comparativas por período, rankings, distribuciones por dimensión, KPIs agregados.
4. NO inventes vistas ni columnas.
5. Si es una replanificación, varía el enfoque y evita repetir queries anteriores.
"""

        try:
            raw_plan = structured_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Solicitud de informe: {question}")
            ])
            plan = _ensure_research_plan(raw_plan)
        except Exception:
            logger.exception("Error generando ResearchPlan")
            return {
                "sql_results": existing_sql_results,
                "research_findings": "No pude construir el plan de investigación para el informe.",
                "last_agent": "researcher",
                "messages": [
                    AIMessage(content="[Researcher] Error al generar plan de investigación")
                ],
                "next_agent_instruction": None,
            }

        warnings = _validate_tasks(plan.tasks, allowed_views)
        if warnings:
            logger.warning("[Researcher] Warnings de validación: %s", warnings)

        task_results: List[SQLContract] = []
        schema_cache: Dict[str, str] = {}

        for raw_task in plan.tasks:
            task = _ensure_sql_payload(raw_task)

            candidate_views = getattr(task, "candidate_views", []) or allowed_views
            preferred = getattr(task, "preferred_view", None) or preferred_view

            cache_key = ",".join(sorted(candidate_views))
            if cache_key not in schema_cache:
                try:
                    schema_cache[cache_key] = get_semantic_schema_for_views(candidate_views)
                except Exception:
                    schema_cache[cache_key] = ""
            schema_info = schema_cache[cache_key]

            payload_dict = task.dict() if hasattr(task, "dict") else dict(task)

            sub_input = {
                "question": question,
                "payload": payload_dict,
                "messages": [],
                "schema_info": schema_info,
                "semantic_context": semantic_context,
                "allowed_views": candidate_views,
                "preferred_view": preferred,
                "generated_sql": "",
                "query_result": None,
                "error_message": "",
                "contract": None,
                "attempts": 0,
            }

            try:
                sub_result = sql_subgraph.invoke(sub_input)
                contract = sub_result.get("contract")
                if contract is None:
                    contract = SQLContract(
                        status="error",
                        error_message="Subgrafo SQL no devolvió contrato",
                        needs_followup=True,
                    )
            except Exception as exc:
                logger.exception("SQL subgraph falló para tarea %s", task.task_id)
                contract = SQLContract(
                    status="error",
                    error_message=str(exc),
                    needs_followup=True,
                )

            contract.task_id = getattr(task, "task_id", "R1")
            contract.allowed_views = candidate_views
            contract.preferred_view = preferred
            task_results.append(contract)

        combined_sql_results = existing_sql_results + task_results

        result_summaries = []
        for c in task_results:
            sample_rows = c.rows[:5] if c.rows else []
            result_summaries.append(
                f"Tarea {c.task_id}: status={c.status}, filas={c.row_count}, "
                f"columnas={c.columns}, muestra={sample_rows}"
            )

        result_text = "\n\n".join(result_summaries)

        synthesis_prompt = f"""
Eres un analista de negocio senior. Tienes los resultados de {len(task_results)} queries ejecutadas sobre la base de datos interna para responder:

"{question}"

Objetivo del plan: {plan.goal}
Métricas cubiertas: {plan.metrics_to_cover}
Dimensiones cubiertas: {plan.dimensions_to_cover}
Secciones sugeridas: {plan.sections}

Resumen de resultados por tarea:
{result_text}

Instrucciones:
1. Genera un informe detallado en Markdown.
2. Incluye resumen ejecutivo, hallazgos por métrica/dimensión, tendencias, comparativas, conclusiones y recomendaciones.
3. Cita números y comparaciones concretas de los resultados.
4. Si alguna query falló, menciona el gap sin inventar datos.
5. Mantén tono ejecutivo y accionable.
"""

        try:
            synthesis = llm.invoke([
                SystemMessage(content=synthesis_prompt),
                HumanMessage(content="Genera el informe de investigación final.")
            ])
            findings = str(synthesis.content)
        except Exception:
            logger.exception("Error sintetizando informe del researcher")
            findings = (
                f"Se ejecutaron {len(task_results)} queries internas, "
                "pero no pudo generarse la síntesis final."
            )

        return {
            "sql_results": combined_sql_results,
            "research_findings": findings,
            "last_agent": "researcher",
            "messages": [
                AIMessage(
                    content=f"[Researcher] {len(task_results)} queries ejecutadas. "
                            f"Findings: {findings[:220]}..."
                )
            ],
            "next_agent_instruction": None,  # ← NUEVO: limpiar instrucción consumida
        }

    return research_node
