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


def _validate_tasks(tasks: List[SQLPayload], allowed_views: List[str]) -> List[str]:
    """Asegura que candidate_views/preferred_view estén dentro de allowed_views."""
    warnings = []
    allowed_set = set(allowed_views)

    for task in tasks:
        task.preferred_view = _normalize_view_name(task.preferred_view)

        # Fallback de preferred_view si quedó fuera del shortlist
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

        # Normalizar candidate_views
        task.candidate_views = [
            _normalize_view_name(cv) for cv in task.candidate_views
            if _normalize_view_name(cv) in allowed_set
        ]

    return warnings


def make_research_node(
    sql_subgraph: StateGraph,
    llm=LLM,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Fabrica un nodo 'researcher' que explora la base de datos interna.
    - sql_subgraph: tu SQL_SUBGRAPH compilado.
    - llm: modelo LLM (default core.llm.LLM).
    """
    structured_llm = llm.with_structured_output(ResearchPlan, method="json_schema")

    def research_node(state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]
        harness = state.get("harness_context", {})
        allowed_views = harness.get("allowed_views", [])
        preferred_view = harness.get("preferred_view")
        semantic_context = state.get("semantic_context", harness.get("semantic_context", ""))
        existing_sql_results = state.get("sql_results", []) or []

        # ------------------------------------------------------------------
        # 1. Generar plan de investigación con múltiples queries
        # ------------------------------------------------------------------
        system_prompt = f"""
Eres un Researcher BI senior. El usuario pide un informe profundo o análisis completo del negocio.

CONTEXTO:
- Vistas semánticas disponibles: {allowed_views}
- Vista preferida por el harness: {preferred_view}
- Contexto semántico: {semantic_context}

OBJETIVO:
Genera un plan de exploración interno con múltiples queries SQL autocontenidas que cubran las métricas, dimensiones y temporalidades relevantes para responder: "{question}"

REGLAS:
1. Usa ÚNICAMENTE vistas de la lista 'allowed_views'.
2. Cada tarea debe ser un SQLPayload con: task_id, task, metrics, dimensions, filters_description, time_window, execution_strategy, candidate_views, preferred_view.
3. Incluye variedad analítica: tendencias históricas, comparativas por período, rankings, distribuciones por dimensión, KPIs agregados.
4. NO inventes vistas ni columnas.
"""

        try:
            plan: ResearchPlan = structured_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Solicitud de informe: {question}")
            ])
        except Exception:
            logger.exception("Error generando ResearchPlan")
            return {
                "sql_results": existing_sql_results,
                "research_findings": "No pude construir el plan de investigación para el informe.",
                "last_agent": "researcher",
                "messages": [
                    AIMessage(content="[Researcher] Error al generar plan de investigación")
                ]
            }

        warnings = _validate_tasks(plan.tasks, allowed_views)
        if warnings:
            logger.warning("[Researcher] Warnings de validación: %s", warnings)

        # ------------------------------------------------------------------
        # 2. Ejecutar cada query a través del SQL_SUBGRAPH existente
        # ------------------------------------------------------------------
        task_results: List[SQLContract] = []

        for task in plan.tasks:
            candidate_views = getattr(task, "candidate_views", []) or allowed_views
            preferred = getattr(task, "preferred_view", None) or preferred_view

            try:
                schema_info = get_semantic_schema_for_views(candidate_views)
            except Exception:
                schema_info = ""

            sub_input = {
                "question": question,
                "payload": task.dict() if hasattr(task, "dict") else dict(task),
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

        # ------------------------------------------------------------------
        # 3. Sintetizar un informe Markdown con todos los hallazgos
        # ------------------------------------------------------------------
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
            ]
        }

    return research_node
