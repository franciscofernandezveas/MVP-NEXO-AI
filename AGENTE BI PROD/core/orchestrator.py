import logging
from typing import Optional, Any, List, Dict, Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, AIMessage

from agents.planner.graph_planner import planner_node
from agents.analyst.graph_analyst import analyst_node
from agents.supervisor.graph_supervisor import supervisor_node
from agents.sql_agent.graph_sql_agent import SQL_SUBGRAPH
from agents.viz_agent.graph_viz_agent import VIZ_SUBGRAPH
from agents.viz_agent.render_node import render_plotly_node
from agents.viz_approval.graph_viz_approval import viz_approval_node
from agents.research.research_node import make_research_node
# NOTE: run_forecast se importa lazy dentro de forecaster_node para no cargar
# el módulo de forecasting (y sus dependencias pesadas) en cada chat.
from core.llm import LLM
from core.contracts import SQLContract

from core.harness import build_harness_context, build_harness_context_cached, _normalize_question
from core.database import get_semantic_schema_for_views

from langsmith import traceable

logger = logging.getLogger(__name__)


class OrchestratorState(TypedDict):
    question: str
    messages: Annotated[List[BaseMessage], add_messages]
    plan: Optional[Any]
    sql_results: List[Any]
    viz_result: Optional[Any]
    viz_approved: Optional[bool]
    viz_rendered: bool
    final_answer: Optional[str]
    iteration_count: int
    last_agent: Optional[str]
    next: Optional[str]  # NUEVO: usado por supervisor para routing condicional
    harness_context: Optional[Dict[str, Any]]
    semantic_context: str
    allowed_views: List[str]
    preferred_view: Optional[str]
    schema_info: str
    research_findings: Optional[str]
    forecast_request: Optional[Dict[str, Any]]
    forecast_results: Optional[List[Dict[str, Any]]]
    forecast_error: Optional[str]


@traceable(name="Orchestrator: Build Harness Context")
def build_harness_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    harness = build_harness_context_cached(_normalize_question(state["question"]))

    return {
        "harness_context": harness,
        "semantic_context": harness.get("semantic_context", ""),
        "allowed_views": harness.get("allowed_views", []),
        "preferred_view": harness.get("preferred_view"),
        "schema_info": "",
        "messages": state.get("messages", []) + [
            AIMessage(
                content=f"[Harness] Preferred: {harness.get('preferred_view')} | "
                        f"Allowed: {harness.get('allowed_views')} | "
                        f"Ambiguity: {harness.get('ambiguity_notes')}"
            )
        ]
    }


@traceable(name="Orchestrator: Execute SQL Tasks")
def sql_agent_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
    if not state.get("plan"):
        raise ValueError("SQL Agent llamado sin plan previo.")

    tasks = state["plan"].tasks
    results: List[SQLContract] = []

    harness_ctx = state.get("harness_context", {})
    global_semantic_context = state.get("semantic_context", harness_ctx.get("semantic_context", ""))

    for task in tasks:
        payload = task.dict() if hasattr(task, "dict") else dict(task)

        candidate_views = getattr(task, "candidate_views", None) or state.get("allowed_views", [])
        preferred = getattr(task, "preferred_view", None) or state.get("preferred_view")

        try:
            schema_info = get_semantic_schema_for_views(candidate_views)
        except Exception:
            schema_info = state.get("schema_info", "")

        sub_input = {
            "question": state["question"],
            "payload": payload,
            "messages": [],
            "schema_info": schema_info,
            "semantic_context": global_semantic_context,
            "allowed_views": candidate_views,
            "preferred_view": preferred,
            "generated_sql": "",
            "query_result": None,
            "error_message": "",
            "contract": None,
            "attempts": 0
        }

        sub_result = SQL_SUBGRAPH.invoke(sub_input)
        contract = sub_result.get("contract")
        if contract is None:
            contract = SQLContract(
                status="error",
                error_message="Subgrafo SQL no devolvió contrato",
                needs_followup=True
            )

        contract.task_id = getattr(task, "task_id", "1")
        contract.allowed_views = candidate_views
        contract.preferred_view = preferred
        contract.semantic_context_used = (
            global_semantic_context[:500] + "..."
            if len(global_semantic_context) > 500
            else global_semantic_context
        )

        results.append(contract)

    all_success = all(
        r.status in ("success", "partial") and r.can_answer
        for r in results
    )
    summary = " | ".join([f"T{r.task_id}:{r.status}({r.row_count})" for r in results])

    return {
        "sql_results": results,
        "last_agent": "sql_agent",
        "messages": [
            AIMessage(
                content=f"[SQL Agent] {len(results)} tareas ejecutadas. "
                        f"OK={all_success} | {summary}"
            )
        ]
    }


def viz_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    viz_input = {
        "question": state["question"],
        "sql_rows": state["sql_results"][0].rows if state["sql_results"] and len(state["sql_results"]) > 0 else [],
        "sql_columns": state["sql_results"][0].columns if state["sql_results"] and len(state["sql_results"]) > 0 else [],
        "chart_type_hint": getattr(state.get("plan"), "chart_type_hint", "auto"),
        "messages": [],
        "figure_spec": None,
        "error_message": "",
        "attempts": 0,
        "contract": None
    }

    viz_result = VIZ_SUBGRAPH.invoke(viz_input)
    return {
        "viz_result": viz_result["contract"],
        "last_agent": "viz_agent",
        "messages": [AIMessage(content="[Viz Agent] Especificación de visualización generada")]
    }


@traceable(name="Orchestrator: Execute Demand Forecast")
def forecaster_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # Lazy import: solo carga forecasting cuando realmente se usa
    from agents.forecasting_agent.graph_demand_forecaster import run_forecast

    logger.info(f"[Forecaster] Estado recibido. forecast_request={state.get('forecast_request')}")
    logger.info(f"[Forecaster] plan question_type={getattr(state.get('plan'), 'question_type', None)}")

    request = state.get("forecast_request")

    if not request:
        plan = state.get("plan")
        if plan and getattr(plan, "question_type", None) == "demand_forecast":
            filters = getattr(plan, "filters", "")
            producto = None
            sede = None
            for part in filters.split(","):
                part = part.strip()
                if part.startswith("producto="):
                    producto = part.split("=", 1)[1].strip()
                elif part.startswith("sede="):
                    sede = part.split("=", 1)[1].strip()

            if producto and sede:
                logger.info(f"[Forecaster] Fallback desde plan: {producto} @ {sede}")
                request = {
                    "producto": producto,
                    "sede": sede,
                    "n_dias": 7,
                    "fecha_inicio": None,
                }

    if not request:
        question = state["question"].lower()
        sedes = ["plaza bolsillo", "merced", "tajamar", "persa victor manuel"]
        sede = None
        for s in sedes:
            if s in question:
                sede = s.title()
                break

        productos = ["americano", "capuccino", "latte", "espresso", "mokaccino", "cortado", "flat white", "iced latte", "chai latte"]
        producto = None
        for p in productos:
            if p in question:
                producto = p
                break

        n_dias = 7
        import re
        dias_match = re.search(r"(\d+)\s*días?|(\d+)\s*dias?", question)
        if dias_match:
            n_dias = int(dias_match.group(1) or dias_match.group(2))

        if producto and sede:
            logger.info(f"[Forecaster] Fallback desde pregunta: {producto} @ {sede}")
            request = {
                "producto": producto,
                "sede": sede,
                "n_dias": n_dias,
                "fecha_inicio": None,
            }

    if not request:
        logger.error("[Forecaster] No hay parámetros de forecast después de todos los fallbacks")
        return {
            "forecast_error": "No hay parámetros de forecast.",
            "final_answer": "No pude determinar el producto y la sede para el pronóstico.",
            "messages": state.get("messages", []) + [
                AIMessage(content="[Forecaster] Sin parámetros de predicción")
            ],
            "last_agent": "forecaster",
        }

    try:
        logger.info(f"[Forecaster] Ejecutando run_forecast: {request}")
        result = run_forecast(
            producto=request["producto"],
            sede=request["sede"],
            n_dias=int(request.get("n_dias", 7)),
            fecha_inicio=request.get("fecha_inicio")
        )

        forecasts = result.get("forecasts", []) if isinstance(result, dict) else []

        return {
            "forecast_results": forecasts,
            "forecast_error": None,
            "last_agent": "forecaster",
            "messages": state.get("messages", []) + [
                AIMessage(
                    content=f"[Forecaster] {len(forecasts)} días pronosticados para "
                            f"{request['producto']} @ {request['sede']}"
                )
            ],
        }

    except Exception as e:
        logger.error(f"[Forecaster Node] Error: {e}", exc_info=True)
        return {
            "forecast_error": str(e),
            "final_answer": f"Error al generar el pronóstico: {str(e)}",
            "messages": state.get("messages", []) + [
                AIMessage(content=f"Error al predecir: {e}")
            ],
            "last_agent": "forecaster",
        }


researcher_node = make_research_node(SQL_SUBGRAPH, LLM)


builder = StateGraph(OrchestratorState)

builder.add_node("build_harness", build_harness_context_node)
builder.add_node("supervisor", supervisor_node)
builder.add_node("planner", planner_node)
builder.add_node("sql_agent", sql_agent_wrapper)
builder.add_node("analyst", analyst_node)
builder.add_node("viz_agent", viz_agent_node)
builder.add_node("render_plotly", render_plotly_node)
builder.add_node("viz_approval", viz_approval_node)
builder.add_node("researcher", researcher_node)
builder.add_node("forecaster", forecaster_node)

# FLUJO ENTRADA
builder.add_edge("__start__", "build_harness")
builder.add_edge("build_harness", "supervisor")

# Supervisor decide el siguiente nodo según el campo "next" del estado
builder.add_conditional_edges(
    "supervisor",
    lambda state: state.get("next", "__end__"),
    {
        "planner": "planner",
        "sql_agent": "sql_agent",
        "analyst": "analyst",
        "viz_agent": "viz_agent",
        "render_plotly": "render_plotly",
        "viz_approval": "viz_approval",
        "researcher": "researcher",
        "forecaster": "forecaster",
        "__end__": "__end__",
    }
)

# Flujos de retorno al supervisor
builder.add_edge("planner", "supervisor")
builder.add_edge("sql_agent", "supervisor")
builder.add_edge("analyst", "supervisor")
builder.add_edge("viz_agent", "supervisor")
builder.add_edge("render_plotly", "supervisor")
builder.add_edge("viz_approval", "supervisor")
builder.add_edge("researcher", "supervisor")
builder.add_edge("forecaster", "supervisor")

memory = MemorySaver()
BI_ORCHESTRATOR = builder.compile(checkpointer=memory, interrupt_before=["viz_approval"])
