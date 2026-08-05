# core/orchestrator.py
# -------------------------------------------------

import logging
import re  # <-- NUEVO: para detección de chitchat
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
# NOTE: run_forecast se importa lazy dentro de forecaster_node
from core.llm import LLM
from core.contracts import SQLContract

from core.harness import build_harness_context, build_harness_context_cached, _normalize_question
from core.database import get_semantic_schema_for_views

from langsmith import traceable

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# NUEVO: Detección de chitchat (saludos, despedidas, gracias, identidad)
# ------------------------------------------------------------------
_CHITCHAT_PATTERNS = [
    r'^(hola|buenos días|buenas tardes|buenas noches|hey|hi|hello)\b',
    r'^(cómo estás|como estas|qué tal|que tal|todo bien)\b',
    r'^(quién eres|quien eres|qué eres|que eres|qué puedes hacer|que puedes hacer|para qué sirves)\b',
    r'^(gracias|muchas gracias|ok|okey|vale|entendido)\b',
    r'^(adiós|adios|hasta luego|nos vemos|chao|bye)\b',
]

_BUSINESS_MARKERS = [
    "venta", "ventas", "vendido", "vender", "unidades", "sede", "producto",
    "categoría", "categoria", "sql", "query", "reporte", "informe", "forecast",
    "pronóstico", "pronosticar", "demanda", "ticket", "promedio", "canje",
    "canjes", "cortesía", "cortesia", "fidelización", "puntos", "cliente",
    "mes", "año", "semana", "ayer", "hoy", "mañana"
]


def _is_chitchat(question: str) -> bool:
    """
    Detecta si una pregunta es puramente conversacional.
    Es conservador: si la frase contiene palabras de negocio, NO es chitchat.
    """
    q = question.lower().strip()

    # ¿Cumple algún patrón de chitchat?
    is_match = any(re.search(p, q) for p in _CHITCHAT_PATTERNS)

    # Si además contiene palabras de negocio, no es chitchat puro
    has_business = any(marker in q for marker in _BUSINESS_MARKERS)

    return is_match and not has_business


def _generate_chitchat_response(question: str) -> str:
    q = question.lower().strip()

    if re.search(r'^(hola|buenos días|buenas tardes|buenas noches|hey|hi|hello)', q):
        return "¡Hola! Soy tu asistente de análisis de datos. ¿En qué puedo ayudarte hoy?"
    if re.search(r'^(cómo estás|como estas|qué tal|que tal)', q):
        return "¡Estoy listo para ayudarte! ¿Qué información necesitas consultar?"
    if re.search(r'^(quién eres|quien eres|qué eres|que eres|qué puedes hacer|que puedes hacer|para qué sirves)', q):
        return "Soy un asistente BI que puede consultar ventas, productos, sedes, fidelización, cortesías y más. ¿Qué te gustaría saber?"
    if re.search(r'^(gracias|muchas gracias)', q):
        return "¡Con gusto! Estoy aquí para lo que necesites."
    if re.search(r'^(adiós|adios|hasta luego|nos vemos|chao|bye)', q):
        return "¡Hasta luego! Que tengas un buen día."
    return "¡Hola! ¿En qué puedo ayudarte?"




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
    next: Optional[str]
    harness_context: Optional[Dict[str, Any]]
    semantic_context: str
    allowed_views: List[str]
    preferred_view: Optional[str]
    schema_info: str
    research_findings: Optional[str]
    forecast_request: Optional[Dict[str, Any]]
    forecast_results: Optional[List[Dict[str, Any]]]
    forecast_error: Optional[str]
    render_attempts: int
    is_chitchat: Optional[bool]      # <-- NUEVO


def _to_dict(obj: Any) -> Dict[str, Any]:
    """Normaliza objetos Pydantic a dicts planos para evitar fricciones en el estado."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "dict"):
        return obj.dict()
    return vars(obj)


# ------------------------------------------------------------------
# NUEVO: Nodos de chitchat
# ------------------------------------------------------------------
def detect_chitchat_node(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "is_chitchat": _is_chitchat(state.get("question", ""))
    }


def chitchat_node(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state.get("question", "")
    response = _generate_chitchat_response(question)

    logger.info(f"[Chitchat] Pregunta='{question}' → Respuesta predefinida")

    return {
        "final_answer": response,
        "last_agent": "chitchat",
        "messages": state.get("messages", []) + [
            AIMessage(content=f"[Chitchat] {response}")
        ]
    }


@traceable(name="Orchestrator: Build Harness Context")
def build_harness_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # FIX: fallback si question no viene en el estado
    question = state.get("question", "")
    harness = build_harness_context_cached(_normalize_question(question))

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
    logger.info(f"[SQL Agent Wrapper] Iniciando. Plan presente: {state.get('plan') is not None}")

    if not state.get("plan"):
        logger.warning("[SQL Agent Wrapper] No hay plan. Devolviendo error.")
        return {
            "sql_results": [SQLContract(
                status="error",
                error_message="SQL Agent llamado sin plan previo.",
                can_answer=True
            )],
            "last_agent": "sql_agent",
            "messages": [AIMessage(content="[SQL Agent] Error: sin plan previo")]
        }

    tasks = state["plan"].tasks
    logger.info(f"[SQL Agent Wrapper] Tareas en plan: {len(tasks) if tasks else 0}")

    if not tasks:
        logger.warning("[SQL Agent Wrapper] Plan sin tareas. Devolviendo error.")
        return {
            "sql_results": [SQLContract(
                status="error",
                error_message="El plan no contiene tareas SQL.",
                can_answer=True
            )],
            "last_agent": "sql_agent",
            "messages": [AIMessage(content="[SQL Agent] Plan sin tareas SQL.")]
        }

    results: List[SQLContract] = []
    harness_ctx = state.get("harness_context", {})
    global_semantic_context = state.get("semantic_context", harness_ctx.get("semantic_context", ""))

    for idx, task in enumerate(tasks):
        logger.info(f"[SQL Agent Wrapper] Ejecutando tarea {idx + 1}/{len(tasks)}")
        payload = task.dict() if hasattr(task, "dict") else dict(task)

        candidate_views = getattr(task, "candidate_views", None) or state.get("allowed_views", [])
        preferred = getattr(task, "preferred_view", None) or state.get("preferred_view")

        try:
            schema_info = get_semantic_schema_for_views(candidate_views)
        except Exception as e:
            logger.warning(f"[SQL Agent Wrapper] Error obteniendo schema: {e}")
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

        try:
            logger.info(f"[SQL Agent Wrapper] Invocando SQL_SUBGRAPH para tarea {idx + 1}")
            sub_result = SQL_SUBGRAPH.invoke(sub_input)
            logger.info(f"[SQL Agent Wrapper] Subgrafo devolvió keys: {list(sub_result.keys()) if isinstance(sub_result, dict) else 'NO ES DICT'}")
            contract = sub_result.get("contract")
            logger.info(f"[SQL Agent Wrapper] Contract: {contract}")

            if contract is None:
                logger.warning("[SQL Agent Wrapper] Subgrafo no devolvió contract. Creando error.")
                contract = SQLContract(
                    status="error",
                    error_message="Subgrafo SQL no devolvió contrato",
                    needs_followup=True
                )
        except Exception as e:
            logger.error(f"[SQL Agent Wrapper] Excepción en subgrafo: {e}", exc_info=True)
            contract = SQLContract(
                status="error",
                error_message=f"Excepción en subgrafo SQL: {str(e)}",
                needs_followup=True
            )

        contract.task_id = getattr(task, "task_id", idx + 1)
        contract.allowed_views = candidate_views
        contract.preferred_view = preferred
        contract.semantic_context_used = (
            global_semantic_context[:500] + "..."
            if len(global_semantic_context) > 500
            else global_semantic_context
        )

        results.append(contract)
        logger.info(f"[SQL Agent Wrapper] Tarea {idx + 1} finalizada: status={contract.status} can_answer={contract.can_answer} rows={contract.row_count}")

    all_success = all(
        r.status in ("success", "partial") and r.can_answer
        for r in results
    )
    summary = " | ".join([
        f"T{r.task_id}:{r.status}({r.row_count})"
        for r in results
    ])

    logger.info(f"[SQL Agent Wrapper] Resumen: OK={all_success} | {summary}")
    logger.info(f"[SQL Agent Wrapper] sql_results tiene {len(results)} items")

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
    sql_results = state.get("sql_results", []) or []
    plan = state.get("plan")

    if not sql_results:
        logger.warning("[Viz Agent Node] No hay resultados SQL")
        return {
            "viz_result": None,
            "last_agent": "viz_agent",
            "messages": [AIMessage(content="[Viz Agent] Sin datos SQL para visualizar")]
        }

    primary_result = next(
        (
            r for r in sql_results
            if _get(r, "status") in ("success", "partial")
            and _get(r, "row_count", 0) > 0
        ),
        sql_results[0]
    )

    rows = _get(primary_result, "rows", []) or []
    columns = _get(primary_result, "columns", []) or []

    chart_type_hint = "auto"
    if plan:
        if isinstance(plan, dict):
            chart_type_hint = plan.get("chart_type_hint") or "auto"
        else:
            chart_type_hint = getattr(plan, "chart_type_hint", None) or "auto"

    viz_input = {
        "question": state["question"],
        "sql_rows": rows,
        "sql_columns": columns,
        "chart_type_hint": chart_type_hint,
        "messages": [],
        "figure_spec": None,
        "error_message": "",
        "attempts": 0,
        "contract": None,
    }

    try:
        viz_result = VIZ_SUBGRAPH.invoke(viz_input)
        contract = viz_result.get("contract") if isinstance(viz_result, dict) else None

        if contract is not None and not isinstance(contract, dict):
            contract = contract.dict() if hasattr(contract, "dict") else vars(contract)

        if contract and contract.get("status") == "error":
            logger.warning(f"[Viz Agent Node] Spec no viable: {contract.get('reasoning')}")
            return {
                "viz_result": contract,
                "last_agent": "viz_agent",
                "messages": [AIMessage(content="[Viz Agent] Los datos no son aptos para visualización.")]
            }

        if contract:
            figure_spec = contract.get("figure_spec") or {}
            if figure_spec.get("z_axis") and not contract.get("z_axis"):
                contract["z_axis"] = figure_spec["z_axis"]
            if figure_spec.get("type") and not contract.get("chart_type"):
                contract["chart_type"] = figure_spec["type"]

        return {
            "viz_result": contract,
            "last_agent": "viz_agent",
            "messages": [AIMessage(content=f"[Viz Agent] Spec generada: {contract.get('chart_type') if contract else None}")]
        }

    except Exception as e:
        logger.error(f"[Viz Agent Node] Error en subgrafo: {e}", exc_info=True)
        return {
            "viz_result": None,
            "last_agent": "viz_agent",
            "messages": [AIMessage(content=f"[Viz Agent] Error generando spec: {e}")]
        }


def _get(obj: Any, attr: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


@traceable(name="Orchestrator: Execute Demand Forecast")
def forecaster_node(state: Dict[str, Any]) -> Dict[str, Any]:
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


def researcher_node(state: Dict[str, Any]) -> Dict[str, Any]:
    from agents.research.research_node import make_research_node
    node = make_research_node(SQL_SUBGRAPH, LLM)
    return node(state)


builder = StateGraph(OrchestratorState)

builder.add_node("build_harness", build_harness_context_node)
builder.add_node("detect_chitchat", detect_chitchat_node)  # <-- NUEVO
builder.add_node("chitchat", chitchat_node)                # <-- NUEVO
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

# NUEVO: corte de chitchat antes del supervisor
builder.add_edge("build_harness", "detect_chitchat")

builder.add_conditional_edges(
    "detect_chitchat",
    lambda state: "chitchat" if state.get("is_chitchat") else "supervisor",
    {
        "chitchat": "chitchat",
        "supervisor": "supervisor",
    }
)

builder.add_edge("chitchat", "__end__")

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
