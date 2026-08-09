# core/orchestrator.py
# -------------------------------------------------
import logging
import os
import re
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Annotated
from typing_extensions import TypedDict, Literal

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage

from agents.planner.graph_planner import planner_node
from agents.analyst.graph_analyst import analyst_node
from agents.supervisor.graph_supervisor import supervisor_node
from agents.sql_agent.graph_sql_agent import SQL_SUBGRAPH
from agents.viz_agent.graph_viz_agent import VIZ_SUBGRAPH
from agents.viz_agent.render_node import render_plotly_node
from agents.viz_approval.graph_viz_approval import viz_approval_node
from agents.research.research_node import make_research_node

from core.llm import LLM
from core.contracts import SQLContract, ForecastRequest
from core.harness import build_harness_context_cached, _normalize_question
from core.database import get_semantic_schema_for_views

from langsmith import traceable

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Checkpointer persistente (Postgres) con fallback a MemorySaver
# ------------------------------------------------------------------
def _build_checkpointer():
    """
    Configura PostgresSaver si existe POSTGRES_URI; de lo contrario,
    cae gracefully a MemorySaver para desarrollo local/tests.
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg_pool import ConnectionPool

        db_uri = os.getenv("POSTGRES_URI")
        if not db_uri:
            raise ValueError("POSTGRES_URI no está definida")

        pool = ConnectionPool(
            conninfo=db_uri,
            max_size=20,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        cp = PostgresSaver(pool)
        # cp.setup()  # Descomenta la primera vez para crear/actualizar tablas
        logger.info("[Checkpointer] PostgresSaver configurado.")
        return cp
    except Exception as e:
        logger.warning(f"[Checkpointer] Fallback a MemorySaver: {e}")
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()


# ------------------------------------------------------------------
# Helpers utilitarios (definidos al inicio del módulo)
# ------------------------------------------------------------------
def _get(obj: Any, attr: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return vars(obj)


# ------------------------------------------------------------------
# Patrones y lógica de Chitchat
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


def _is_chitchat(question: Optional[str]) -> bool:
    """Detecta si una pregunta es puramente conversacional."""
    q = (question or "").lower().strip()
    if not q:
        return False
    is_match = any(re.search(p, q) for p in _CHITCHAT_PATTERNS)
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


# ------------------------------------------------------------------
# Tipado estricto del estado del orquestador
# ------------------------------------------------------------------
NextStep = Literal[
    "planner", "sql_agent", "analyst", "viz_agent",
    "render_plotly", "viz_approval", "researcher",
    "forecaster", "__end__"
]
VALID_NEXT_STEPS = set(NextStep.__args__)


class OrchestratorState(TypedDict):
    question: str
    messages: Annotated[List[BaseMessage], add_messages]
    plan: Optional[Any]
    sql_results: List[SQLContract]
    viz_result: Optional[Any]
    viz_approved: Optional[bool]
    viz_rendered: bool
    final_answer: Optional[str]
    iteration_count: int
    last_agent: Optional[str]
    next: Optional[NextStep]
    harness_context: Optional[Dict[str, Any]]
    semantic_context: str
    allowed_views: List[str]
    preferred_view: Optional[str]
    schema_info: str
    research_findings: Optional[str]
    forecast_request: Optional[ForecastRequest]
    forecast_results: Optional[List[Dict[str, Any]]]
    forecast_error: Optional[str]
    render_attempts: int
    is_chitchat: Optional[bool]


# ------------------------------------------------------------------
# Decorador de resiliencia para nodos worker
# ------------------------------------------------------------------
def resilient_node(node_name: str):
    """
    Envuelve un nodo para capturar excepciones inesperadas.
    En caso de fallo, registra el error y devuelve un mensaje explicativo
    para que el supervisor decida cómo continuar.
    """
    def decorator(fn: Callable[[OrchestratorState], OrchestratorState]) -> Callable[[OrchestratorState], OrchestratorState]:
        @wraps(fn)
        def wrapper(state: OrchestratorState) -> OrchestratorState:
            try:
                return fn(state)
            except Exception as e:
                logger.exception(f"[{node_name}] Nodo falló inesperadamente")
                return {
                    "final_answer": f"Error interno en el agente {node_name}: {e}",
                    "messages": [AIMessage(content=f"[{node_name} Error] {e}")],
                    "last_agent": node_name,
                }
        return wrapper
    return decorator


# ------------------------------------------------------------------
# Nodos del Grafo
# ------------------------------------------------------------------
def detect_chitchat_node(state: OrchestratorState) -> OrchestratorState:
    return {
        "is_chitchat": _is_chitchat(state.get("question"))
    }


def chitchat_node(state: OrchestratorState) -> OrchestratorState:
    question = state.get("question", "")
    response = _generate_chitchat_response(question)
    logger.info(f"[Chitchat] Pregunta='{question}' → Respuesta predefinida")

    return {
        "final_answer": response,
        "last_agent": "chitchat",
        "messages": [AIMessage(content=f"[Chitchat] {response}")]
    }


@traceable(name="Orchestrator: Build Harness Context")
def build_harness_context_node(state: OrchestratorState) -> OrchestratorState:
    question = state.get("question", "")
    harness = build_harness_context_cached(_normalize_question(question))

    return {
        "harness_context": harness,
        "semantic_context": harness.get("semantic_context", ""),
        "allowed_views": harness.get("allowed_views", []),
        "preferred_view": harness.get("preferred_view"),
        "schema_info": "",
        "messages": [
            AIMessage(
                content=f"[Harness] Preferred: {harness.get('preferred_view')} | "
                        f"Allowed: {harness.get('allowed_views')} | "
                        f"Ambiguity: {harness.get('ambiguity_notes')}"
            )
        ]
    }


@traceable(name="Orchestrator: Execute SQL Tasks")
def sql_agent_wrapper(state: OrchestratorState) -> OrchestratorState:
    logger.info(f"[SQL Agent Wrapper] Iniciando. Plan presente: {state.get('plan') is not None}")

    if not state.get("plan"):
        return {
            "sql_results": [SQLContract(
                status="error",
                error_message="SQL Agent llamado sin plan previo.",
                can_answer=True
            )],
            "last_agent": "sql_agent",
            "messages": [AIMessage(content="[SQL Agent] Error: sin plan previo")]
        }

    plan = state["plan"]
    tasks = plan.tasks if hasattr(plan, "tasks") else plan.get("tasks", [])

    if not tasks:
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
            sub_result = SQL_SUBGRAPH.invoke(sub_input)
            contract = sub_result.get("contract") if isinstance(sub_result, dict) else None
            if contract is None:
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
        logger.info(
            f"[SQL Agent Wrapper] Tarea {idx + 1} finalizada: "
            f"status={contract.status} can_answer={contract.can_answer} rows={contract.row_count}"
        )

    all_success = all(r.status in ("success", "partial") and r.can_answer for r in results)
    summary = " | ".join([f"T{r.task_id}:{r.status}({r.row_count})" for r in results])

    return {
        "sql_results": results,
        "last_agent": "sql_agent",
        "messages": [
            AIMessage(content=f"[SQL Agent] {len(results)} tareas ejecutadas. OK={all_success} | {summary}")
        ]
    }


def viz_agent_node(state: OrchestratorState) -> OrchestratorState:
    sql_results = state.get("sql_results", []) or []
    plan = state.get("plan")

    if not sql_results:
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
        chart_type_hint = plan.get("chart_type_hint") if isinstance(plan, dict) else getattr(plan, "chart_type_hint", "auto")

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
        if contract and not isinstance(contract, dict):
            contract = contract.dict() if hasattr(contract, "dict") else vars(contract)

        if contract and contract.get("status") == "error":
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


@traceable(name="Orchestrator: Execute Demand Forecast")
def forecaster_node(state: OrchestratorState) -> OrchestratorState:
    from agents.forecasting_agent.graph_demand_forecaster import run_forecast

    request = state.get("forecast_request")
    if isinstance(request, dict):
        try:
            request = ForecastRequest(**request)
        except Exception as e:
            logger.error(f"[Forecaster] forecast_request inválido: {e}")
            request = None

    if not request:
        logger.error("[Forecaster] No hay forecast_request en el estado")
        return {
            "forecast_error": "No hay parámetros de forecast.",
            "final_answer": "No pude determinar el producto y la sede para el pronóstico.",
            "last_agent": "forecaster",
            "messages": [AIMessage(content="[Forecaster] Sin parámetros de predicción")]
        }

    try:
        result = run_forecast(
            producto=request.producto,
            sede=request.sede,
            n_dias=int(request.n_dias),
            fecha_inicio=request.fecha_inicio
        )
        forecasts = result.get("forecasts", []) if isinstance(result, dict) else []

        return {
            "forecast_results": forecasts,
            "forecast_error": None,
            "last_agent": "forecaster",
            "messages": [
                AIMessage(
                    content=f"[Forecaster] {len(forecasts)} días pronosticados para "
                            f"{request.producto} @ {request.sede}"
                )
            ],
        }

    except Exception as e:
        logger.error(f"[Forecaster] Error: {e}", exc_info=True)
        return {
            "forecast_error": str(e),
            "final_answer": f"Error al generar el pronóstico: {e}",
            "last_agent": "forecaster",
            "messages": [AIMessage(content=f"[Forecaster Error] {e}")]
        }


# Instanciación única del nodo de investigación para evitar overhead
_RESEARCHER_NODE = make_research_node(SQL_SUBGRAPH, LLM)

def researcher_node(state: OrchestratorState) -> OrchestratorState:
    return _RESEARCHER_NODE(state)


def safe_supervisor_node(state: OrchestratorState) -> OrchestratorState:
    """
    Supervisor con guard de iteraciones y validación defensiva del next.
    """
    max_iterations = 8
    current_count = state.get("iteration_count", 0)

    # Guard contra loops infinitos
    if current_count >= max_iterations:
        logger.warning(f"[Guard] Límite de iteraciones alcanzado ({current_count}). Forzando cierre.")
        return {
            "next": "__end__",
            "final_answer": state.get("final_answer") or "Se alcanzó el límite de pasos de razonamiento para esta consulta.",
            "iteration_count": current_count + 1,
            "messages": [AIMessage(content=f"[Guard] Máximo de iteraciones alcanzado ({max_iterations}).")]
        }

    try:
        result = supervisor_node(state)
        if not isinstance(result, dict):
            result = _to_dict(result)

        # Normaliza el campo de decisión: admite "next", "next_agent" o "FINISH"
        raw_next = result.get("next") or result.get("next_agent", "__end__")
        if raw_next == "FINISH":
            raw_next = "__end__"

        if raw_next not in VALID_NEXT_STEPS:
            logger.warning(f"[Supervisor] next='{raw_next}' inválido. Fallback a __end__.")
            raw_next = "__end__"
            if not result.get("final_answer"):
                result["final_answer"] = "No pude determinar el siguiente paso del análisis."

        result["next"] = raw_next
        result["iteration_count"] = current_count + 1
        return result

    except Exception as e:
        logger.exception("[Supervisor] Error crítico")
        return {
            "next": "__end__",
            "final_answer": "Ocurrió un error interno al coordinar los agentes.",
            "iteration_count": current_count + 1,
            "messages": [AIMessage(content=f"[Supervisor Error] {e}")]
        }


# ------------------------------------------------------------------
# Construcción del Grafo
# ------------------------------------------------------------------
builder = StateGraph(OrchestratorState)

builder.add_node("detect_chitchat", resilient_node("detect_chitchat")(detect_chitchat_node))
builder.add_node("chitchat", resilient_node("chitchat")(chitchat_node))
builder.add_node("build_harness", resilient_node("build_harness")(build_harness_context_node))
builder.add_node("supervisor", safe_supervisor_node)
builder.add_node("planner", resilient_node("planner")(planner_node))
builder.add_node("sql_agent", resilient_node("sql_agent")(sql_agent_wrapper))
builder.add_node("analyst", resilient_node("analyst")(analyst_node))
builder.add_node("viz_agent", resilient_node("viz_agent")(viz_agent_node))
builder.add_node("render_plotly", resilient_node("render_plotly")(render_plotly_node))
builder.add_node("viz_approval", resilient_node("viz_approval")(viz_approval_node))
builder.add_node("researcher", resilient_node("researcher")(researcher_node))
builder.add_node("forecaster", resilient_node("forecaster")(forecaster_node))

# Flujo de entrada optimizado: chitchat se detecta antes de cualquier procesamiento costoso
builder.add_edge("__start__", "detect_chitchat")
builder.add_conditional_edges(
    "detect_chitchat",
    lambda state: "chitchat" if state.get("is_chitchat") else "build_harness",
    {"chitchat": "chitchat", "build_harness": "build_harness"}
)

builder.add_edge("chitchat", "__end__")
builder.add_edge("build_harness", "supervisor")

# Supervisor ruteo dinámico
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

# Todos los workers retornan al supervisor para la siguiente decisión
for worker in ["planner", "sql_agent", "analyst", "viz_agent", "render_plotly", "viz_approval", "researcher", "forecaster"]:
    builder.add_edge(worker, "supervisor")

BI_ORCHESTRATOR = builder.compile(
    checkpointer=_build_checkpointer(),
    interrupt_before=["viz_approval"]
)
