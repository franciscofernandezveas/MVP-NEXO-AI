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
from core.llm import LLM
from core.contracts import SQLContract, FilterSpec

from core.harness import build_harness_context_cached, _normalize_question
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


def _task_to_payload(task: Any) -> Dict[str, Any]:
    """
    Serializa un SQLPayload a dict plano, asegurando que FilterSpec sean dicts.
    """
    if hasattr(task, "model_dump"):
        return task.model_dump()
    elif hasattr(task, "dict"):
        return task.dict()
    else:
        payload = dict(task)
        # Asegurar que filters sea lista de dicts
        raw_filters = payload.get("filters", [])
        if raw_filters and not isinstance(raw_filters[0], dict):
            payload["filters"] = [
                f.model_dump() if hasattr(f, "model_dump") else
                f.dict() if hasattr(f, "dict") else dict(f)
                for f in raw_filters
            ]
        return payload


def _extract_filters_from_plan(plan: Any) -> List[FilterSpec]:
    """
    Extrae filtros estructurados de un plan, manejando tanto List[FilterSpec]
    como filters_description legacy.
    """
    filters = getattr(plan, "filters", None)
    if filters:
        if isinstance(filters, list) and filters and isinstance(filters[0], FilterSpec):
            return filters
        if isinstance(filters, list):
            return [FilterSpec(**f) if isinstance(f, dict) else f for f in filters]

    filters_desc = getattr(plan, "filters_description", "")
    if not filters_desc:
        return []

    # Parseo legacy
    import re
    text_columns = {
        "nombre_sede", "sede", "sucursal", "local", "tienda", "plaza",
        "producto", "descripcion", "descripción", "categoria", "categoría"
    }
    result = []
    for part in re.split(r",\s*(?=\w+\s*[=<>])", filters_desc):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"(\w+)\s*([=<>]|ILIKE|LIKE|IN)\s*(.+)", part, re.IGNORECASE)
        if not m:
            continue
        col, op, val = m.groups()
        op = op.upper()
        vt = "string"
        if op == "IN":
            val = [v.strip().strip("'\"") for v in val.strip("()[]").split(",")]
            vt = "list"
        elif re.match(r"^\d+(\.\d+)?$", val.strip()):
            vt = "number"
        if col in text_columns and op in ("=", "LIKE"):
            op = "ILIKE"
        result.append(FilterSpec(column=col, operator=op, value=val, value_type=vt))
    return result


@traceable(name="Orchestrator: Execute SQL Tasks")
def sql_agent_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info(f"[SQL Agent Wrapper] Iniciando. Plan presente: {state.get('plan') is not None}")

    if not state.get("plan"):
        logger.warning("[SQL Agent Wrapper] No hay plan. Devolviendo error.")
        return {
            "sql_results": [SQLContract(
                status="error",
                error_message="SQL Agent llamado sin plan previo.",
                can_answer=False,
                needs_followup=True
            )],
            "last_agent": "sql_agent",
            "messages": [AIMessage(content="[SQL Agent] Error: sin plan previo")]
        }

    plan = state["plan"]
    tasks = plan.tasks
    logger.info(f"[SQL Agent Wrapper] Tareas en plan: {len(tasks) if tasks else 0}")

    if not tasks:
        logger.warning("[SQL Agent Wrapper] Plan sin tareas. Devolviendo error.")
        return {
            "sql_results": [SQLContract(
                status="error",
                error_message="El plan no contiene tareas SQL.",
                can_answer=False,
                needs_followup=True
            )],
            "last_agent": "sql_agent",
            "messages": [AIMessage(content="[SQL Agent] Plan sin tareas SQL.")]
        }

    results: List[SQLContract] = []
    harness_ctx = state.get("harness_context", {})
    global_semantic_context = state.get("semantic_context", harness_ctx.get("semantic_context", ""))

    for idx, task in enumerate(tasks):
        logger.info(f"[SQL Agent Wrapper] Ejecutando tarea {idx + 1}/{len(tasks)}")

        payload = _task_to_payload(task)
        task_id = getattr(task, "task_id", idx + 1)
        task_description = getattr(task, "task", "")

        candidate_views = getattr(task, "candidate_views", None) or state.get("allowed_views", [])
        preferred = getattr(task, "preferred_view", None) or state.get("preferred_view")

        if not candidate_views:
            logger.warning(f"[SQL Agent Wrapper] Tarea {task_id} sin candidate_views. Usando allowed_views globales.")
            candidate_views = state.get("allowed_views", [])

        if not preferred:
            logger.warning(f"[SQL Agent Wrapper] Tarea {task_id} sin preferred_view.")

        # ------------------------------------------------------------------
        # OBTENER ESQUEMA REAL DE LA VISTA PREFERIDA (no todas las candidatas)
        # ------------------------------------------------------------------
        views_for_schema = [preferred] if preferred else candidate_views
        try:
            schema_info = get_semantic_schema_for_views(views_for_schema)
            logger.info(f"[SQL Agent Wrapper] Schema obtenido para {views_for_schema}: {len(schema_info)} chars")
        except Exception as e:
            logger.warning(f"[SQL Agent Wrapper] Error obteniendo schema de {views_for_schema}: {e}")
            schema_info = state.get("schema_info", "")

        # La pregunta para el subgrafo SQL debe ser la TAREA técnica, no la pregunta global
        sub_question = task_description or state["question"]

        sub_input = {
            "question": sub_question,
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
            "attempts": 0,
            "schema_used": []
        }

        try:
            logger.info(f"[SQL Agent Wrapper] Invocando SQL_SUBGRAPH para tarea {task_id}")
            sub_result = SQL_SUBGRAPH.invoke(sub_input)
            logger.info(f"[SQL Agent Wrapper] Subgrafo devolvió keys: {list(sub_result.keys()) if isinstance(sub_result, dict) else 'NO ES DICT'}")

            contract = sub_result.get("contract")
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

        # Enriquecer contrato con metadatos de la subtarea
        contract.task_id = task_id
        if not contract.allowed_views:
            contract.allowed_views = candidate_views
        if not contract.preferred_view:
            contract.preferred_view = preferred
        if not contract.semantic_context_used:
            contract.semantic_context_used = (
                global_semantic_context[:500] + "..."
                if len(global_semantic_context) > 500
                else global_semantic_context
            )

        results.append(contract)
        logger.info(
            f"[SQL Agent Wrapper] Tarea {task_id} finalizada: "
            f"status={contract.status} can_answer={contract.can_answer} rows={contract.row_count}"
        )

    all_success = all(
        r.status in ("success", "partial") and r.can_answer
        for r in results
    )
    summary = " | ".join([
        f"T{r.task_id}:{r.status}({r.row_count})"
        for r in results
    ])

    logger.info(f"[SQL Agent Wrapper] Resumen: OK={all_success} | {summary}")

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



def _select_viz_source(state: Dict[str, Any]) -> tuple:
    """
    Selecciona las filas/columnas apropiadas para visualización.
    - Si hay una sola tarea exitosa, la usa.
    - Si hay múltiples tareas, usa la primera exitosa con filas.
    """
    sql_results = state.get("sql_results", [])
    if not sql_results:
        return [], []

    # Filtrar contratos exitosos con datos
    valid_results = [
        r for r in sql_results
        if r.status in ("success", "partial") and r.rows and len(r.rows) > 0
    ]

    if valid_results:
        source = valid_results[0]
    else:
        # Fallback: usar el primer resultado aunque tenga error
        source = sql_results[0]

    return source.rows, source.columns


def viz_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    plan = state.get("plan")
    sql_results = state.get("sql_results", [])

    # Verificar si visualización está habilitada
    if not getattr(plan, "visualization_candidate", False):
        logger.info("[Viz Agent] visualization_candidate=False. Omitiendo visualización.")
        return {
            "viz_result": None,
            "last_agent": "viz_agent",
            "messages": [AIMessage(content="[Viz Agent] Visualización omitida por configuración del plan")]
        }

    rows, columns = _select_viz_source(state)

    if not rows:
        logger.warning("[Viz Agent] No hay filas SQL válidas para visualizar.")
        return {
            "viz_result": None,
            "last_agent": "viz_agent",
            "messages": [AIMessage(content="[Viz Agent] No hay datos para visualizar")]
        }

    viz_input = {
        "question": state["question"],
        "sql_rows": rows,
        "sql_columns": columns,
        "chart_type_hint": getattr(plan, "chart_type_hint", "auto") if plan else "auto",
        "messages": [],
        "figure_spec": None,
        "error_message": "",
        "attempts": 0,
        "contract": None
    }

    viz_result = VIZ_SUBGRAPH.invoke(viz_input)
    return {
        "viz_result": viz_result.get("contract") if isinstance(viz_result, dict) else viz_result,
        "last_agent": "viz_agent",
        "messages": [AIMessage(content="[Viz Agent] Especificación de visualización generada")]
    }


@traceable(name="Orchestrator: Execute Demand Forecast")
def forecaster_node(state: Dict[str, Any]) -> Dict[str, Any]:
    from agents.forecasting_agent.graph_demand_forecaster import run_forecast

    logger.info(f"[Forecaster] Estado recibido. forecast_request={state.get('forecast_request')}")
    logger.info(f"[Forecaster] plan question_type={getattr(state.get('plan'), 'question_type', None)}")

    request = state.get("forecast_request")

    # Fallback desde plan si no hay forecast_request directo
    if not request:
        plan = state.get("plan")
        if plan and getattr(plan, "question_type", None) == "demand_forecast":
            filters = _extract_filters_from_plan(plan)

            producto = None
            sede = None
            n_dias = 7

            for f in filters:
                col_norm = f.column.lower().replace("_", " ")
                if col_norm in ("producto", "sku", "articulo", "artículo", "item"):
                    producto = f.value
                elif col_norm in ("sede", "sucursal", "local", "tienda", "plaza", "nombre_sede"):
                    sede = f.value
                elif col_norm in ("dias", "días", "n_dias", "num_dias"):
                    try:
                        n_dias = int(f.value)
                    except (ValueError, TypeError):
                        pass

            if producto and sede:
                logger.info(f"[Forecaster] Fallback desde plan: {producto} @ {sede}")
                request = {
                    "producto": producto,
                    "sede": sede,
                    "n_dias": n_dias,
                    "fecha_inicio": None,
                }

    # Fallback desde pregunta original
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
    """
    Wrapper lazy para el nodo de research.
    """
    node = make_research_node(SQL_SUBGRAPH, LLM)
    return node(state)


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
