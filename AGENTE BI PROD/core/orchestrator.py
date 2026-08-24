# core/orchestrator.py
# -------------------------------------------------
# Cambios aplicados (alineación con sql_agent sin SEMANTIC_MAP):
#  4a Conjuntos de éxito incluyen "no_data" (0 filas exitosas = respuesta terminal,
#     ya no se reejecutan) — en is_successful y en los dos all_success
#  4b B3: el subgrafo recibe previous_sql/error/row_count (corrección accionable);
#     payload con model_dump() (era .dict(), deprecado en Pydantic v2)
#  4c Early-return filtrado al plan vigente (sin resultados huérfanos de otros planes)
#  4d Instrucción dinámica del sql_agent referencia la TAREA FALLIDA (no la última);
#     el branch de "0 filas" queda eliminado (muerto tras status="no_data")
# -------------------------------------------------
import hashlib
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
# orchestrator.py línea 27:
from agents.sql_agent.graph_sql_agent import SQL_SUBGRAPH

from agents.viz_agent.graph_viz_agent import VIZ_SUBGRAPH
from agents.viz_agent.render_node import render_plotly_node
from agents.viz_approval.graph_viz_approval import viz_approval_node
from agents.research.research_node import make_research_node

from core.llm import LLM
from core.contracts import SQLContract, ForecastRequest, TaskLedger, ProgressLedger, FactItem
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
    cae gracefulmente a MemorySaver para desarrollo local/tests.
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
# Helpers utilitarios
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
    return vars(obj)


def _ensure_ledger(obj: Any, model_class: type) -> Any:
    """Convierte un dict serializado de vuelta a Pydantic si es necesario."""
    if obj is None:
        return model_class()
    if isinstance(obj, model_class):
        return obj
    if isinstance(obj, dict):
        return model_class(**obj)
    return model_class(**_to_dict(obj))


def _compute_state_hash(state: "OrchestratorState") -> str:
    """
    Hash representativo del progreso real del sistema.
    Cambia cuando cambian outputs significativos (SQL, viz, forecast, findings).
    """
    plan = state.get("plan")
    question_type = _get(plan, "question_type", "none") if plan else "none"

    sql_results = state.get("sql_results", []) or []
    sql_summary = tuple(
        (
            _get(r, "task_id", idx),
            _get(r, "status"),
            _get(r, "row_count", 0),
            _get(r, "can_answer", False),
        )
        for idx, r in enumerate(sql_results)
    )

    viz = state.get("viz_result")
    viz_summary = (
        _get(viz, "status", "none"),
        _get(viz, "chart_type", "none"),
        _get(viz, "suitable_for_visualization", False),
    )

    key = (
        question_type,
        sql_summary,
        viz_summary,
        state.get("viz_rendered", False),
        state.get("viz_approved"),
        bool(state.get("final_answer")),
        bool(state.get("research_findings")),
        bool(state.get("forecast_results")),
        state.get("last_agent"),
        state.get("forecast_error"),
    )
    return hashlib.md5(str(key).encode()).hexdigest()


def _build_dynamic_instruction(state: "OrchestratorState", next_step: str) -> str:
    """
    Genera una instrucción contextual para el siguiente agente.
    Puede ser heurística o, en una versión posterior, generada por LLM.
    """
    plan = state.get("plan")
    question_type = _get(plan, "question_type", "general") if plan else "general"
    sql_results = state.get("sql_results", []) or []
    progress = state.get("progress_ledger") or {}

    if next_step == "planner":
        if progress.get("stall_count", 0) > 0:
            return (
                "El plan anterior no generó progreso. Reformúlalo: simplifica la pregunta, "
                "cambia de vista semántica, divide en subtareas más pequeñas, o ajusta filtros de fecha/producto/sede."
            )
        return (
            "Genera un plan de ejecución claro para responder la pregunta del usuario "
            "usando exclusivamente las vistas semánticas autorizadas."
        )

    if next_step == "sql_agent":
        # (4d) con status="no_data", los 0-filas ya NO se reintentan:
        # solo errores reales re-rutean aquí. Se referencia la tarea FALLIDA
        # (no la última del lote, que es impreciso en multi-query).
        failed_sql = next(
            (r for r in reversed(sql_results) if _get(r, "status") == "error"),
            None,
        )
        if failed_sql:
            return (
                f"La consulta de la tarea {_get(failed_sql, 'task_id', '?')} falló: "
                f"{_get(failed_sql, 'error_message', 'error desconocido')}. "
                f"Tienes el intento anterior como referencia en el contexto: NO lo repitas; "
                f"corrige usando las columnas válidas del catálogo o elige otra vista."
            )
        return "Ejecuta las tareas SQL del plan actual y devuelve un contrato por cada una."

    if next_step == "analyst":
        if question_type == "demand_forecast" and state.get("forecast_results"):
            return "Resume los resultados del pronóstico de demanda en lenguaje claro y accionable."
        if state.get("research_findings"):
            return "El researcher generó un informe detallado. Entrega una respuesta final concisa basada en esos hallazgos."
        if not sql_results:
            return "No hay datos disponibles. Explica por qué no se puede responder y qué información haría falta."
        if not all(_get(r, "can_answer", False) for r in sql_results):
            return (
                "Algunas consultas no pudieron responder completamente. "
                "Genera una respuesta parcial o estimación razonada (educated guess) con los datos disponibles, "
                "señalando claramente los límites."
            )
        return "Genera la respuesta final en lenguaje natural a partir de los resultados SQL."

    if next_step == "viz_agent":
        return "Genera una especificación de visualización adecuada para los datos SQL obtenidos."

    if next_step == "render_plotly":
        return "Renderiza la especificación de visualización en una figura Plotly."

    if next_step == "viz_approval":
        return "Presenta la visualización al usuario para aprobación o rechazo."

    if next_step == "researcher":
        return (
            "Realiza una exploración profunda con múltiples queries SQL autocontenidas "
            "y genera un informe ejecutivo completo."
        )

    if next_step == "forecaster":
        return "Ejecuta el pronóstico de demanda con los parámetros estructurados del plan."

    return "Procesa la tarea asignada con criterio de calidad y trazabilidad."


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
        return "¡Hola! Soy tu Capo. ¿En qué puedo ayudarte hoy?"
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

    # Ledger general del supervisor
    task_ledger: Optional[TaskLedger]
    progress_ledger: Optional[ProgressLedger]
    next_agent_instruction: Optional[str]

    # Campos operativos
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

    # Señales estructuradas de control (2.2)
    is_replan: Optional[bool]
    replan_reason: Optional[str]


# ------------------------------------------------------------------
# Decorador de resiliencia para nodos worker
# ------------------------------------------------------------------
def resilient_node(node_name: str):
    def decorator(
        fn: Callable[[OrchestratorState], OrchestratorState]
    ) -> Callable[[OrchestratorState], OrchestratorState]:
        @wraps(fn)
        def wrapper(state: OrchestratorState, **kwargs) -> OrchestratorState:
            try:
                return fn(state, **kwargs)
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
def detect_chitchat_node(state: OrchestratorState, **kwargs) -> OrchestratorState:
    return {"is_chitchat": _is_chitchat(state.get("question"))}


def chitchat_node(state: OrchestratorState, **kwargs) -> OrchestratorState:
    question = state.get("question", "")
    response = _generate_chitchat_response(question)
    logger.info(f"[Chitchat] Pregunta='{question}' → Respuesta predefinida")

    return {
        "final_answer": response,
        "last_agent": "chitchat",
        "messages": [AIMessage(content=f"[Chitchat] {response}")]
    }


@traceable(name="Orchestrator: Build Harness Context")
def build_harness_context_node(state: OrchestratorState, **kwargs) -> OrchestratorState:
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


# ------------------------------------------------------------------
# Ejecución de tareas SQL (multi-query) con reuse + intento previo
# ------------------------------------------------------------------
@traceable(name="Orchestrator: Execute SQL Tasks")
def sql_agent_wrapper(state: "OrchestratorState", **kwargs) -> "OrchestratorState":
    logger.info(f"[SQL Agent Wrapper] Iniciando. Plan presente: {state.get('plan') is not None}")

    instruction = state.get("next_agent_instruction")
    if instruction:
        logger.info(f"[SQL Agent] Instrucción recibida: {instruction}")

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
    tasks = _get(plan, "tasks", []) or []

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

    # ------------------------------------------------------------------
    # MULTI_QUERY: reutilizar resultados previos terminados con éxito
    # ------------------------------------------------------------------
    existing_results = state.get("sql_results", []) or []
    results_by_id: Dict[str, SQLContract] = {}

    for r in existing_results:
        tid = str(_get(r, "task_id", ""))
        if tid:
            results_by_id[tid] = r

    # IDs del plan vigente (versionados v{n}-t{i}) → filtro estricto (4c)
    plan_ids = {str(_get(t, "task_id", "")) for t in tasks}

    tasks_to_run = []
    for t in tasks:
        tid = str(_get(t, "task_id", ""))
        res = results_by_id.get(tid)

        # (4a) "no_data" cuenta como terminal-éxito: 0 filas != fallo
        is_successful = (
            res is not None
            and _get(res, "status") in ("success", "partial", "no_data")
            and _get(res, "can_answer", False)
        )

        if not is_successful:
            tasks_to_run.append(t)

    harness_ctx = state.get("harness_context", {})
    global_semantic_context = state.get("semantic_context", harness_ctx.get("semantic_context", ""))

    # Si no hay nada pendiente, devolver SOLO los resultados del plan vigente
    if not tasks_to_run:
        # (4c) antes devolvía TODOS los results_by_id (incl. huérfanos de otros planes)
        current_results = [r for tid, r in results_by_id.items() if tid in plan_ids]
        all_success = all(
            _get(r, "status") in ("success", "partial", "no_data")  # (4a)
            and _get(r, "can_answer", False)
            for r in current_results
        )
        summary = " | ".join([
            f"T{_get(r, 'task_id')}:{_get(r, 'status')}({_get(r, 'row_count', 0)})"
            for r in current_results
        ])
        return {
            "sql_results": current_results,
            "last_agent": "sql_agent",
            "messages": [
                AIMessage(content=f"[SQL Agent] 0 tareas nuevas. OK={all_success} | {summary}")
            ]
        }

    for idx, task in enumerate(tasks_to_run):
        payload = task.model_dump() if hasattr(task, "model_dump") else dict(task)  # (4b) era .dict()
        candidate_views = getattr(task, "candidate_views", None) or state.get("allowed_views", [])
        preferred = getattr(task, "preferred_view", None) or state.get("preferred_view")

        try:
            schema_info = get_semantic_schema_for_views(candidate_views)
        except Exception as e:
            logger.warning(f"[SQL Agent Wrapper] Error obteniendo schema: {e}")
            schema_info = state.get("schema_info", "")

        # NUEVO (4b/B3): el subgrafo VE su intento anterior → corrección accionable
        tid_str = str(_get(task, "task_id", "") or "")
        prev_contract = results_by_id.get(tid_str)

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
            "attempts": 0,
            "supervisor_instruction": instruction,
            # B3: referencia del intento previo (vacía en la primera ejecución)
            "previous_sql": _get(prev_contract, "generated_sql", "") or "",
            "previous_error": _get(prev_contract, "error_message") or "",
            "previous_row_count": _get(prev_contract, "row_count", 0) or 0,
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

        task_id = getattr(task, "task_id", idx + 1)
        contract.task_id = task_id
        contract.allowed_views = candidate_views
        contract.preferred_view = preferred
        contract.semantic_context_used = (
            global_semantic_context[:500] + "..."
            if len(global_semantic_context) > 500
            else global_semantic_context
        )

        results_by_id[str(task_id)] = contract
        logger.info(
            f"[SQL Agent Wrapper] Tarea {task_id} finalizada: "
            f"status={contract.status} can_answer={contract.can_answer} rows={contract.row_count}"
        )

    final_results = [
        results_by_id[str(_get(t, "task_id"))]
        for t in tasks
        if str(_get(t, "task_id")) in results_by_id
    ]

    all_success = all(
        _get(r, "status") in ("success", "partial", "no_data")  # (4a)
        and _get(r, "can_answer", False)
        for r in final_results
    )
    summary = " | ".join([f"T{r.task_id}:{r.status}({r.row_count})" for r in final_results])

    return {
        "sql_results": final_results,
        "last_agent": "sql_agent",
        "messages": [
            AIMessage(content=f"[SQL Agent] {len(final_results)} tareas en total. OK={all_success} | {summary}")
        ]
    }


def viz_agent_node(state: OrchestratorState, **kwargs) -> OrchestratorState:
    sql_results = state.get("sql_results", []) or []
    plan = state.get("plan")

    instruction = state.get("next_agent_instruction")
    if instruction:
        logger.info(f"[Viz Agent] Instrucción recibida: {instruction}")

    if not sql_results:
        return {
            "viz_result": None,
            "last_agent": "viz_agent",
            "messages": [AIMessage(content="[Viz Agent] Sin datos SQL para visualizar")]
        }

    # "no_data" tiene row_count=0 → nunca se elige como primary (correcto: sin filas no hay gráfico)
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
        "supervisor_instruction": instruction,
    }

    try:
        viz_result = VIZ_SUBGRAPH.invoke(viz_input)
        contract = viz_result.get("contract") if isinstance(viz_result, dict) else None
        if contract and not isinstance(contract, dict):
            contract = contract.model_dump() if hasattr(contract, "model_dump") else vars(contract)

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
def forecaster_node(state: OrchestratorState, **kwargs) -> OrchestratorState:
    from agents.forecasting_agent.graph_demand_forecaster import run_forecast

    instruction = state.get("next_agent_instruction")
    if instruction:
        logger.info(f"[Forecaster] Instrucción recibida: {instruction}")

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


def researcher_node(state: OrchestratorState, **kwargs) -> OrchestratorState:
    instruction = state.get("next_agent_instruction")
    if instruction:
        logger.info(f"[Researcher] Instrucción recibida: {instruction}")
    return _RESEARCHER_NODE(state)


def safe_supervisor_node(state: OrchestratorState, **kwargs) -> OrchestratorState:
    """
    Supervisor envuelto con:
    - Task Ledger y Progress Ledger
    - Stall detection real por hash de estado
    - Replanificación automática tras >2 stalls
      (limpia artefactos del plan muerto + señal estructurada is_replan)
    - Educated guess como salida válida
    - Instrucciones dinámicas para el siguiente agente
    """
    max_iterations = 20
    current_count = state.get("iteration_count", 0)

    # Recuperar o inicializar ledgers
    task_ledger = _ensure_ledger(state.get("task_ledger"), TaskLedger)
    progress_ledger = _ensure_ledger(state.get("progress_ledger"), ProgressLedger)

    # FIX 1: Actualizar Task Ledger SIEMPRE a partir del plan actual
    question = state.get("question", "")
    if question and not task_ledger.original_question:
        task_ledger.original_question = question

    plan = state.get("plan")
    if plan:
        task_ledger.plan = plan
        task_ledger.question_type = _get(plan, "question_type")
        task_ledger.intent = _get(plan, "intent", "")

        # Recalcular facts verificados del plan actual (replanificación puede cambiarlos)
        facts_verified: List[FactItem] = []
        metrics = _get(plan, "metrics", []) or []
        dimensions = _get(plan, "dimensions", []) or []
        filters = _get(plan, "filters", "")
        time_window = _get(plan, "time_window")

        for m in metrics:
            facts_verified.append(FactItem(content=f"Métrica objetivo: {m}", source="plan", verified=True))
        for d in dimensions:
            facts_verified.append(FactItem(content=f"Dimensión objetivo: {d}", source="plan", verified=True))
        if filters:
            facts_verified.append(FactItem(content=f"Filtros: {filters}", source="plan", verified=True))
        if time_window:
            facts_verified.append(FactItem(content=f"Ventana temporal: {time_window}", source="plan", verified=True))

        task_ledger.facts_verified = facts_verified

    # Guard absoluto de iteraciones
    if current_count >= max_iterations:
        logger.warning(f"[Ledger Guard] Límite de iteraciones ({max_iterations}) alcanzado.")
        return {
            "next": "__end__",
            "final_answer": state.get("final_answer") or "Se alcanzó el límite de pasos de razonamiento.",
            "iteration_count": current_count + 1,
            "last_agent": "supervisor",
            "task_ledger": _to_dict(task_ledger),
            "progress_ledger": _to_dict(progress_ledger),
            "messages": [AIMessage(
                content=f"[Supervisor] Límite de {max_iterations} iteraciones alcanzado. Forzando cierre."
            )]
        }

    # Stall detection real
    current_hash = _compute_state_hash(state)
    if progress_ledger.last_state_hash == current_hash:
        progress_ledger.stall_count += 1
    else:
        progress_ledger.stall_count = 0
        progress_ledger.last_state_hash = current_hash

    progress_ledger.unproductive_loop_detected = progress_ledger.stall_count >= 2

    logger.info(
        f"[Ledger] Iteración {current_count} | hash={current_hash[:8]} | "
        f"stall_count={progress_ledger.stall_count} | last_agent={state.get('last_agent')}"
    )

    # ------------------------------------------------------------------
    # Replanificación automática tras stall > 2
    # ------------------------------------------------------------------
    if progress_ledger.stall_count > 2:
        stall_count_at_replan = progress_ledger.stall_count  # capturar antes de resetear
        logger.warning(f"[Ledger] Stall detectado ({stall_count_at_replan}). Replanificando...")

        progress_ledger.completed_steps.append("replan")
        progress_ledger.stall_count = 0
        progress_ledger.last_state_hash = None

        instruction = (
            "Reformula el plan desde cero. Los agentes anteriores no lograron avanzar. "
            "Considera simplificar la consulta, cambiar de vista semántica, o dividir la pregunta."
        )
        return {
            "next": "planner",
            "plan": None,

            # --- Limpieza de artefactos del plan muerto ------------------
            "sql_results": [],
            "viz_result": None,
            "viz_approved": None,        # evita que un approval obsoleto salte el HITL
            "viz_rendered": False,
            "forecast_request": None,    # evita re-ejecutar un forecast ya fallido
            "forecast_results": None,
            # OJO: "forecast_error" SE CONSERVA deliberadamente — el planner
            # lo necesita (fix 2.4): con is_replan=True + forecast_error presente,
            # devuelve needs_followup con el motivo en vez de regenerar los
            # mismos parámetros y caer en el mismo error.
            # --------------------------------------------------------------

            # --- Señal estructurada de replan (2.2) ----------------------
            "is_replan": True,
            "replan_reason": f"Stall x{stall_count_at_replan} sin progreso",
            # --------------------------------------------------------------

            "next_agent_instruction": instruction,
            "iteration_count": current_count + 1,
            "last_agent": "supervisor",
            "task_ledger": _to_dict(task_ledger),
            "progress_ledger": _to_dict(progress_ledger),
            "messages": [AIMessage(
                content=f"[Supervisor] Replanificando tras {stall_count_at_replan} iteraciones sin progreso."
            )]
        }

    # Delegar routing al supervisor interno
    try:
        result = supervisor_node(state)
        if not isinstance(result, dict):
            result = _to_dict(result)

        raw_next = result.get("next") or result.get("next_agent", "__end__")
        if raw_next == "FINISH":
            raw_next = "__end__"

        if raw_next not in VALID_NEXT_STEPS:
            logger.warning(f"[Supervisor] next='{raw_next}' inválido. Fallback a __end__.")
            raw_next = "__end__"
            if not result.get("final_answer"):
                result["final_answer"] = "No pude determinar el siguiente paso del análisis."

        result["next"] = raw_next

    except Exception as e:
        logger.exception("[Supervisor] Error crítico en routing")
        return {
            "next": "__end__",
            "final_answer": "Ocurrió un error interno al coordinar los agentes.",
            "iteration_count": current_count + 1,
            "last_agent": "supervisor",
            "task_ledger": _to_dict(task_ledger),
            "progress_ledger": _to_dict(progress_ledger),
            "messages": [AIMessage(content=f"[Supervisor Error] {e}")]
        }

    # Educated guess como salida válida
    if result.get("next") == "__end__" and not state.get("final_answer"):
        has_some_data = (
            state.get("sql_results")
            or state.get("forecast_results")
            or state.get("research_findings")
        )
        if has_some_data:
            logger.info("[Ledger] Sin respuesta final pero hay datos. Pidiendo educated guess al analyst.")
            result["next"] = "analyst"
            result["final_answer"] = None
            instruction = (
                "No se pudo completar la tarea de forma definitiva. "
                "Genera una respuesta parcial o estimación razonada (educated guess) "
                "con los datos disponibles, explicando claramente los límites y gaps."
            )
            result["next_agent_instruction"] = instruction
            progress_ledger.instruction = instruction
        else:
            result["final_answer"] = (
                result.get("final_answer")
                or "No pude obtener datos suficientes para responder esta consulta."
            )

    # FIX 2: Respetar instrucción específica del supervisor interno; fallback heurístico
    if result.get("next") and result.get("next") != "__end__":
        instruction = result.get("next_agent_instruction") or _build_dynamic_instruction(state, result["next"])
        result["next_agent_instruction"] = instruction
        progress_ledger.instruction = instruction
        progress_ledger.next_agent = result["next"]

    # Actualizar Progress Ledger
    progress_ledger.completed_steps.append(result.get("next", "__end__"))
    result["progress_ledger"] = _to_dict(progress_ledger)
    result["task_ledger"] = _to_dict(task_ledger)
    result["iteration_count"] = current_count + 1
    result["last_agent"] = "supervisor"

    return result


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

# Flujo de entrada optimizado
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

# Todos los workers retornan al supervisor
for worker in ["planner", "sql_agent", "analyst", "viz_agent", "render_plotly", "viz_approval", "researcher", "forecaster"]:
    builder.add_edge(worker, "supervisor")

BI_ORCHESTRATOR = builder.compile(
    checkpointer=_build_checkpointer(),
    interrupt_before=["viz_approval"]
)
