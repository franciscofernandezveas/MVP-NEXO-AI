# core/graph_planner.py
# -------------------------------------------------
from typing import Any, Dict, List, Optional, Set
import json
import logging
import re
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from core.llm import LLM
from core.contracts import PlannerContract, SQLPayload
from core.harness import BusinessMemory
from core.semantic_retriever import (
    obtener_candidatas_detalles,
    payload_to_column_hints,
    seleccionar_vista_principal,
)

logger = logging.getLogger(__name__)

# Cargar catálogo una sola vez al importar el módulo
_biz_mem = BusinessMemory.from_file()


# ------------------------------------------------------------------
# NUEVO: Contexto conversacional para preguntas de seguimiento
# ------------------------------------------------------------------
def _build_conversational_context(messages: List[Any], current_question: str, max_turns: int = 4) -> str:
    """
    Construye un contexto conversacional reciente para que el Planner entienda
    preguntas de seguimiento o referencias implícitas.
    """
    if not messages:
        return current_question

    recent = messages[-max_turns * 2:]
    context_lines = []

    for msg in recent:
        role = "Usuario" if isinstance(msg, HumanMessage) else "Asistente"
        content = getattr(msg, "content", str(msg))
        if content:
            context_lines.append(f"{role}: {content[:300]}")

    if len(context_lines) <= 1:
        return current_question

    return (
        "Contexto reciente de la conversación:\n"
        + "\n".join(context_lines)
        + f"\n\nPregunta actual del usuario: {current_question}"
    )


# ------------------------------------------------------------------
# NUEVO: Soporte para instrucciones de replanificación del supervisor
# ------------------------------------------------------------------
def _is_replan_instruction(instruction: Optional[str]) -> bool:
    """Detecta si el supervisor pide una replanificación."""
    if not instruction:
        return False
    markers = [
        "reformula", "replan", "no generó progreso", "no lograron avanzar",
        "reformular", "plan alternativo", "nuevo plan", "replantear"
    ]
    return any(marker in instruction.lower() for marker in markers)


def _build_replan_context(state: Dict[str, Any]) -> str:
    """
    Construye un contexto estructurado con los hechos y el progreso previo
    para que el planner genere un plan alternativo cuando hay estancamiento.
    """
    task_ledger = state.get("task_ledger", {}) or {}
    progress_ledger = state.get("progress_ledger", {}) or {}
    sql_results = state.get("sql_results", []) or []

    parts: List[str] = []

    original_question = task_ledger.get("original_question") or state.get("question", "")
    parts.append(f"Pregunta original: {original_question}")

    completed_steps = progress_ledger.get("completed_steps", [])
    if completed_steps:
        parts.append(f"Pasos completados previamente: {completed_steps}")

    stall_count = progress_ledger.get("stall_count", 0)
    if stall_count:
        parts.append(f"Iteraciones sin progreso detectadas: {stall_count}")

    # Resumen de resultados SQL previos
    if sql_results:
        summaries = []
        for r in sql_results:
            summaries.append(
                f"- Tarea {_get(r, 'task_id')}: "
                f"status={_get(r, 'status')}, "
                f"filas={_get(r, 'row_count')}, "
                f"can_answer={_get(r, 'can_answer')}, "
                f"error={_get(r, 'error_message', 'ninguno')}"
            )
        parts.append("Resultados SQL previos:\n" + "\n".join(summaries))

    if state.get("forecast_error"):
        parts.append(f"Error previo en forecast: {state['forecast_error']}")

    if state.get("research_findings"):
        parts.append("Ya existen hallazgos de research previos; considéralos en el nuevo plan.")

    # Hechos verificados del Task Ledger
    facts = task_ledger.get("facts_verified", [])
    if facts:
        facts_text = "\n".join([f"- {f.get('content', '')}" for f in facts])
        parts.append(f"Hechos verificados hasta ahora:\n{facts_text}")

    return "\n\n".join(parts)


# ------------------------------------------------------------------
# Detección de demand forecast
# ------------------------------------------------------------------
FORECAST_KEYWORDS = [
    "pronosticar", "predicción", "predice", "forecast",
    "pronostico", "pronóstico", "demanda futura", "demanda proyectada",
    "cuánto se venderá", "cuanto se vendera", "cuánto venderemos",
    "cuanto venderemos", "proyección de demanda", "proyeccion de demanda",
    "estimar ventas", "pronosticar ventas"
]


class _ForecastParamsInternal(BaseModel):
    producto: str
    sede: str
    n_dias: int = 7
    fecha_inicio: Optional[str] = None


def _is_demand_forecast_question(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in FORECAST_KEYWORDS)


def _extract_forecast_params(question: str) -> Dict[str, Any]:
    parser = LLM.with_structured_output(_ForecastParamsInternal)
    try:
        result = parser.invoke(
            f"Extrae los parámetros para predecir demanda de la siguiente pregunta. "
            f"Si no hay fecha de inicio, devuelve null.\n\nPregunta: {question}"
        )
        if isinstance(result, dict):
            result = _ForecastParamsInternal(**result)
        return result.model_dump()
    except Exception as e:
        logger.warning(f"[Planner] Falló extracción estructurada de forecast: {e}")
        return {}


def _fallback_extract_forecast_params(question: str) -> Dict[str, Any]:
    q = question.lower()
    sedes = ["plaza bolsillo", "merced", "tajamar", "persa victor manuel"]
    sede_detectada = next((sede.title() for sede in sedes if sede in q), None)

    productos = [
        "americano", "capuccino", "latte", "espresso", "mokaccino",
        "cortado", "flat white", "iced latte", "chai latte", "chocolate caliente"
    ]
    producto_detectado = next((prod for prod in productos if prod in q), None)

    n_dias = 7
    dias_match = re.search(r"(\d+)\s*días?|(\d+)\s*dias?", q)
    if dias_match:
        n_dias = int(dias_match.group(1) or dias_match.group(2))

    return {
        "producto": producto_detectado,
        "sede": sede_detectada,
        "n_dias": n_dias,
        "fecha_inicio": None,
    }


def _normalize(text: str) -> str:
    return text.lower().strip().replace("_", " ").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")


def _get(obj: Any, attr: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _column_exists(view_info, column_name: str) -> bool:
    """Validación léxica rápida de existencia de columna en una vista."""
    if not view_info:
        return False

    available = set()
    for key in view_info.metricas.keys():
        available.add(_normalize(key))
    for col in view_info.columnas_fecha:
        available.add(_normalize(col))
    available.add(_normalize(view_info.nombre))

    requested = _normalize(column_name)
    for avail in available:
        if requested in avail or avail in requested:
            return True

    semantic_map = {
        "producto": ["producto", "descripcion", "descripción", "nombre_producto", "articulo", "artículo"],
        "sucursal": ["sucursal", "nombre_sede", "sede", "local", "tienda", "plaza"],
        "categoria": ["categoria", "categoría", "categoria_nueva"],
        "fecha": ["fecha", "fecha_completa", "fecha_venta", "mes"],
        "venta_total": ["venta_total", "ventas", "ventas_totales", "subtotal_diario", "ingreso"],
        "unidades": ["unidades", "cantidad", "unidades_totales", "unidades_vendidas"],
        "transacciones": ["transacciones", "total_transacciones", "numero_transacciones"],
        "ticket_promedio": ["ticket_promedio"],
    }

    variants = semantic_map.get(requested, [requested])
    for variant in variants:
        v = _normalize(variant)
        for avail in available:
            if v in avail or avail in v:
                return True

    return False


def _build_view_catalog(allowed_views: List[str]) -> Dict[str, Any]:
    catalog = {}
    for view_full_name in allowed_views:
        view_name = view_full_name.replace("semantic.", "").strip()
        view_info = _biz_mem.get_view(view_name)
        if not view_info:
            continue
        catalog[view_name] = {
            "tipo": view_info.tipo,
            "descripcion": view_info.descripcion,
            "granularidad": view_info.granularidad,
            "filtro_fecha": view_info.filtro_fecha,
            "metricas": list(view_info.metricas.keys()),
            "columnas_fecha": view_info.columnas_fecha,
            "notas": view_info.notas,
        }
    return catalog


def _validate_task_integrity(task, allowed_views: List[str]) -> List[str]:
    """Validación léxica final: la vista asignada existe y tiene las columnas."""
    errors = []
    preferred = task.preferred_view
    if not preferred:
        errors.append("La tarea no tiene preferred_view asignada.")
        return errors

    if preferred not in allowed_views:
        errors.append(f"La vista '{preferred}' no está en allowed_views.")
        return errors

    view_name = preferred.replace("semantic.", "").strip()
    view_info = _biz_mem.get_view(view_name)
    if not view_info:
        errors.append(f"La vista '{preferred}' no está documentada en AGENTS.md.")
        return errors

    for metric in (getattr(task, "metrics", []) or []):
        if not _column_exists(view_info, metric):
            errors.append(
                f"La vista '{view_name}' no contiene la métrica/columna '{metric}'. "
                f"Disponibles: {list(view_info.metricas.keys()) + view_info.columnas_fecha}"
            )

    for dim in (getattr(task, "dimensions", []) or []):
        if not _column_exists(view_info, dim):
            errors.append(
                f"La vista '{view_name}' no contiene la dimensión/columna '{dim}'. "
                f"Disponibles: {list(view_info.metricas.keys()) + view_info.columnas_fecha}"
            )

    return errors


def _find_compatible_view(task, allowed_views: List[str]) -> Optional[str]:
    """Fallback léxico: busca una vista en allowed_views que tenga todas las columnas requeridas."""
    required_cols = set()
    for metric in (getattr(task, "metrics", []) or []):
        required_cols.add(_normalize(metric))
    for dim in (getattr(task, "dimensions", []) or []):
        required_cols.add(_normalize(dim))

    if not required_cols:
        return None

    candidate_views = getattr(task, "candidate_views", []) or allowed_views
    for view_full_name in candidate_views:
        if view_full_name not in allowed_views:
            continue
        view_name = view_full_name.replace("semantic.", "").strip()
        view_info = _biz_mem.get_view(view_name)
        if not view_info:
            continue

        all_available = set()
        for key in view_info.metricas.keys():
            all_available.add(_normalize(key))
        for col in view_info.columnas_fecha:
            all_available.add(_normalize(col))

        missing = [
            col for col in required_cols
            if not any(col in avail or avail in col for avail in all_available)
        ]

        if not missing:
            return view_full_name

    return None


def _ensure_semantic_prefix(view_name: Optional[str]) -> Optional[str]:
    if not view_name:
        return None
    return view_name if view_name.startswith("semantic.") else f"semantic.{view_name}"


# ------------------------------------------------------------------
# Selección de vista unificada (Retriever + Fallback léxico)
# ------------------------------------------------------------------
def _select_view_for_task(
    task: SQLPayload,
    allowed_views: List[str],
    original_query: str = ""
) -> tuple[Optional[str], List[str], List[str]]:
    """
    Selecciona la vista para una subtarea usando semantic_retriever,
    con fallback léxico si Chroma falla.

    Retorna: (preferred_view, candidate_views, errors)
    """
    errors: List[str] = []

    hints = payload_to_column_hints(task, original_query=original_query or task.task)

    original_candidates = list(getattr(task, "candidate_views", []) or [])
    original_candidates = [
        _ensure_semantic_prefix(cv)
        for cv in original_candidates
        if cv and _ensure_semantic_prefix(cv) in allowed_views
    ]

    preferred: Optional[str] = None
    retriever_candidates: List[str] = []

    try:
        detailed = obtener_candidatas_detalles(
            query=task.task,
            k=5,
            allowed_views=allowed_views,
            column_hints=hints,
        )

        retriever_candidates = [
            _ensure_semantic_prefix(c["view_name"])
            for c in detailed
            if _ensure_semantic_prefix(c["view_name"]) in allowed_views
        ]

        for c in detailed:
            if c.get("can_answer"):
                preferred = _ensure_semantic_prefix(c["view_name"])
                break

        if not preferred and detailed:
            preferred = _ensure_semantic_prefix(detailed[0]["view_name"])
            errors.append(
                f"Tarea {task.task_id}: ninguna vista es completamente compatible; "
                f"se usará la mejor disponible: {preferred}"
            )

    except Exception as e:
        logger.warning(f"[Planner] semantic_retriever falló para tarea {task.task_id}: {e}")
        preferred = _find_compatible_view(task, allowed_views)
        if not preferred and original_candidates:
            preferred = original_candidates[0]
        if not preferred:
            errors.append(f"Tarea {task.task_id}: no se pudo seleccionar vista (retriever y fallback léxico fallaron).")

    if not preferred:
        preferred = _find_compatible_view(task, allowed_views)

    merged_candidates: List[str] = []
    seen: Set[str] = set()
    for cv in retriever_candidates + original_candidates:
        if cv and cv not in seen and cv in allowed_views:
            merged_candidates.append(cv)
            seen.add(cv)
    if preferred and preferred not in seen:
        merged_candidates.insert(0, preferred)

    if preferred:
        task.preferred_view = preferred
        task.candidate_views = merged_candidates
        lexical_errors = _validate_task_integrity(task, allowed_views)
        if lexical_errors:
            fallback = _find_compatible_view(task, allowed_views)
            if fallback:
                logger.info(f"[Planner] Fallback léxico a {fallback}")
                preferred = fallback
                task.preferred_view = preferred
                if preferred not in merged_candidates:
                    merged_candidates.insert(0, preferred)
                task.candidate_views = merged_candidates
            else:
                errors.extend([f"Tarea {task.task_id}: {err}" for err in lexical_errors])

    return preferred, merged_candidates, errors


# ------------------------------------------------------------------
# Nodo principal
# ------------------------------------------------------------------
def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    harness = state.get("harness_context", {})
    allowed_views = harness.get("allowed_views", [])
    ambiguity_notes = harness.get("ambiguity_notes", [])
    question = state["question"]
    messages = state.get("messages", [])

    # NUEVO: leer instrucción del supervisor
    instruction = state.get("next_agent_instruction") or state.get("supervisor_instruction")
    is_replan = _is_replan_instruction(instruction)

    # NUEVO: enriquecer la pregunta con contexto conversacional
    contextual_question = _build_conversational_context(messages, question)

    logger.info(f"[Planner] Pregunta contextual: {contextual_question[:200]}...")
    logger.info(f"[Planner] allowed_views={allowed_views}")
    logger.info(f"[Planner] ambiguity_notes={ambiguity_notes}")
    logger.info(f"[Planner] preferred_view={harness.get('preferred_view')}")
    if instruction:
        logger.info(f"[Planner] Instrucción del supervisor: {instruction[:200]}...")

    if not allowed_views:
        logger.error("[Planner] ERROR CRÍTICO: allowed_views está vacío.")
        plan = PlannerContract(
            intent="unknown",
            goal="",
            question_type="unknown",
            metrics=[],
            dimensions=[],
            filters="",
            tasks=[],
            confidence=0.0,
            needs_followup=True,
            followup_reason="No hay vistas de datos disponibles para responder la pregunta. Revisa AGENTS.md y el catálogo semántico."
        )
        return {
            "plan": plan,
            "last_agent": "planner",
            "messages": [AIMessage(content="[Planner] Error: no hay vistas de datos disponibles.")],
            "next_agent_instruction": None,
        }

    # ================================================================
    # PRIORIDAD MÁXIMA: Detección de demand forecast
    # ================================================================
    if _is_demand_forecast_question(contextual_question):
        logger.info("[Planner] Detectada pregunta de demand forecast")
        params = _extract_forecast_params(contextual_question)

        if not params.get("producto") or not params.get("sede"):
            params = _fallback_extract_forecast_params(contextual_question)

        if not params.get("producto") or not params.get("sede"):
            return {
                "plan": PlannerContract(
                    intent="demand_forecast",
                    goal="",
                    question_type="demand_forecast",
                    metrics=[],
                    dimensions=[],
                    filters="",
                    tasks=[],
                    confidence=0.5,
                    visualization_candidate=False,
                    needs_followup=True,
                    followup_reason="Necesito que especifiques el producto y la sede para generar el pronóstico de demanda."
                ),
                "forecast_request": None,
                "last_agent": "planner",
                "messages": [
                    AIMessage(content="[Planner] Pregunta de predicción de demanda detectada. Necesito producto y sede.")
                ],
                "next_agent_instruction": None,
            }

        n_dias = int(params.get("n_dias", 7))

        plan = PlannerContract(
            intent="demand_forecast",
            goal=f"Predecir la demanda diaria de {params['producto']} en {params['sede']} para los próximos {n_dias} días",
            question_type="demand_forecast",
            metrics=["prediccion", "prediccion_con_buffer", "safety_stock"],
            dimensions=["fecha", "producto", "sede"],
            filters=f"producto={params['producto']}, sede={params['sede']}",
            time_window=f"next_{n_dias}_days",
            tasks=[],
            confidence=0.9,
            visualization_candidate=False,
            needs_followup=False
        )

        return {
            "plan": plan,
            "forecast_request": {
                "producto": params["producto"],
                "sede": params["sede"],
                "n_dias": n_dias,
                "fecha_inicio": params.get("fecha_inicio"),
            },
            "last_agent": "planner",
            "messages": [
                AIMessage(
                    content=f"[Planner] Demand forecast: {params['producto']} @ {params['sede']} | {n_dias} días"
                )
            ],
            "next_agent_instruction": None,
        }

    # ================================================================
    # Corte temprano si hay ambigüedad
    # ================================================================
    preferred_view = harness.get("preferred_view")
    if ambiguity_notes and not preferred_view:
        plan = PlannerContract(
            intent="unknown",
            goal="",
            question_type="unknown",
            metrics=[],
            dimensions=[],
            filters="",
            tasks=[],
            confidence=0.0,
            needs_followup=True,
            followup_reason=f"Ambigüedad en selección de vista: {'; '.join(ambiguity_notes)}"
        )
        return {
            "plan": plan,
            "last_agent": "planner",
            "messages": [AIMessage(content=f"[Planner] Ambigüedad detectada: {plan.followup_reason}")],
            "next_agent_instruction": None,
        }

    # ================================================================
    # Planificación normal de consultas SQL
    # ================================================================
    view_catalog = _build_view_catalog(allowed_views)

    # NUEVO: construir contexto de replanificación si aplica
    replan_context = ""
    if is_replan:
        replan_context = _build_replan_context(state)
        logger.info(f"[Planner] Contexto de replanificación:\n{replan_context[:500]}...")

    system_prompt = f"""
Eres un Planner BI avanzado. Transformas preguntas de negocio en planes operacionales estructurados.

CATÁLOGO DE VISTAS PERMITIDAS (con columnas disponibles):
{json.dumps(view_catalog, indent=2, ensure_ascii=False)}

{"=" * 60}
INSTRUCCIÓN DEL SUPERVISOR:
{instruction or "Ninguna instrucción adicional. Planifica la pregunta del usuario de forma directa."}
{"=" * 60 if is_replan else ""}
{replan_context if is_replan else ""}
{"=" * 60 if is_replan else ""}

REGLAS CRÍTICAS DE SELECCIÓN DE VISTA:
1. ANTES de asignar una vista a una tarea, verifica en el catálogo de arriba que esa vista contenga EXPLÍCITAMENTE las columnas que la tarea requiere.
2. NUNCA asignes una vista si la columna requerida no aparece en su lista de métricas/columnas.
3. Si la pregunta pide desglose por PRODUCTO, elige ÚNICAMENTE vistas que tengan 'producto', 'descripcion' o similar.
4. Si la pregunta pide desglose por SEDE/LOCAL, elige vistas que tengan 'sucursal', 'nombre_sede' o similar.
5. Si la pregunta pide desglose por CATEGORÍA, elige vistas que tengan 'categoria'.
6. SEGURIDAD: Usa ÚNICAMENTE vistas presentes en el catálogo de arriba. NO inventes vistas.
7. DESCOMPOSICIÓN: Si la pregunta tiene múltiples KPIs de distinta naturaleza o contextos temporales distintos, genera tareas SEPARADAS.
8. NO DESCOMPONER si es la misma intención, misma granularidad y misma temporalidad.
9. Si la pregunta del usuario no requiere de generar consulta no digas que ha fallado el plan.
10. Si el usuario hace preguntas como 'Hola', 'como estás?' o 'quien eres?' responde cordialmente, no se necesitan consultas para estas preguntas.
11. Si el supervisor solicita REPLANIFICACIÓN (instrucción de replan arriba), genera un plan ALTERNO: cambia de vista, simplifica la pregunta, o divide en subtareas más pequeñas. NO repitas el plan anterior.

REGLAS DE NEGOCIO:
- "Se han vendido" / "ventas" / "unidades vendidas" → vistas de VENTAS NORMALES.
- "Canjes", "fidelización", "puntos" → vistas de FIDELIZACIÓN.
- "Cortesías", "gratis", "regalos" → vistas de CORTESÍA.

OUTPUT: JSON con schema PlannerContract.
- tasks: lista de SQLPayload con task_id, task, execution_strategy, metrics, dimensions, candidate_views, preferred_view.
- needs_followup: true si hay ambigüedad insalvable.

REGLAS ADICIONALES:
- "informe completo", "reporte detallado", "análisis profundo", "deep dive" → question_type="deep_research".
"""

    # NUEVO: pasar contextual_question al LLM
    human = HumanMessage(content=f"Pregunta del usuario: {contextual_question}")
    planner_llm = LLM.with_structured_output(PlannerContract, method="function_calling")
    plan_raw = planner_llm.invoke([SystemMessage(content=system_prompt), human])

    if isinstance(plan_raw, dict):
        plan = PlannerContract(**plan_raw)
    else:
        plan = plan_raw

    # ============================================================
    # VALIDACIÓN Y ENRIQUECIMIENTO SEMÁNTICO POST-LLM
    # ============================================================
    validation_errors: List[str] = []

    if plan.question_type == "demand_forecast":
        plan.tasks = []
        plan.visualization_candidate = False

    for task in plan.tasks:
        preferred, candidates, errs = _select_view_for_task(task, allowed_views, original_query=question)

        if errs:
            validation_errors.extend(errs)

        if preferred:
            task.preferred_view = preferred
            task.candidate_views = candidates
        else:
            validation_errors.append(f"Tarea {task.task_id}: no se pudo asignar ninguna vista.")

    # ============================================================
    # FALLBACK: Si no se generaron tareas, crear una tarea genérica
    # ============================================================
    if not plan.tasks and plan.question_type not in ("demand_forecast", "unknown"):
        logger.warning(f"[Planner] LLM no generó tareas. Creando tarea fallback.")
        if allowed_views:
            hints = {"original_query": contextual_question}
            try:
                default_vista = seleccionar_vista_principal(query=contextual_question, column_hints=hints, allowed_views=allowed_views)
                default_view = default_vista["view_name"] if default_vista else allowed_views[0]
            except Exception as e:
                logger.warning(f"[Planner] Retriever falló en fallback general: {e}")
                default_view = allowed_views[0]

            default_view = _ensure_semantic_prefix(default_view)

            plan.tasks = [
                SQLPayload(
                    task_id="1",
                    task=f"Responder a la pregunta: {contextual_question}",
                    metrics=[],
                    dimensions=[],
                    filters_description="",
                    execution_strategy="single_view",
                    candidate_views=list(allowed_views),
                    preferred_view=default_view
                )
            ]
            plan.needs_followup = False
            logger.info(f"[Planner] Tarea fallback creada con vista: {default_view}")
        else:
            plan.needs_followup = True
            plan.followup_reason = "No hay vistas disponibles para responder la pregunta."

    if validation_errors:
        plan.needs_followup = True
        plan.followup_reason = " | ".join(validation_errors)
        logger.warning(f"[Planner] Errores de validación: {validation_errors}")

    task_summary = " | ".join([f"{t.task_id}:{t.task}" for t in plan.tasks])
    prefs = [t.preferred_view for t in plan.tasks]

    return {
        "plan": plan,
        "last_agent": "planner",
        "messages": [
            AIMessage(
                content=f"[Planner] Intención: {plan.intent} | Tasks: {len(plan.tasks)} ({task_summary}) "
                        f"| Confianza: {plan.confidence} | Preferred: {prefs} | Followup: {plan.needs_followup}"
            )
        ],
        "next_agent_instruction": None,  # limpiar instrucción ya consumida
    }
