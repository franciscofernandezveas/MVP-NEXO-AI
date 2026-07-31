from typing import Any, Dict, List, Optional, Set
import json
import logging
import re
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from core.llm import LLM
from core.contracts import PlannerContract, SQLPayload, FilterSpec
from core.harness import BusinessMemory
from core.semantic_retriever import (
    column_exists_in_view,
    resolve_column,
    find_compatible_view,
    get_view_columns,
)

logger = logging.getLogger(__name__)

_biz_mem = BusinessMemory.from_file()


# ------------------------------------------------------------------
# NUEVO: Detección de demand forecast
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
    sede_detectada = None
    for sede in sedes:
        if sede in q:
            sede_detectada = sede.title()
            break

    productos = [
        "americano", "capuccino", "latte", "espresso", "mokaccino",
        "cortado", "flat white", "iced latte", "chai latte", "chocolate caliente"
    ]
    producto_detectado = None
    for prod in productos:
        if prod in q:
            producto_detectado = prod
            break

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


def _build_view_catalog(allowed_views: List[str]) -> Dict[str, Any]:
    """
    Catálogo de vistas permitidas para inyectar al prompt del planner.
    Las keys usan el nombre completo con prefijo semantic.
    """
    catalog = {}
    for view_full_name in allowed_views:
        view_name = view_full_name.replace("semantic.", "").strip()
        view_info = _biz_mem.get_view(view_name)
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


def _parse_filters_description(description: str) -> List[FilterSpec]:
    """
    Fallback: convierte descripción textual de filtros en filtros estructurados.
    """
    if not description:
        return []

    text_columns = {
        "nombre_sede", "sede", "sucursal", "local", "tienda", "plaza",
        "producto", "descripcion", "descripción", "categoria", "categoría"
    }
    filters = []

    for part in re.split(r",\s*(?=\w+\s*[=<>])", description):
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

        filters.append(FilterSpec(column=col, operator=op, value=val, value_type=vt))

    return filters


def _validate_task_integrity(task: SQLPayload) -> List[str]:
    """
    Valida que la vista preferida soporte métricas, dimensiones y filtros del payload.
    """
    errors = []
    preferred = task.preferred_view
    if not preferred:
        errors.append("La tarea no tiene preferred_view asignada.")
        return errors

    view_name = preferred.replace("semantic.", "").strip()
    view_info = _biz_mem.get_view(view_name)
    if not view_info:
        errors.append(f"La vista '{preferred}' no está documentada en AGENTS.md.")
        return errors

    for metric in (task.metrics or []):
        if not column_exists_in_view(preferred, metric):
            real = resolve_column(preferred, metric)
            hint = f" ¿Quizás quisiste '{real}'?" if real else ""
            errors.append(
                f"La vista '{view_name}' no contiene la métrica '{metric}'.{hint} "
                f"Columnas disponibles: {get_view_columns(preferred)}"
            )

    for dim in (task.dimensions or []):
        if not column_exists_in_view(preferred, dim):
            real = resolve_column(preferred, dim)
            hint = f" ¿Quizás quisiste '{real}'?" if real else ""
            errors.append(
                f"La vista '{view_name}' no contiene la dimensión '{dim}'.{hint} "
                f"Columnas disponibles: {get_view_columns(preferred)}"
            )

    for f in (task.filters or []):
        if not column_exists_in_view(preferred, f.column):
            real = resolve_column(preferred, f.column)
            hint = f" ¿Quizás quisiste '{real}'?" if real else ""
            errors.append(
                f"La vista '{view_name}' no contiene la columna de filtro '{f.column}'.{hint}"
            )

    return errors


# ------------------------------------------------------------------
# Nodo principal
# ------------------------------------------------------------------
def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    harness = state.get("harness_context", {})
    allowed_views = harness.get("allowed_views", [])
    ambiguity_notes = harness.get("ambiguity_notes", [])
    question = state["question"]

    logger.info(f"[Planner] allowed_views={allowed_views}")
    logger.info(f"[Planner] ambiguity_notes={ambiguity_notes}")
    logger.info(f"[Planner] preferred_view={harness.get('preferred_view')}")

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
            "messages": state.get("messages", []) + [
                AIMessage(content="[Planner] Error: no hay vistas de datos disponibles.")
            ]
        }

    # ================================================================
    # PRIORIDAD MÁXIMA: Detección de demand forecast
    # ================================================================
    if _is_demand_forecast_question(question):
        logger.info("[Planner] Detectada pregunta de demand forecast")
        params = _extract_forecast_params(question)

        if not params.get("producto") or not params.get("sede"):
            params = _fallback_extract_forecast_params(question)

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
                "messages": state.get("messages", []) + [
                    AIMessage(content="[Planner] Pregunta de predicción de demanda detectada. Necesito producto y sede.")
                ]
            }

        n_dias = int(params.get("n_dias", 7))

        plan = PlannerContract(
            intent="demand_forecast",
            goal=f"Predecir la demanda diaria de {params['producto']} en {params['sede']} para los próximos {n_dias} días",
            question_type="demand_forecast",
            metrics=["prediccion", "prediccion_con_buffer", "safety_stock"],
            dimensions=["fecha", "producto", "sede"],
            filters_description=f"producto={params['producto']}, sede={params['sede']}",
            filters=[
                FilterSpec(column="producto", operator="ILIKE", value=params["producto"], value_type="string"),
                FilterSpec(column="sede", operator="ILIKE", value=params["sede"], value_type="string"),
            ],
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
            "messages": state.get("messages", []) + [
                AIMessage(
                    content=f"[Planner] Demand forecast: {params['producto']} @ {params['sede']} | {n_dias} días"
                )
            ]
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
            "messages": state.get("messages", []) + [
                AIMessage(content=f"[Planner] Ambigüedad detectada: {plan.followup_reason}")
            ]
        }

    # ================================================================
    # Planificación normal de consultas SQL
    # ================================================================
    view_catalog = _build_view_catalog(allowed_views)

    system = SystemMessage(content=f"""
Eres un Planner BI avanzado. Transformas preguntas de negocio en planes operacionales estructurados.

CATÁLOGO DE VISTAS PERMITIDAS (con columnas disponibles):
{json.dumps(view_catalog, indent=2, ensure_ascii=False)}

REGLAS CRÍTICAS DE SELECCIÓN DE VISTA:
1. ANTES de asignar una vista a una tarea, verifica en el catálogo de arriba que esa vista contenga EXPLÍCITAMENTE las columnas que la tarea requiere.
2. NUNCA asignes una vista si la columna requerida no aparece en su lista de métricas/columnas.
3. Si la pregunta pide desglose por PRODUCTO, elige ÚNICAMENTE vistas que tengan 'producto', 'descripcion' o similar.
4. Si la pregunta pide desglose por SEDE/LOCAL, elige vistas que tengan 'sucursal', 'nombre_sede' o similar.
5. Si la pregunta pide desglose por CATEGORÍA, elige vistas que tengan 'categoria'.
6. SEGURIDAD: Usa ÚNICAMENTE vistas presentes en el catálogo de arriba. NO inventes vistas.
7. DESCOMPOSICIÓN: Si la pregunta tiene múltiples KPIs de distinta naturaleza o contextos temporales distintos, genera tareas SEPARADAS.
8. NO DESCOMPONER si es la misma intención, misma granularidad y misma temporalidad.

REGLAS DE NEGOCIO:
- "Se han vendido" / "ventas" / "unidades vendidas" → vistas de VENTAS NORMALES.
- "Canjes", "fidelización", "puntos" → vistas de FIDELIZACIÓN.
- "Cortesías", "gratis", "regalos" → vistas de CORTESÍA.

REGLAS DE FILTROS:
- Genera SIEMPRE filtros estructurados en el campo `filters` de SQLPayload.
- Para columnas de texto (sede, producto, categoría, local) usa operator="ILIKE" y value_type="string".
- Ejemplo: {{"column": "nombre_sede", "operator": "ILIKE", "value": "merced", "value_type": "string"}}
- Para valores numéricos usa operator="=" y value_type="number".
- Para listas usa operator="IN" y value_type="list".

REGLAS DE TIME WINDOW:
- "últimos 7 días" → time_window="last_7_days"
- "este mes" → time_window="current_month"
- "mes pasado" → time_window="previous_month"
- "año actual" → time_window="current_year"

REGLAS DE ESTRATEGIA:
- Una sola métrica, sin desglose → execution_strategy="single_view"
- Desglose por sede → execution_strategy="by_branch"
- Desglose por producto → execution_strategy="by_product"
- Desglose mensual → execution_strategy="monthly"
- Comparación vs periodo anterior → execution_strategy="compare_periods"
- Serie histórica → execution_strategy="historical"

REGLAS DE DEMAND FORECAST:
- "Predice", "pronostica", "cuánto se venderá", "demanda futura" → question_type="demand_forecast".
- El planner detectará esto automáticamente y extraerá producto/sede/n_días.

REGLAS ADICIONALES:
- "informe completo", "reporte detallado", "análisis profundo", "deep dive" → question_type="deep_research".

OUTPUT: JSON con schema PlannerContract.
- tasks: lista de SQLPayload con task_id, task, execution_strategy, metrics, dimensions, filters, time_window, candidate_views, preferred_view.
- needs_followup: true si hay ambigüedad insalvable.
""")

    human = HumanMessage(content=f"Pregunta del usuario: {question}")
    planner_llm = LLM.with_structured_output(PlannerContract, method="function_calling")
    plan_raw = planner_llm.invoke([system, human])

    if isinstance(plan_raw, dict):
        plan = PlannerContract(**plan_raw)
    else:
        plan = plan_raw

    # ============================================================
    # VALIDACIÓN DE INTEGRIDAD POST-LLM
    # ============================================================
    validation_errors: List[str] = []

    if plan.question_type == "demand_forecast":
        plan.tasks = []
        plan.visualization_candidate = False

    for task in plan.tasks:
        # Normalizar prefijos
        if task.preferred_view and not task.preferred_view.startswith("semantic."):
            task.preferred_view = f"semantic.{task.preferred_view}"

        for i, cv in enumerate(task.candidate_views or []):
            if not cv.startswith("semantic."):
                task.candidate_views[i] = f"semantic.{cv}"

        # Fallback de filters_description a filters estructurados
        if not task.filters and task.filters_description:
            task.filters = _parse_filters_description(task.filters_description)

        # Si preferred_view no está en allowed_views, buscar compatible
        if task.preferred_view and task.preferred_view not in allowed_views:
            fallback = find_compatible_view(task, allowed_views)
            if fallback:
                logger.info(f"[Planner] Fallback de {task.preferred_view} a {fallback}")
                task.preferred_view = fallback
                if fallback not in (task.candidate_views or []):
                    task.candidate_views = list(task.candidate_views or []) + [fallback]
            else:
                validation_errors.append(
                    f"Tarea {task.task_id}: {task.preferred_view} no está en allowed_views."
                )
                continue

        # Validar que la vista preferida contenga métricas/dimensiones/filtros
        task_errors = _validate_task_integrity(task)
        if task_errors:
            fallback = find_compatible_view(task, allowed_views)
            if fallback:
                logger.info(f"[Planner] Fallback por columnas incompatibles a {fallback}")
                task.preferred_view = fallback
                if fallback not in (task.candidate_views or []):
                    task.candidate_views = list(task.candidate_views or []) + [fallback]
            else:
                validation_errors.extend([f"Tarea {task.task_id}: {err}" for err in task_errors])

    # ============================================================
    # FALLBACK: Si no se generaron tareas
    # ============================================================
    if not plan.tasks and plan.question_type not in ("demand_forecast", "unknown"):
        logger.warning(f"[Planner] LLM no generó tareas. Creando tarea fallback.")
        if allowed_views:
            default_view = allowed_views[0]
            plan.tasks = [
                SQLPayload(
                    task_id="1",
                    task=f"Responder a la pregunta: {question}",
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
        plan.confidence = min(plan.confidence, 0.5)
        logger.warning(f"[Planner] Errores de validación: {validation_errors}")

    task_summary = " | ".join([f"{t.task_id}:{t.task}" for t in plan.tasks])
    prefs = [t.preferred_view for t in plan.tasks]

    return {
        "plan": plan,
        "last_agent": "planner",
        "messages": state.get("messages", []) + [
            AIMessage(
                content=f"[Planner] Intención: {plan.intent} | Tasks: {len(plan.tasks)} ({task_summary}) "
                        f"| Confianza: {plan.confidence} | Preferred: {prefs} | Followup: {plan.needs_followup}"
            )
        ]
    }
