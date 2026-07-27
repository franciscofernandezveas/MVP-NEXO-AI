from typing import Any, Dict, List, Optional
import json
import logging
import re
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from core.llm import LLM
from core.contracts import PlannerContract, SQLPayload, FilterClause
from core.harness import BusinessMemory
from core.database import get_semantic_schema_for_views, get_semantic_schema_info

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


def _normalize(text: str) -> str:
    return text.lower().strip().replace("_", " ").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")


def _column_exists(view_info, column_name: str) -> bool:
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


def _validate_task_integrity(task, catalog: Dict[str, Any]) -> List[str]:
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

    required_metrics = getattr(task, "metrics", []) or []
    for metric in required_metrics:
        if not _column_exists(view_info, metric):
            errors.append(
                f"La vista '{view_name}' no contiene la métrica/columna '{metric}'. "
                f"Columnas disponibles: {list(view_info.metricas.keys()) + view_info.columnas_fecha}"
            )

    required_dimensions = getattr(task, "dimensions", []) or []
    for dim in required_dimensions:
        if not _column_exists(view_info, dim):
            errors.append(
                f"La vista '{view_name}' no contiene la dimensión/columna '{dim}'. "
                f"Columnas disponibles: {list(view_info.metricas.keys()) + view_info.columnas_fecha}"
            )

    return errors


def _find_compatible_view(task, catalog: Dict[str, Any], allowed_views: List[str]) -> Optional[str]:
    required_cols = set()
    for metric in (getattr(task, "metrics", []) or []):
        required_cols.add(_normalize(metric))
    for dim in (getattr(task, "dimensions", []) or []):
        required_cols.add(_normalize(dim))

    if not required_cols:
        return None

    candidate_views = getattr(task, "candidate_views", []) or allowed_views
    for view_full_name in candidate_views:
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


# ------------------------------------------------------------------
# NUEVO: Fuerza preferred_view del harness en las tareas
# ------------------------------------------------------------------
def _ensure_preferred_view_from_harness(
    contract: PlannerContract,
    preferred_view: Optional[str]
) -> PlannerContract:
    """
    Si el harness indica una vista preferida, fuerza que todas las tareas
    la consideren como preferred_view (si aplica) y que esté en candidate_views.
    """
    if not preferred_view:
        return contract

    clean_pref = (
        preferred_view
        if preferred_view.startswith("semantic.")
        else f"semantic.{preferred_view}"
    )

    for task in contract.tasks:
        # Asegurar que candidate_views incluya la vista preferida del harness
        if clean_pref not in (task.candidate_views or []):
            task.candidate_views = list(task.candidate_views or []) + [clean_pref]

        # Si la tarea no tiene preferred_view, usar la del harness
        if not task.preferred_view:
            task.preferred_view = clean_pref
            logger.info(f"[Planner] Asignando preferred_view del harness: {clean_pref}")
            continue

        # Si la tarea ya tiene otra preferred_view, priorizar la del harness
        current = task.preferred_view
        if current != clean_pref:
            logger.info(
                f"[Planner] preferred_view existente '{current}' será reemplazada por "
                f"la del harness '{clean_pref}'"
            )
            task.preferred_view = clean_pref

    return contract


def _build_planner_system_prompt(
    view_catalog: Dict[str, Any],
    business_memory: Any,
    preferred_view: Optional[str],
    schema_info: str
) -> str:
    view_catalog_str = json.dumps(view_catalog, indent=2, ensure_ascii=False)

    business_memory_str = (
        json.dumps(business_memory, indent=2, ensure_ascii=False)
        if business_memory else "No hay reglas de negocio adicionales."
    )

    preferred_view_instruction = ""
    if preferred_view:
        preferred_view_instruction = f"""
=== VISTA PREFERIDA DEL HARNESS ===
{preferred_view}

REGLA IMPORTANTE: Si la vista preferida de arriba contiene TODAS las columnas requeridas por la pregunta (métricas, dimensiones y filtros), entonces preferred_view DEBE ser {preferred_view} y candidate_views DEBE incluir {preferred_view} como primera opción.
"""

    example_output = json.dumps({
        "intent": "consultar ventas por sede del mes pasado",
        "goal": "Obtener el total de ventas desagregado por sede para abril",
        "question_type": "single_kpi",
        "metrics": ["venta_total"],
        "dimensions": ["nombre_sede"],
        "filters": [
            {"column": "nombre_sede", "operator": "ILIKE", "value": "merced", "reasoning": "Filtro por sede específica usando ILIKE"}
        ],
        "filters_description": "Sede 'merced' con ILIKE",
        "date_range": {
            "start": "2024-04-01",
            "end": "2024-04-30",
            "grain": "month",
            "relative_label": "last_month"
        },
        "time_window": "mes pasado",
        "assumptions": ["Se asume ventas normales porque el usuario no especificó canjes ni cortesías"],
        "missing_information": [],
        "tasks": [
            {
                "task_id": "t1",
                "task": "Calcular total de ventas por sede para abril 2024",
                "metrics": ["venta_total"],
                "dimensions": ["nombre_sede"],
                "filters": [{"column": "nombre_sede", "operator": "ILIKE", "value": "merced"}],
                "date_range": {"start": "2024-04-01", "end": "2024-04-30", "grain": "month", "relative_label": "last_month"},
                "execution_strategy": "by_branch",
                "candidate_views": ["semantic.vw_ventas_sede"],
                "preferred_view": "semantic.vw_ventas_sede",
                "assumptions": []
            }
        ],
        "confidence": 0.95,
        "visualization_candidate": True,
        "chart_type_hint": "bar",
        "needs_followup": False,
        "followup_reason": None,
        "followup_question": None
    }, indent=2, ensure_ascii=False)

    prompt_template = """
Eres un Planner BI avanzado. Tu trabajo es transformar una pregunta de negocio en un plan operacional JSON válido según el contrato PlannerContract.

=== CATÁLOGO DE VISTAS PERMITIDAS ===
{view_catalog_str}

=== SCHEMA TÉCNICO REAL (DDL de PostgreSQL) ===
{schema_info}

=== MEMORIA DE REGLAS DE NEGOCIO ===
{business_memory_str}
{preferred_view_instruction}

=== REGLAS DE MAPEO SEMÁNTICO (muy importantes) ===
- Las métricas y dimensiones que detectes del usuario son CONCEPTUALES.
- Debes encontrar la columna FÍSICA equivalente en la vista elegida usando el SCHEMA TÉCNICO.
- Ejemplos de mapeos comunes (no exhaustivos):
  * "ventas" puede mapear a: ventas, venta_total, total_ventas, monto, ingreso...
  * "unidades" puede mapear a: unidades, unidades_totales, cantidad...
  * "sede" puede mapear a: sucursal, nombre_sede, local, tienda...
  * "producto" puede mapear a: producto, nombre_producto, descripcion...
  * "fecha" puede mapear a: fecha, fecha_completa, fecha_venta...
- NO exijas nombres exactos entre vistas. Cada vista puede llamar distinto a una misma métrica.
- Si el usuario pide "ventas por sede" y la vista elegida tiene `venta_total` + `nombre_sede`, eso es válido.

=== REGLA DE SELECCIÓN DE VISTA ===
1. Si `preferred_view` del contexto tiene todas las métricas/dimensiones necesarias, úsala como `preferred_view`.
2. Si no, elige la vista más específica de `allowed_views` que resuelva todo.
3. La vista debe tener una columna física para CADA métrica y dimensión solicitada.
4. Si ninguna vista lo tiene todo, marca `needs_followup=true` con una pregunta clara.

=== DEFINICIONES ===
- "ventas_normales": ventas, facturación, ticket, unidades vendidas, revenue, ingresos.
- "fidelización": canjes, puntos, recompensas, loyalty.
- "cortesía": gratis, regalos, cortesías.
- "demand_forecast": predicción, pronóstico, cuánto se venderá, demanda futura.
- "deep_research": informe completo, reporte detallado, análisis profundo, deep dive.

=== ALGORITMO OBLIGATORIO ===
1. Normaliza la pregunta: extrae métrica, dimensiones, filtros de texto y período.
2. Resuelve fechas relativas a un DateRange concreto:
   - "hoy" / "ayer" / "esta semana" / "semana pasada" / "mes pasado" / "año pasado" / "últimos N días".
   - Si no hay período, asume last_30_days.
3. Selecciona candidate_views: una vista solo es candidata si contiene EXACTAMENTE todas las columnas de metrics, dimensions y filters.
4. Elige preferred_view: la primera de candidate_views que cubra todo. preferred_view DEBE estar en candidate_views.
   {preferred_view_rule}
5. Determina question_type: single_kpi | multi_kpi | comparison | trend | lookup | deep_research | demand_forecast | unknown.
6. Descompón en múltiples SQLPayload solo si la pregunta mezcla:
   - KPIs de distinta naturaleza (ventas + canjes),
   - granularidades incompatibles (por sede y por producto sin que una vista lo cubra),
   - ventanas temporales distintas (mes pasado vs año pasado).
   En otro caso, genera UNA sola tarea.
7. Si ninguna vista cubre las columnas requeridas, devuelve:
   - needs_followup = true
   - followup_reason = "missing_columns"
   - followup_question = pregunta clara al usuario pidiendo aclaración o reformulación.
8. Revisa tu output: verifica que cada columna en metrics/dimensions/filters exista en preferred_view.

=== REGLAS DE SELECCIÓN ===
- Desglose por producto: requiere columna 'producto', 'descripcion' o 'sku'.
- Desglose por sede/local: requiere 'sucursal', 'nombre_sede' o 'local'.
- Desglose por categoría: requiere 'categoria'.
- Filtros de texto siempre con operador ILIKE (nunca = directo).
- Seguridad: solo vistas del catálogo. Nunca inventes vistas ni columnas.

=== FORMATO DE SALIDA (PlannerContract) ===
Devuelve ÚNICAMENTE JSON válido con esta estructura exacta:

{example_output}
"""
    return prompt_template.format(
        view_catalog_str=view_catalog_str,
        schema_info=schema_info,
        business_memory_str=business_memory_str,
        preferred_view_instruction=preferred_view_instruction,
        preferred_view_rule=f"IMPORTANTE: Si el harness indica preferred_view='{preferred_view}' y esa vista cubre todo, ÚSALA como preferred_view." if preferred_view else "Elige preferred_view según el algoritmo.",
        example_output=example_output
    )


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

    # ================================================================
    # Catálogo disponible para todo el grafo
    # ================================================================
    view_catalog = _build_view_catalog(allowed_views)

    if not allowed_views:
        logger.error("[Planner] ERROR CRÍTICO: allowed_views está vacío. El harness no cargó vistas.")
        plan = PlannerContract(
            intent="unknown",
            goal="",
            question_type="unknown",
            metrics=[],
            dimensions=[],
            filters=[],
            tasks=[],
            confidence=0.0,
            needs_followup=True,
            followup_reason="No hay vistas de datos disponibles para responder la pregunta. Revisa AGENTS.md y el catálogo semántico."
        )
        return {
            "plan": plan,
            "view_catalog": view_catalog,
            "last_agent": "planner",
            "messages": state.get("messages", []) + [
                AIMessage(content="[Planner] Error: no hay vistas de datos disponibles.")
            ]
        }

    # ================================================================
    # Carga schema técnico real
    # ================================================================
    schema_info = state.get("schema_info", "")
    if not schema_info.strip():
        try:
            schema_info = get_semantic_schema_for_views(allowed_views)
            if not schema_info.strip():
                schema_info = get_semantic_schema_info(max_objects=30)
        except Exception as e:
            logger.warning(f"[Planner] Error cargando schema: {e}")
            try:
                schema_info = get_semantic_schema_info(max_objects=30)
            except Exception as e2:
                logger.warning(f"[Planner] Error schema completo: {e2}")
                schema_info = "-- Schema no disponible"

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
                    filters=[],
                    tasks=[],
                    confidence=0.5,
                    visualization_candidate=False,
                    needs_followup=True,
                    followup_reason="Necesito que especifiques el producto y la sede para generar el pronóstico de demanda."
                ),
                "forecast_request": None,
                "view_catalog": view_catalog,
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
            filters=[],
            filters_description=f"producto={params['producto']}, sede={params['sede']}",
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
            "view_catalog": view_catalog,
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
            filters=[],
            tasks=[],
            confidence=0.0,
            needs_followup=True,
            followup_reason=f"Ambigüedad en selección de vista: {'; '.join(ambiguity_notes)}"
        )
        return {
            "plan": plan,
            "view_catalog": view_catalog,
            "last_agent": "planner",
            "messages": state.get("messages", []) + [
                AIMessage(content=f"[Planner] Ambigüedad detectada: {plan.followup_reason}")
            ]
        }

    # ================================================================
    # Planificación normal de consultas SQL
    # ================================================================
    business_memory = harness.get("business_memory", {})

    system_prompt = _build_planner_system_prompt(
        view_catalog,
        business_memory,
        harness.get("preferred_view"),
        schema_info
    )

    system = SystemMessage(content=system_prompt)
    human = HumanMessage(content=f"Pregunta del usuario: {question}")

    planner_llm = LLM.with_structured_output(PlannerContract, method="function_calling")
    plan_raw = planner_llm.invoke([system, human])

    if isinstance(plan_raw, dict):
        plan = PlannerContract(**plan_raw)
    else:
        plan = plan_raw

    # ============================================================
    # NUEVO: Forzar preferred_view del harness
    # ============================================================
    plan = _ensure_preferred_view_from_harness(plan, harness.get("preferred_view"))

    # ============================================================
    # VALIDACIÓN DE INTEGRIDAD POST-LLM
    # ============================================================
    validation_errors: List[str] = []

    if plan.question_type == "demand_forecast":
        plan.tasks = []
        plan.visualization_candidate = False

    for task in plan.tasks:
        if task.preferred_view and not task.preferred_view.startswith("semantic."):
            task.preferred_view = f"semantic.{task.preferred_view}"

        for i, cv in enumerate(task.candidate_views or []):
            if not cv.startswith("semantic."):
                task.candidate_views[i] = f"semantic.{cv}"

        if task.preferred_view and task.preferred_view not in allowed_views:
            fallback = _find_compatible_view(task, view_catalog, allowed_views)
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

        task_errors = _validate_task_integrity(task, view_catalog)
        if task_errors:
            fallback = _find_compatible_view(task, view_catalog, allowed_views)
            if fallback:
                logger.info(f"[Planner] Fallback por columnas incompatibles a {fallback}")
                task.preferred_view = fallback
                if fallback not in (task.candidate_views or []):
                    task.candidate_views = list(task.candidate_views or []) + [fallback]
            else:
                validation_errors.extend([f"Tarea {task.task_id}: {err}" for err in task_errors])

    # ============================================================
    # FALLBACK: Si no se generaron tareas, crear una tarea genérica
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
                    filters=[],
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
        "view_catalog": view_catalog,
        "last_agent": "planner",
        "messages": state.get("messages", []) + [
            AIMessage(
                content=f"[Planner] Intención: {plan.intent} | Tasks: {len(plan.tasks)} ({task_summary}) "
                        f"| Confianza: {plan.confidence} | Preferred: {prefs} | Followup: {plan.needs_followup}"
            )
        ]
    }
