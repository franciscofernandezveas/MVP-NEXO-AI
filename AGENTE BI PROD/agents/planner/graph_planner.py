from typing import Any, Dict, List, Optional, Set
import json
import logging
import re
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from core.llm import LLM
from core.contracts import PlannerContract, SQLPayload
from core.harness import BusinessMemory

logger = logging.getLogger(__name__)

# Cargar catálogo una sola vez al importar el módulo
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
    """
    Extrae producto y sede de la pregunta usando LLM con structured output.
    Fallback a regex simple si el LLM falla.
    """
    parser = LLM.with_structured_output(_ForecastParamsInternal)

    try:
        result = parser.invoke(
            f"Extrae los parámetros para predecir demanda de la siguiente pregunta. "
            f"Si no hay fecha de inicio, devuelve null.\n\nPregunta: {question}"
        )
        # langchain-openai 0.1.8 puede devolver dict
        if isinstance(result, dict):
            result = _ForecastParamsInternal(**result)
        return result.model_dump()
    except Exception as e:
        logger.warning(f"[Planner] Falló extracción estructurada de forecast: {e}")
        return {}


def _fallback_extract_forecast_params(question: str) -> Dict[str, Any]:
    """
    Regex simple para extraer producto/sede si el LLM falla.
    """
    q = question.lower()

    # Sedes conocidas
    sedes = ["plaza bolsillo", "merced", "tajamar", "persa victor manuel"]
    sede_detectada = None
    for sede in sedes:
        if sede in q:
            sede_detectada = sede.title()
            break

    # Productos comunes
    productos = [
        "americano", "capuccino", "latte", "espresso", "mokaccino",
        "cortado", "flat white", "iced latte", "chai latte", "chocolate caliente"
    ]
    producto_detectado = None
    for prod in productos:
        if prod in q:
            producto_detectado = prod
            break

    # Días
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
        logger.error("[Planner] ERROR CRÍTICO: allowed_views está vacío. El harness no cargó vistas.")
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

        # Si el LLM no logró extraer, usar fallback
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
Eres un Planner BI avanzado. Tu trabajo es transformar una pregunta de negocio en un plan operacional JSON válido según el contrato PlannerContract.

=== CATÁLOGO DE VISTAS PERMITIDAS ===
{view_catalog}

=== MEMORIA DE REGLAS DE NEGOCIO ===
{business_memory}

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

{
  "intent": "intención detectada",
  "goal": "objetivo ejecutivo de la consulta",
  "question_type": "single_kpi|multi_kpi|comparison|trend|lookup|deep_research|demand_forecast|unknown",
  "metrics": ["nombre_exacto_columna_metrica"],
  "dimensions": ["nombre_exacto_columna_dimension"],
  "filters": [
    {"column": "nombre_exacto_columna", "operator": "ILIKE", "value": "valor", "reasoning": "..."}
  ],
  "filters_description": "Filtros en lenguaje natural para logs",
  "date_range": {
    "start": "YYYY-MM-DD",
    "end": "YYYY-MM-DD",
    "grain": "day|week|month|quarter|year",
    "relative_label": "last_30_days|..."
  },
  "time_window": "texto original del período detectado",
  "assumptions": ["asumimos X porque el usuario no especificó Y"],
  "missing_information": ["información que falta pero que no bloquea"],
  "tasks": [
    {
      "task_id": "t1",
      "task": "descripción clara y ejecutable",
      "metrics": ["..."],
      "dimensions": ["..."],
      "filters": [{"column": "...", "operator": "ILIKE", "value": "..."}],
      "date_range": {"start": "...", "end": "...", "grain": "...", "relative_label": "..."},
      "execution_strategy": "single_view|daily|compare_periods|historical|monthly|by_branch|by_product|demand_forecast",
      "candidate_views": ["semantic.vw_ventas_sede"],
      "preferred_view": "semantic.vw_ventas_sede",
      "assumptions": []
    }
  ],
  "confidence": 0.0-1.0,
  "visualization_candidate": true|false,
  "chart_type_hint": "bar|line|pie|auto",
  "needs_followup": false,
  "followup_reason": null,
  "followup_question": null
}

""")

    human = HumanMessage(content=f"Pregunta del usuario: {question}")
    planner_llm = LLM.with_structured_output(PlannerContract, method="function_calling")
    plan_raw = planner_llm.invoke([system, human])

    # langchain-openai 0.1.8 con function_calling puede devolver dict
    if isinstance(plan_raw, dict):
        plan = PlannerContract(**plan_raw)
    else:
        plan = plan_raw

    # ============================================================
    # VALIDACIÓN DE INTEGRIDAD POST-LLM
    # ============================================================
    validation_errors: List[str] = []

    # Seguridad: si por alguna razón el LLM devolvió demand_forecast con tareas SQL, limpiamos
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
        "messages": state.get("messages", []) + [
            AIMessage(
                content=f"[Planner] Intención: {plan.intent} | Tasks: {len(plan.tasks)} ({task_summary}) "
                        f"| Confianza: {plan.confidence} | Preferred: {prefs} | Followup: {plan.needs_followup}"
            )
        ]
    }
