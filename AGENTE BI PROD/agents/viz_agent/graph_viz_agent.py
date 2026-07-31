# agents/viz_agent/graph_viz_agent.py
# -------------------------------------------------
"""
Subgrafo Viz Agent - Genera especificación JSON del gráfico.
Soporta: bar, line, pie, scatter, heatmap (cohortes).
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from core.llm import LLM
from core.contracts import VizSpecContract

logger = logging.getLogger("bi_orchestrator")


class VIZAgentSTATE(TypedDict):
    question: str
    sql_rows: List[Dict[str, Any]]
    sql_columns: List[str]
    chart_type_hint: Optional[str]
    messages: List[Any]
    figure_spec: Optional[Dict[str, Any]]
    error_message: str
    attempts: int
    contract: Optional[VizSpecContract]


# ============================================================================
# Metadatos de columnas
# ============================================================================

def _is_temporal_column(column_name: str) -> bool:
    temporal_markers = [
        "fecha", "date", "mes", "anio", "año", "dia", "día",
        "semana", "hora", "year", "month", "day", "week"
    ]
    return any(marker in column_name.lower() for marker in temporal_markers)


def _infer_column_metadata(
    rows: List[Dict[str, Any]],
    columns: List[str]
) -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    if not rows:
        return metadata

    for col in columns:
        values = [row.get(col) for row in rows if row.get(col) is not None]
        if not values:
            metadata[col] = {"type": "empty", "unique_count": 0, "is_temporal": _is_temporal_column(col)}
            continue

        if all(isinstance(v, (int, float)) for v in values):
            col_type = "numeric"
        elif all(isinstance(v, str) for v in values):
            col_type = "categorical"
        else:
            col_type = "mixed"

        metadata[col] = {
            "type": col_type,
            "unique_count": len(set(str(v) for v in values)),
            "is_temporal": _is_temporal_column(col),
            "sample": [str(v) for v in values[:3]],
        }

    return metadata


def _detect_intent(question: str) -> str:
    q = question.lower()

    if any(k in q for k in ["comparar", "comparación", "versus", "vs", "ranking", "top", "más vendido", "mayor"]):
        return "comparison"

    if any(k in q for k in ["proporción", "participación", "porcentaje", "distribución", "mix", "pie", "torta"]):
        return "proportion"

    if any(k in q for k in ["evolución", "tendencia", "serie temporal", "histórico", "a lo largo del tiempo", "como cambia"]):
        return "trend"

    if any(k in q for k in ["correlación", "relación entre", "scatter", "dispersión"]):
        return "correlation"

    if any(k in q for k in ["cohorte", "matriz", "heatmap", "mapa de calor", "retención"]):
        return "cohort"

    return "auto"


# ============================================================================
# Generación de especificación
# ============================================================================

def viz_generate_spec(state: VIZAgentSTATE) -> Dict[str, Any]:
    system = SystemMessage(content="""
Eres un Director de Visualización de Datos (BI) Senior. Tu trabajo es analizar los datos estructurados y la pregunta del usuario para elegir el tipo de gráfico óptimo y generar su especificación JSON.

TIPOS DE GRÁFICO SOPORTADOS Y CUÁNDO USARLOS:
1. "bar" (barras): Comparaciones entre categorías, rankings de sucursales/productos, o evolución con pocas fechas. Usa "orientation": "h" si los nombres de categoría son largos.
2. "line" (líneas): EXCLUSIVO para series temporales continuas (evolución diaria/semanal/mensual). NUNCA uses líneas para comparar sucursales o categorías independientes.
3. "pie" (torta): Proporciones o participación sobre un total, con máximo 8 categorías. Si hay más de 8 categorías, usa "bar" en su lugar.
4. "scatter" (dispersión): Relación entre dos variables numéricas (correlación).
5. "heatmap" (mapa de calor / cohorte): Cruce de dos dimensiones categóricas/temporales con una métrica numérica representada por color.

REGLAS ABSOLUTAS:
- La salida debe ser un JSON válido con estos campos exactos:
  {
    "type": "bar" | "line" | "pie" | "scatter" | "heatmap" | null,
    "title": "Título ejecutivo del gráfico",
    "x": "columna eje X",
    "y": "columna eje Y",
    "z": "columna métrica numérica para heatmap (opcional para otros)",
    "color": "columna opcional para agrupar series o para el color en heatmap",
    "orientation": "v" | "h" (solo para barras)"
  }
- "x" y "y" deben ser nombres EXACTOS de la lista "Columnas disponibles".
- Para heatmap: "x" = dimensión 1, "y" = dimensión 2, "z" = métrica numérica del color.
- Responde SOLO con el JSON, sin markdown ni explicaciones.
- Si los datos no son aptos para visualización, devuelve {"type": null}.
""")

    rows = state.get("sql_rows", []) or []
    columns = state.get("sql_columns", []) or []
    hint = state.get("chart_type_hint", "auto")
    question = state.get("question", "")

    sample_size = min(8, len(rows))
    rows_sample = rows[:sample_size]
    metadata = _infer_column_metadata(rows_sample, columns)
    intent = _detect_intent(question)

    column_descriptions = []
    for col in columns:
        meta = metadata.get(col, {})
        desc = f"{col} ({meta.get('type', 'unknown')}, únicos estimados: {meta.get('unique_count', '?')}"
        if meta.get("is_temporal"):
            desc += ", temporal"
        desc += f", ejemplos: {meta.get('sample', [])})"
        column_descriptions.append(desc)

    data_preview = json.dumps(rows_sample, ensure_ascii=False, default=str)

    human = HumanMessage(content=f"""
Pregunta original del usuario: {question}
Intención detectada: {intent}
Columnas disponibles: {columns}
Descripción de columnas:
{chr(10).join(column_descriptions)}
Sugerencia inicial de tipo: {hint}

Muestra de datos (primeras {sample_size} de {len(rows)} filas):
{data_preview}

Genera la especificación JSON óptima para este análisis. Responde SOLO con el JSON:
""")

    response: Any = None
    try:
        response = LLM.invoke([system, human])
        content = response.content

        logger.info(f"[Viz] Respuesta LLM: {content[:300]}...")

        match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
        json_str = match.group(1).strip() if match else content.strip()
        json_str = json_str.strip().lstrip("`").rstrip("`").strip()

        spec = json.loads(json_str)
        logger.info(f"[Viz] Spec parseada: {spec}")

    except json.JSONDecodeError as e:
        logger.warning(f"[Viz] Error parseando JSON: {e}. Usando fallback.")
        spec = _generate_fallback_spec(state, metadata, intent)
    except Exception as e:
        logger.warning(f"[Viz] Error general en generación: {e}")
        spec = _generate_fallback_spec(state, metadata, intent)

    spec = _validate_and_fix_spec(spec, columns, rows, metadata)

    new_messages = [AIMessage(content="[Viz] Spec generada")]
    if response is not None:
        new_messages.insert(0, response)

    return {
        "figure_spec": spec,
        "messages": state.get("messages", []) + new_messages,
    }


def _generate_fallback_spec(
    state: VIZAgentSTATE,
    metadata: Dict[str, Dict[str, Any]],
    intent: str
) -> Dict[str, Any]:
    columns = state.get("sql_columns", []) or []
    if not columns:
        return {"type": None}

    numeric_cols = [c for c, m in metadata.items() if m.get("type") == "numeric"]
    categorical_cols = [c for c, m in metadata.items() if m.get("type") == "categorical"]
    temporal_cols = [c for c, m in metadata.items() if m.get("is_temporal")]

    if not numeric_cols:
        return {"type": None}

    if intent in ("trend", "cohort") and temporal_cols:
        x_col = temporal_cols[0]
    elif categorical_cols:
        x_col = categorical_cols[0]
    else:
        x_col = columns[0]

    y_col = numeric_cols[-1]

    chart_type = "bar"
    if intent == "trend" and x_col in temporal_cols:
        chart_type = "line"
    elif intent == "proportion" and x_col in categorical_cols:
        unique_count = metadata.get(x_col, {}).get("unique_count", 999)
        if 2 <= unique_count <= 8:
            chart_type = "pie"
        else:
            chart_type = "bar"
    elif intent == "correlation" and len(numeric_cols) >= 2:
        chart_type = "scatter"
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
    elif intent == "cohort" and len(categorical_cols) >= 2:
        chart_type = "heatmap"
        x_col = categorical_cols[0]
        y_col = categorical_cols[1]

    orientation = "h" if intent in ("comparison", "proportion") and x_col in categorical_cols else "v"

    return {
        "type": chart_type,
        "title": f"Análisis: {state.get('question', 'Datos')[:50]}".strip(),
        "x": x_col,
        "y": y_col,
        "z": None,
        "color": None,
        "orientation": orientation,
    }


def _validate_and_fix_spec(
    spec: Dict[str, Any],
    available_columns: List[str],
    data_rows: List[Dict[str, Any]],
    metadata: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        return _generate_fallback_spec(
            {"sql_columns": available_columns, "sql_rows": data_rows, "question": ""},
            metadata,
            "auto"
        )

    valid_types = ["bar", "line", "pie", "scatter", "heatmap", None]
    if spec.get("type") not in valid_types:
        spec["type"] = "bar"

    for axis in ["x", "y", "z", "color"]:
        val = spec.get(axis)
        if val and val not in available_columns:
            if axis == "x" and available_columns:
                spec[axis] = available_columns[0]
            elif axis == "y" and len(available_columns) > 1:
                spec[axis] = available_columns[-1]
            else:
                spec[axis] = None

    if not spec.get("title"):
        spec["title"] = f"Análisis visual de {spec.get('type', 'datos')}"

    # Si heatmap no tiene z, usar color o buscar numérica
    if spec.get("type") == "heatmap":
        z_val = spec.get("z") or spec.get("color")
        if not z_val or z_val not in available_columns:
            numeric_cols = [c for c, m in metadata.items() if m.get("type") == "numeric"]
            z_val = numeric_cols[0] if numeric_cols else None
        spec["z"] = z_val
        spec["color"] = z_val

    # Pie: máximo 8 categorías
    if spec.get("type") == "pie" and data_rows:
        x_col = spec.get("x")
        if x_col:
            unique_count = len(set(str(row.get(x_col)) for row in data_rows))
            if unique_count > 8:
                logger.info(f"[Viz] Pie chart con {unique_count} categorías; forzando barras horizontales.")
                spec["type"] = "bar"
                spec["orientation"] = "h"

    # Line: requiere eje temporal
    if spec.get("type") == "line":
        x_col = spec.get("x")
        if x_col and not metadata.get(x_col, {}).get("is_temporal"):
            logger.info(f"[Viz] Línea sin eje temporal; forzando barras.")
            spec["type"] = "bar"
            spec["orientation"] = "v"

    return spec


def viz_package(state: VIZAgentSTATE) -> Dict[str, Any]:
    err = state.get("error_message", "")
    spec = state.get("figure_spec", {}) or {}
    columns = state.get("sql_columns", []) or []

    suitable = bool(
        spec.get("type") and
        spec.get("type") != "null" and
        spec.get("x") in columns and
        spec.get("y") in columns
    )

    logger.info(f"[Viz] Evaluando spec - suitable: {suitable}, spec: {spec}")

    contract = VizSpecContract(
        status="success" if not err and suitable else "error",
        chart_type=spec.get("type") if suitable else None,
        title=spec.get("title") if suitable else None,
        x_axis=spec.get("x") if suitable else None,
        y_axis=spec.get("y") if suitable else None,
        z_axis=spec.get("z") if suitable else None,
        color_column=spec.get("color") if suitable else None,
        orientation=spec.get("orientation", "v") if suitable else "v",
        figure_spec=spec if suitable else None,
        reasoning="Spec generada correctamente" if not err and suitable else f"Fallo: {err or 'datos no adecuados para visualización'}",
        error_message=err or None,
        suitable_for_visualization=suitable,
    )

    logger.info(f"[Viz] Contrato generado - status: {contract.status}, chart_type: {contract.chart_type}")

    return {"contract": contract}


# Compilación del subgrafo
viz_builder = StateGraph(VIZAgentSTATE)
viz_builder.add_node("generate_spec", viz_generate_spec)
viz_builder.add_node("package", viz_package)

viz_builder.add_edge(START, "generate_spec")
viz_builder.add_edge("generate_spec", "package")
viz_builder.add_edge("package", END)

VIZ_SUBGRAPH = viz_builder.compile()
