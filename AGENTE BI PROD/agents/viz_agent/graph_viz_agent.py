"""
Subgrafo Viz Agent - Genera especificación JSON del gráfico.
NO renderiza Plotly. El renderizado ocurre en un nodo posterior del orquestador.
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


def viz_generate_spec(state: VIZAgentSTATE) -> Dict[str, Any]:
    """
    Usa LLM para generar una especificación JSON del gráfico.
    """
    system = SystemMessage(content="""
Eres un experto en visualización de datos. Genera una especificación JSON para un gráfico.

REGLAS:
- La salida debe ser un JSON válido con estos campos exactos:
  {
    "type": "bar" | "line" | "pie" | "scatter",
    "title": "string descriptivo",
    "x": "nombre de columna para eje X",
    "y": "nombre de columna para eje Y",
    "color": "nombre de columna para agrupar por color (opcional)",
    "orientation": "v" | "h" (orientación del gráfico, opcional)
  }
- Solo usa columnas que existan en los datos proporcionados.
- Para gráficos de barras con múltiples categorías, usa "color" para agrupar.
- Para gráficos de líneas con múltiples series, usa "color" para distinguir líneas.
- Responde SOLO con el JSON, sin markdown, sin explicaciones adicionales.
- Si los datos no son adecuados para visualización, devuelve type=null.
""")

    # Preparar datos para el LLM
    data_preview = json.dumps(state["sql_rows"][:3], ensure_ascii=False, default=str) if state["sql_rows"] else "[]"
    hint = state.get("chart_type_hint", "auto")
    
    # Analizar tipos de datos para sugerir mejor visualización
    column_types = {}
    if state["sql_rows"]:
        for col in state["sql_columns"]:
            sample_values = [row.get(col) for row in state["sql_rows"][:5] if row.get(col) is not None]
            if sample_values:
                if all(isinstance(v, (int, float)) for v in sample_values):
                    column_types[col] = "numeric"
                elif all(isinstance(v, str) for v in sample_values):
                    column_types[col] = "categorical"
                else:
                    column_types[col] = "mixed"
    
    human = HumanMessage(content=f"""
Datos SQL (muestra de {len(state['sql_rows'])} filas):
{data_preview}

Columnas disponibles: {state['sql_columns']}
Tipos de columnas: {column_types}
Tipo sugerido por planner: {hint}
Pregunta original: {state['question']}

Genera la spec JSON (SOLO JSON, sin explicaciones):
""")
    
    try:
        response = LLM.invoke([system] + state.get("messages", []) + [human])
        content = response.content
        
        logger.info(f"[Viz] Respuesta LLM: {content[:200]}...")
        
        # Extraer JSON del contenido
        match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
        json_str = match.group(1).strip() if match else content.strip()
        
        # Limpiar el JSON
        json_str = json_str.strip().lstrip('`').rstrip('`').strip()
        
        # Parsear JSON
        spec = json.loads(json_str)
        logger.info(f"[Viz] Spec parseada: {spec}")
        
    except json.JSONDecodeError as e:
        logger.warning(f"[Viz] Error parseando JSON: {e}. Contenido: {content[:100]}...")
        # Fallback mejorado
        spec = _generate_fallback_spec(state, column_types)
    except Exception as e:
        logger.warning(f"[Viz] Error general en generación: {e}")
        spec = _generate_fallback_spec(state, {})

    # Validación y ajuste de columnas
    spec = _validate_and_fix_spec(spec, state["sql_columns"], state["sql_rows"])
    
    return {
        "figure_spec": spec,
        "messages": state.get("messages", []) + [response, AIMessage(content="[Viz] Spec generada")]
    }


def _generate_fallback_spec(state: VIZAgentSTATE, column_types: Dict[str, str]) -> Dict[str, Any]:
    """Genera una especificación de fallback más inteligente"""
    if not state["sql_columns"]:
        return {"type": None}
    
    # Intentar identificar columnas numéricas y categóricas
    numeric_cols = [col for col, col_type in column_types.items() if col_type == "numeric"]
    categorical_cols = [col for col, col_type in column_types.items() if col_type == "categorical"]
    
    if not categorical_cols:
        categorical_cols = [col for col in state["sql_columns"] if col not in numeric_cols][:1]
    
    x_col = categorical_cols[0] if categorical_cols else state["sql_columns"][0]
    y_col = numeric_cols[-1] if numeric_cols else state["sql_columns"][-1]
    
    return {
        "type": "bar",
        "title": f"Análisis de {state['question'][:30]}...",
        "x": x_col,
        "y": y_col,
        "color": None
    }


def _validate_and_fix_spec(spec: Dict[str, Any], available_columns: List[str], data_rows: List[Dict]) -> Dict[str, Any]:
    """Valida y corrige la especificación"""
    if not isinstance(spec, dict):
        return {"type": "bar", "title": "Gráfico por defecto", "x": None, "y": None}
    
    # Validar tipo de gráfico
    valid_types = ["bar", "line", "pie", "scatter", None]
    if spec.get("type") not in valid_types:
        spec["type"] = "bar"
    
    # Validar columnas
    for axis in ["x", "y", "color"]:
        if spec.get(axis) and spec[axis] not in available_columns:
            if axis == "x" and available_columns:
                spec[axis] = available_columns[0]
            elif axis == "y" and len(available_columns) > 1:
                spec[axis] = available_columns[-1]
            else:
                spec[axis] = None
    
    # Asegurar que haya título
    if not spec.get("title"):
        spec["title"] = f"Gráfico de {spec.get('type', 'datos')}"
    
    return spec


def viz_package(state: VIZAgentSTATE) -> Dict[str, Any]:
    err = state.get("error_message", "")
    spec = state.get("figure_spec", {})
    
    # Verificar si es adecuado para visualización
    suitable = bool(
        spec.get("type") and 
        spec.get("type") != "null" and
        spec.get("x") and 
        spec.get("y") and 
        spec.get("x") in state["sql_columns"] and 
        spec.get("y") in state["sql_columns"]
    )
    
    logger.info(f"[Viz] Evaluando spec - suitable: {suitable}, spec: {spec}")
    
    contract = VizSpecContract(
        status="success" if not err and suitable else "error",
        chart_type=spec.get("type") if suitable else None,
        title=spec.get("title") if suitable else None,
        x_axis=spec.get("x") if suitable else None,
        y_axis=spec.get("y") if suitable else None,
        color_column=spec.get("color") if suitable else None,
        orientation=spec.get("orientation", "v") if suitable else "v",
        figure_spec=spec if suitable else None,
        reasoning="Spec generada correctamente" if not err and suitable else f"Fallo: {err or 'datos no adecuados para visualización'}",
        error_message=err or None,
        suitable_for_visualization=suitable
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
