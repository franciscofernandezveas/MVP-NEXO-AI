import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
from typing import Dict, Any
from langchain_core.messages import AIMessage
import logging

logger = logging.getLogger("bi_orchestrator")


def render_plotly_node(state: Dict[str, Any]) -> Dict[str, Any]:
    viz_contract = state.get("viz_result")
    sql_results = state.get("sql_results", [])

    if not viz_contract:
        logger.warning("[Render] No hay contrato de visualización")
        return {"viz_rendered": False, "last_agent": "render_plotly"}

    if not sql_results:
        logger.warning("[Render] No hay resultados SQL")
        return {"viz_rendered": False, "last_agent": "render_plotly"}

    # Tomamos la primera tarea SQL
    data = sql_results[0].rows if sql_results else []
    if not data:
        logger.warning("[Render] No hay datos para renderizar")
        return {"viz_rendered": False, "last_agent": "render_plotly"}

    try:
        df = pd.DataFrame(data)
        logger.info(f"[Render] DataFrame creado con {len(df)} filas y columnas: {list(df.columns)}")

        # Obtener parámetros del contrato
        chart_type = getattr(viz_contract, 'chart_type', 'bar') or 'bar'
        title = getattr(viz_contract, 'title', 'Gráfico de datos')
        x_axis = getattr(viz_contract, 'x_axis', None)
        y_axis = getattr(viz_contract, 'y_axis', None)
        color_column = getattr(viz_contract, 'color_column', None)
        orientation = getattr(viz_contract, 'orientation', 'v')

        if not x_axis or not y_axis:
            logger.warning("[Render] No se especificaron ejes para el gráfico")
            return {"viz_rendered": False, "last_agent": "render_plotly"}

        # Verificar que las columnas existan
        if x_axis not in df.columns:
            logger.warning(f"[Render] Columna X '{x_axis}' no encontrada en datos")
            x_axis = df.columns[0] if len(df.columns) > 0 else None

        if y_axis not in df.columns:
            logger.warning(f"[Render] Columna Y '{y_axis}' no encontrada en datos")
            y_axis = df.columns[-1] if len(df.columns) > 1 else None

        if color_column and color_column not in df.columns:
            logger.warning(f"[Render] Columna Color '{color_column}' no encontrada en datos")
            color_column = None

        if not x_axis or not y_axis:
            logger.warning("[Render] No se pudieron determinar las columnas para graficar")
            return {"viz_rendered": False, "last_agent": "render_plotly"}

        # Generar gráfico según tipo con mejor manejo de datos
        fig = _create_figure(df, chart_type, x_axis, y_axis, title, color_column, orientation)
        
        if fig is None:
            logger.warning("[Render] No se pudo crear la figura")
            return {"viz_rendered": False, "last_agent": "render_plotly"}

        # Configurar layout mejorado
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            hovermode='closest',
            showlegend=True if color_column else False
        )

        # ✅ Guardar en ruta conocida: {BACKEND_DIR}/files/charts/
        backend_dir = Path(os.getenv("BACKEND_DIR", Path(__file__).resolve().parent.parent.parent.parent))
        charts_dir = backend_dir / "files" / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        chart_html_path = charts_dir / "chart.html"
        chart_png_path = charts_dir / "chart.png"

        # Guardar como HTML
        with open(str(chart_html_path), "w", encoding="utf-8") as f:
            f.write(fig.to_html(include_plotlyjs='cdn'))

        # Guardar como imagen PNG
        pio.write_image(fig, str(chart_png_path), format="png", width=1000, height=600, scale=2)
        
        logger.info(f"[Render] Gráfico guardado en {charts_dir}")
        return {
            "viz_rendered": True,
            "last_agent": "render_plotly",
            "messages": [AIMessage(content=f"[Render] Gráfico guardado en {charts_dir}")]
        }
        
    except Exception as e:
        logger.error(f"[Render] Error renderizando gráfico: {e}", exc_info=True)
        return {
            "viz_rendered": False,
            "last_agent": "render_plotly",
            "messages": [AIMessage(content=f"[Render] Error: {str(e)}")]
        }


def _create_figure(df: pd.DataFrame, chart_type: str, x_axis: str, y_axis: str, title: str, 
                   color_column: str = None, orientation: str = 'v') -> go.Figure:
    """Crea la figura con mejor manejo de datos y colores"""
    try:
        # Manejar datos categóricos en el eje X
        if df[x_axis].dtype == 'object' or df[x_axis].dtype == 'category':
            # Limitar el número de categorías para mejor visualización
            if len(df[x_axis].unique()) > 20:
                top_categories = df[x_axis].value_counts().head(20).index
                df = df[df[x_axis].isin(top_categories)]
                logger.info(f"[Render] Limitando a top 20 categorías en {x_axis}")

        # Generar gráfico según tipo
        if chart_type == "bar":
            if color_column and color_column in df.columns:
                fig = px.bar(df, x=x_axis, y=y_axis, color=color_column, 
                           title=title, orientation=orientation)
            else:
                fig = px.bar(df, x=x_axis, y=y_axis, title=title, orientation=orientation)
                
        elif chart_type == "line":
            if color_column and color_column in df.columns:
                fig = px.line(df, x=x_axis, y=y_axis, color=color_column, title=title)
            else:
                fig = px.line(df, x=x_axis, y=y_axis, title=title)
                
        elif chart_type == "pie":
            # Para gráficos de pie, usamos el eje X como nombres y Y como valores
            fig = px.pie(df, values=y_axis, names=x_axis, title=title)
            
        elif chart_type == "scatter":
            if color_column and color_column in df.columns:
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color_column, title=title)
            else:
                fig = px.scatter(df, x=x_axis, y=y_axis, title=title)
        else:
            # Fallback a gráfico de barras
            fig = px.bar(df, x=x_axis, y=y_axis, title=title)
            
        return fig
        
    except Exception as e:
        logger.error(f"[Render] Error creando figura: {e}")
        return None
