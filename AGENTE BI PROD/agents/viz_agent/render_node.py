# agents/viz_agent/render_node.py
# -------------------------------------------------

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from langchain_core.messages import AIMessage

logger = logging.getLogger("bi_orchestrator")


def render_plotly_node(state: Dict[str, Any]) -> Dict[str, Any]:
    viz_contract = state.get("viz_result")
    sql_results = state.get("sql_results", []) or []

    if not viz_contract:
        logger.warning("[Render] No hay contrato de visualización")
        return {"viz_rendered": False, "last_agent": "render_plotly"}

    chart_type = _get(viz_contract, "chart_type")
    title = _get(viz_contract, "title")
    x_axis = _get(viz_contract, "x_axis")
    y_axis = _get(viz_contract, "y_axis")
    z_axis = _get(viz_contract, "z_axis")
    color_column = _get(viz_contract, "color_column")
    orientation = _get(viz_contract, "orientation") or "v"

    if not chart_type or chart_type == "null":
        logger.warning("[Render] Tipo de gráfico no especificado o nulo")
        return {"viz_rendered": False, "last_agent": "render_plotly"}

    if not x_axis or not y_axis:
        logger.warning("[Render] No se especificaron ejes")
        return {"viz_rendered": False, "last_agent": "render_plotly"}

    df = _select_dataframe(sql_results, x_axis, y_axis, color_column, z_axis)
    if df is None or df.empty:
        logger.warning("[Render] No hay datos para renderizar")
        return {"viz_rendered": False, "last_agent": "render_plotly"}

    logger.info(f"[Render] DataFrame creado con {len(df)} filas y columnas: {list(df.columns)}")

    x_axis, y_axis, color_column, z_axis = _resolve_columns(df, x_axis, y_axis, color_column, z_axis)

    if not x_axis or not y_axis:
        logger.warning("[Render] No se pudieron determinar las columnas para graficar")
        return {"viz_rendered": False, "last_agent": "render_plotly"}

    try:
        fig = _create_figure(
            df=df,
            chart_type=chart_type,
            x_axis=x_axis,
            y_axis=y_axis,
            z_axis=z_axis,
            title=title or "Gráfico de datos",
            color_column=color_column,
            orientation=orientation,
        )

        if fig is None:
            logger.warning("[Render] No se pudo crear la figura")
            return {"viz_rendered": False, "last_agent": "render_plotly"}

        _apply_layout(fig, title or "Gráfico de datos", chart_type, orientation)

        files_dir = Path(os.getenv("FILES_DIR", "/app/files")).resolve()
        charts_dir = files_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        chart_html_path = charts_dir / "chart.html"
        chart_png_path = charts_dir / "chart.png"

        with open(str(chart_html_path), "w", encoding="utf-8") as f:
            f.write(fig.to_html(include_plotlyjs="cdn"))

        pio.write_image(fig, str(chart_png_path), format="png", width=1000, height=600, scale=2)

        logger.info(
            f"[Render] Gráfico {chart_type} guardado en {charts_dir} | "
            f"PNG: {chart_png_path} ({chart_png_path.stat().st_size} bytes)"
        )

        return {
            "viz_rendered": True,
            "last_agent": "render_plotly",
            "messages": [AIMessage(content=f"[Render] Gráfico {chart_type} guardado en {charts_dir}")],
        }

    except Exception as e:
        logger.error(f"[Render] Error renderizando gráfico: {e}", exc_info=True)
        return {
            "viz_rendered": False,
            "last_agent": "render_plotly",
            "messages": [AIMessage(content=f"[Render] Error: {str(e)}")]
        }


# ============================================================================
# Helpers
# ============================================================================

def _get(obj: Any, attr: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _select_dataframe(
    sql_results: List[Any],
    x_axis: str,
    y_axis: str,
    color_column: Optional[str],
    z_axis: Optional[str],
) -> Optional[pd.DataFrame]:
    """
    Elige el mejor dataframe entre los resultados SQL disponibles.
    Prioriza el que contenga x_axis, y_axis y preferentemente z/color.
    """
    candidates: List[pd.DataFrame] = []

    for result in sql_results:
        rows = _get(result, "rows") or []
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if x_axis in df.columns and y_axis in df.columns:
            candidates.append(df)

    if candidates:
        score_fn = lambda df: sum(
            1 for col in [x_axis, y_axis, color_column, z_axis] if col and col in df.columns
        )
        best = max(candidates, key=score_fn)
        return best.copy()

    for result in sql_results:
        rows = _get(result, "rows") or []
        if rows:
            return pd.DataFrame(rows).copy()

    return None


def _resolve_columns(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    color_column: Optional[str],
    z_axis: Optional[str],
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    x = x_axis if x_axis in df.columns else (df.columns[0] if len(df.columns) > 0 else None)
    y = y_axis if y_axis in df.columns else (df.columns[-1] if len(df.columns) > 1 else None)
    color = color_column if color_column and color_column in df.columns else None
    z = z_axis if z_axis and z_axis in df.columns else None
    return x, y, color, z


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def _coerce_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def _looks_like_date(series: pd.Series) -> bool:
    converted = _coerce_datetime(series)
    return converted.notna().sum() / max(len(series), 1) > 0.8


def _create_figure(
    df: pd.DataFrame,
    chart_type: str,
    x_axis: str,
    y_axis: str,
    z_axis: Optional[str],
    title: str,
    color_column: Optional[str],
    orientation: str,
) -> Optional[go.Figure]:
    df = df.copy()
    df[y_axis] = _coerce_numeric(df[y_axis])

    if chart_type == "line":
        return _create_line(df, x_axis, y_axis, color_column, title)

    if chart_type == "bar":
        return _create_bar(df, x_axis, y_axis, color_column, title, orientation)

    if chart_type == "pie":
        return _create_pie(df, x_axis, y_axis, title)

    if chart_type == "scatter":
        return _create_scatter(df, x_axis, y_axis, color_column, title)

    if chart_type == "heatmap":
        metric_col = z_axis or color_column
        return _create_heatmap(df, x_axis, y_axis, metric_col, title)

    logger.warning(f"[Render] Tipo '{chart_type}' no soportado; usando barras.")
    return _create_bar(df, x_axis, y_axis, color_column, title, "v")


def _create_line(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    color_column: Optional[str],
    title: str,
) -> go.Figure:
    df["_sort"] = _coerce_datetime(df[x_axis])
    if df["_sort"].notna().all():
        df = df.sort_values("_sort")
    else:
        df = df.sort_values(x_axis)
    df = df.drop(columns=["_sort"])

    if color_column:
        fig = px.line(df, x=x_axis, y=y_axis, color=color_column, title=title, markers=True)
    else:
        fig = px.line(df, x=x_axis, y=y_axis, title=title, markers=True)

    if _looks_like_date(df[x_axis]):
        fig.update_xaxes(tickformat="%Y-%m-%d")

    return fig


def _create_bar(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    color_column: Optional[str],
    title: str,
    orientation: str,
) -> go.Figure:
    group_cols = [x_axis]
    if color_column:
        group_cols.append(color_column)

    df_agg = df.groupby(group_cols, as_index=False)[y_axis].sum()

    if orientation == "h":
        fig = px.bar(df_agg, y=x_axis, x=y_axis, color=color_column, title=title, orientation="h")
        fig.update_yaxes(categoryorder="total ascending")
    else:
        fig = px.bar(df_agg, x=x_axis, y=y_axis, color=color_column, title=title)
        if len(df_agg[x_axis].unique()) > 20:
            fig.update_xaxes(categoryorder="total descending")

    return fig


def _create_pie(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    title: str,
) -> go.Figure:
    df_agg = df.groupby(x_axis, as_index=False)[y_axis].sum()

    if len(df_agg) > 8:
        top = df_agg.nlargest(7, y_axis)
        others_sum = df_agg[~df_agg[x_axis].isin(top[x_axis])][y_axis].sum()
        others_row = pd.DataFrame({x_axis: ["Otros"], y_axis: [others_sum]})
        df_agg = pd.concat([top, others_row], ignore_index=True)

    fig = px.pie(df_agg, values=y_axis, names=x_axis, title=title)
    fig.update_traces(textinfo="percent+label", sort=False)
    return fig


def _create_scatter(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    color_column: Optional[str],
    title: str,
) -> go.Figure:
    df[x_axis] = _coerce_numeric(df[x_axis])

    if color_column:
        fig = px.scatter(
            df, x=x_axis, y=y_axis, color=color_column,
            title=title, opacity=0.7, trendline="ols"
        )
    else:
        fig = px.scatter(df, x=x_axis, y=y_axis, title=title, opacity=0.7, trendline="ols")

    return fig


def _create_heatmap(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    metric_col: Optional[str],
    title: str,
) -> go.Figure:
    if not metric_col or metric_col not in df.columns:
        metric_col = _find_numeric_column(df, [x_axis, y_axis])

    if not metric_col:
        logger.warning("[Render] Heatmap sin métrica numérica válida; convirtiendo a barras.")
        return _create_bar(df, x_axis, y_axis, None, title, "v")

    df[metric_col] = _coerce_numeric(df[metric_col])

    pivot = df.pivot_table(
        index=y_axis,
        columns=x_axis,
        values=metric_col,
        aggfunc="sum",
        fill_value=0,
    )

    fig = px.imshow(
        pivot,
        labels=dict(x=x_axis, y=y_axis, color=metric_col),
        title=title,
        aspect="auto",
        color_continuous_scale="Blues",
    )

    fig.update_xaxes(side="top")
    return fig


def _find_numeric_column(df: pd.DataFrame, exclude: List[str]) -> Optional[str]:
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


def _apply_layout(fig: go.Figure, title: str, chart_type: str, orientation: str) -> None:
    fig.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18, "color": "#1f2937"},
        },
        font=dict(family="Arial, sans-serif", size=12, color="#374151"),
        paper_bgcolor="white",
        plot_bgcolor="#f9fafb",
        hovermode="closest",
        showlegend=chart_type not in ("pie",),
        margin=dict(l=60, r=40, t=80, b=60),
    )

    if chart_type in ("bar", "line", "scatter"):
        fig.update_yaxes(gridcolor="#e5e7eb", zeroline=False)
        fig.update_xaxes(gridcolor="#e5e7eb", zeroline=False)

    if chart_type == "bar" and orientation == "h":
        fig.update_layout(height=max(600, min(1200, 150 * len(fig.data[0].y))))
