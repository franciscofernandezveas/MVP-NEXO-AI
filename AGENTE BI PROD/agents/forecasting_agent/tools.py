# --------------------------------------------------------------
# tools.py
# Tool de LangChain para predicción de demanda.
# --------------------------------------------------------------
from typing import Optional
from langchain_core.tools import tool
import pandas as pd

from graph_demand_forecaster import (
    DemandForecasterState,
    train_or_load_artifact,
    predict_node,
)


@tool
def predecir_demanda(
    producto: str,
    sede: str,
    n_dias: int = 7,
    fecha_inicio: Optional[str] = None,
) -> str:
    """
    Predice la demanda diaria de un producto en una sede para los próximos N días.

    Args:
        producto: Nombre del producto (ej: americano, capuccino, latte).
        sede: Nombre de la sede (ej: Plaza Bolsillo, Merced, Tajamar).
        n_dias: Cantidad de días a pronosticar.
        fecha_inicio: Fecha inicial del pronóstico en formato YYYY-MM-DD.
                      Si no se indica, se usa el día después del último dato histórico.
    """
    if fecha_inicio is None:
        fecha_inicio = (pd.Timestamp.now() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    state = DemandForecasterState(
        question=f"Predice la demanda de {producto} en {sede}",
        producto=producto.strip().lower(),
        sede=sede.strip(),
        modo="n_dias",
        fecha_inicio=fecha_inicio,
        n_dias=n_dias,
        artifact=None,
        dataset=None,
        forecasts=[],
        messages=[],
    )

    state = train_or_load_artifact(state)
    result = predict_node(state)

    lineas = [
        f"**Pronóstico de demanda: {producto} @ {sede}**",
        f"Modelo: {state['artifact']['modelo_version']}",
        f"R² del modelo: {state['artifact']['metrics']['R2']:.3f}",
        f"Safety stock p80: +{state['artifact']['safety_stock']:.1f} unidades",
        "",
        "| Fecha | Predicción | Con buffer |",
        "|-------|-----------:|-----------:|",
    ]

    for r in result["forecasts"]:
        lineas.append(
            f"| {r['fecha']} | {r['prediccion']} | {r['prediccion_con_buffer']} |"
        )

    return "\n".join(lineas)
