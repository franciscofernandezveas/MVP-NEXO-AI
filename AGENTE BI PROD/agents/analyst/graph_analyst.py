# agents/analyst/graph_analyst.py
# -------------------------------------------------

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.llm import LLM
from langsmith import traceable


def _get_attr(obj: Any, attr: str, default: Any = None) -> Any:
    """Lee atributo o clave, soportando objetos Pydantic y dicts."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _format_rows_markdown(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    """Formatea filas como tabla Markdown usando únicamente las columnas reportadas."""
    if not rows or not columns:
        return ""

    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---" for _ in columns]) + "|"
    lines = [header, separator]

    for row in rows:
        values = []
        for col in columns:
            val = row.get(col, "")
            # Formato numérico legible
            if isinstance(val, float):
                val = f"{val:,.2f}"
            elif isinstance(val, int):
                val = f"{val:,}"
            values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def _build_sql_context(results: list, question: str) -> str:
    """Construye el contexto crudo de resultados SQL para el prompt, incluyendo warnings y datos vacíos."""
    if not results:
        return "No hay resultados SQL disponibles."

    tasks_context = []
    for contract in results:
        task_id = _get_attr(contract, "task_id", "?")
        status = _get_attr(contract, "status", "unknown")
        row_count = _get_attr(contract, "row_count", 0)
        rows = _get_attr(contract, "rows", []) or []
        columns = _get_attr(contract, "columns", []) or []
        warnings = _get_attr(contract, "warnings", []) or []
        error_message = _get_attr(contract, "error_message", "")

        empty_note = ""
        if status == "success" and row_count == 0:
            empty_note = (
                "\n⚠️ ESTA CONSULTA SE EJECUTÓ CORRECTAMENTE PERO NO DEVOLVIÓ NINGUNA FILA "
                "(posiblemente no hay registros para los filtros solicitados).\n"
            )

        warnings_note = ""
        if warnings:
            warnings_note = (
                "\n⚠️ ADVERTENCIAS DEL SISTEMA:\n"
                + "\n".join(f"  - {w}" for w in warnings)
                + "\n"
            )

        error_note = ""
        if status == "error" and error_message:
            error_note = f"\n❌ ERROR EN TAREA: {error_message}\n"

        task_ctx = f"""
--- TAREA {task_id} ---
Estrategia: {_get_attr(contract, "execution_strategy", "N/A")}
Vista usada: {_get_attr(contract, "preferred_view", "N/A")}
SQL ejecutado: {_get_attr(contract, "generated_sql", "")}
Estado: {status}
Filas: {row_count}
Columnas: {columns}
{empty_note}{warnings_note}{error_note}
Datos:
{_format_rows_markdown(rows, columns) if rows else "(sin filas)"}

JSON raw:
{json.dumps(rows, indent=2, ensure_ascii=False, default=str)}
"""
        tasks_context.append(task_ctx)

    return f"""
Pregunta original: {question}

Resultados de {len(results)} tarea(s) SQL:
{chr(10).join(tasks_context)}
"""


def _format_forecast_table(forecast_results: List[Dict[str, Any]]) -> str:
    """Formatea los resultados del forecast como tabla markdown."""
    lineas = [
        "| Fecha | Predicción | Con buffer |",
        "|-------|-----------:|-----------:|",
    ]
    for r in forecast_results:
        lineas.append(
            f"| {r.get('fecha', 'N/A')} | {r.get('prediccion', 'N/A')} | {r.get('prediccion_con_buffer', 'N/A')} |"
        )
    return "\n".join(lineas)


def _build_forecast_context(question: str, forecast_results: List[Dict[str, Any]], forecast_error: str = None) -> str:
    if forecast_error:
        return f"""
Pregunta original: {question}

El sistema intentó generar un pronóstico de demanda, pero ocurrió un error técnico:
{forecast_error}

Redacta una respuesta clara y profesional para el usuario, explicando que no fue posible generar el pronóstico y sugiriendo posibles causas (falta de datos históricos, producto o sede no encontrados, problema técnico temporal).
"""

    if not forecast_results:
        return f"""
Pregunta original: {question}

El sistema intentó generar un pronóstico de demanda, pero no se obtuvieron resultados numéricos.
Redacta una respuesta honesta indicando que no hay datos suficientes para proyectar la demanda.
"""

    primer = forecast_results[0]
    metricas = _get_attr(primer, "metricas", {})
    metricas_str = ", ".join([f"{k}={v:.3f}" for k, v in metricas.items()])

    return f"""
Pregunta original: {question}

El sistema ha generado un pronóstico de demanda con los siguientes datos:

- Producto: {_get_attr(primer, 'producto', 'N/A')}
- Sede: {_get_attr(primer, 'sede', 'N/A')}
- Días pronosticados: {len(forecast_results)}
- Modelo usado: {_get_attr(primer, 'modelo_version', 'N/A')}
- Métricas del modelo: {metricas_str}
- Buffer de seguridad (safety stock): {_get_attr(primer, 'safety_stock', 0):.1f} unidades

Tabla de predicciones:
{_format_forecast_table(forecast_results)}

Instrucciones:
1. Empieza con una breve introducción indicando qué se predijo (producto, sucursal y horizonte de tiempo).
2. Analiza la tendencia general de la demanda.
3. Destaca los días con mayor demanda y los días con menor demanda.
4. Explica qué representa el buffer de seguridad y recomienda utilizarlo para planificar inventario o producción.
5. Finaliza indicando que la tabla contiene el detalle completo de las predicciones.
6. Muestra la tabla al final de la respuesta.
"""


@traceable(name="Analyst: Generate Final Answer")
def analyst_node(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state["question"]
    results = state.get("sql_results", []) or []
    research_findings = state.get("research_findings")
    plan = state.get("plan")
    forecast_results = state.get("forecast_results")
    forecast_error = state.get("forecast_error")

    # ------------------------------------------------------------------
    # 0. Manejo de demand forecast
    # ------------------------------------------------------------------
    if plan and _get_attr(plan, "question_type", None) == "demand_forecast":
        system = SystemMessage(content="""
Eres un Planificador de Demanda y Analista de Operaciones Senior de una cadena de cafeterías.
Interpretas pronósticos cuantitativos para gerentes de tienda y equipos de logística.

ESTRUCTURA DE RESPUESTA REQUERIDA:
1. **Contexto:** Qué producto, sede y horizonte temporal abarca la predicción.
2. **Análisis de Tendencia:** Identificar picos de alta demanda y valles.
3. **Gestión de Stock:** Explicar el propósito del buffer de seguridad (safety stock) para evitar quiebres de inventario.
4. **Detalle Tabular:** Presentar la tabla Markdown con las proyecciones completas.

REGLAS ABSOLUTAS:
- CERO ALUCINACIONES: No inventes datos, causas ni tendencias que no estén en la tabla de predicciones.
- NO RECALCULES: No modifiques ni extrapoles los números del pronóstico. Usa exactamente los valores entregados.
- Lenguaje claro: Explica conceptos técnicos (como safety stock) en lenguaje de negocio accesible para operarios.
- Destaca máximos y mínimos usando **negritas**.
""")

        data_context = _build_forecast_context(question, forecast_results, forecast_error)

        response = LLM.invoke([system, HumanMessage(content=data_context)])

        return {
            "final_answer": response.content,
            "last_agent": "analyst",
            "messages": [AIMessage(content=response.content, name="analyst")]
        }

    # ------------------------------------------------------------------
    # 1. Validación mínima: si no hay nada con qué trabajar
    # ------------------------------------------------------------------
    if not results and not research_findings:
        return {
            "final_answer": "No fue posible obtener datos para responder tu consulta.",
            "last_agent": "analyst",
            "messages": [AIMessage(content="[Analyst] Sin datos ni research findings para analizar.")]
        }

    # ------------------------------------------------------------------
    # 2. Detección de errores persistentes en SQL
    # ------------------------------------------------------------------
    errors = [r for r in results if _get_attr(r, "status") == "error"]
    successful = [r for r in results if _get_attr(r, "status") == "success"]
    empty_successful = [r for r in successful if _get_attr(r, "row_count", 0) == 0]

    # Si TODO falló y no hay research
    if errors and not successful and not research_findings:
        err_msgs = "; ".join([
            f"Tarea {_get_attr(e, 'task_id', '?')}: {_get_attr(e, 'error_message', '')}"
            for e in errors
        ])
        return {
            "final_answer": f"Encontré problemas técnicos en las consultas: {err_msgs}",
            "last_agent": "analyst",
            "messages": [AIMessage(content="[Analyst] Reportando errores técnicos.")]
        }

    # ------------------------------------------------------------------
    # 3. Prompt según disponibilidad de research_findings
    # ------------------------------------------------------------------
    if research_findings:
        # ========== FLUJO DEEP RESEARCH ==========
        system = SystemMessage(content="""
Eres un Consultor de Negocio Senior. Recibes un informe de investigación profunda generado por un agente interno (Researcher) respaldado por datos crudos de múltiples consultas SQL.

ESTRUCTURA OBLIGATORIA DEL INFORME:
- **Resumen Ejecutivo:** Conclusión principal en 1 o 2 frases.
- **Hallazgos Clave:** Puntos más destacados apoyados por las métricas.
- **Desglose y Comparativas:** Análisis de las dimensiones evaluadas.
- **Conclusiones y Recomendaciones:** Acciones prácticas sugeridas para el negocio.

REGLAS ABSOLUTAS:
1. CERO ALUCINACIONES: Usa ÚNICAMENTE los datos provistos. Si una consulta devolvió 0 filas, indícalo sin inventar explicaciones.
2. NO RECALCULES: No calcules porcentajes, sumas o promedios manuales si el SQL no los devolvió ya calculados.
3. VERIFICA EL INFORME: Contrasta el informe del Researcher con los "Datos crudos de apoyo (SQL results)". Si hay discrepancias o queries fallidas, menciónalas con transparencia.
4. FORMATO EJECUTIVO: Usa **negritas** para resaltar KPIs clave y tablas Markdown para comparaciones de múltiples valores.
5. TONO: Profesional, persuasivo y orientado a la toma de decisiones.
""")

        data_context = f"""
Pregunta original: {question}

--- INFORME DEL RESEARCHER ---
{research_findings}

--- DATOS CRUDOS DE APOYO (SQL results) ---
{_build_sql_context(results, question)}
"""

    else:
        # ========== FLUJO NORMAL DE KPIs ==========
        system = SystemMessage(content="""
Eres un Analista de Negocio Senior. Recibes datos estructurados de MÚLTIPLES consultas SQL independientes.
Debes redactar UNA respuesta coherente que integre todos los hallazgos y responda directamente la pregunta del usuario.

REGLAS ABSOLUTAS:
1. CERO ALUCINACIONES: Utiliza ÚNICAMENTE los datos provistos. Si una consulta devolvió 0 filas (lista vacía []), indícalo claramente: "No se registraron registros para los criterios solicitados".
2. NO RECALCULES: No calcules porcentajes, sumas o promedios manuales. Cíñete a las métricas que el SQL ya devolvió.
3. RESPUESTA DIRECTA: Comienza con la respuesta directa a la pregunta en la primera oración.
4. FORMATO EJECUTIVO:
   - Usa **negritas** para resaltar KPIs clave (ej. **$1,250,000**, **14%**).
   - Si hay comparación de múltiples valores o dimensiones, usa tablas Markdown breves.
   - No escribas párrafos gigantescos llenos de números crudos.
5. TONO: Profesional, analítico y conversacional orientado a decisiones ("Esto sugiere que...", "El volumen se concentra en...").
6. ADVERTENCIAS: Si los resultados incluyen warnings del sistema, menciónalos solo si afectan la interpretación de los datos.
""")

        data_context = _build_sql_context(results, question)

        # Caso especial: todas las consultas exitosas devolvieron 0 filas
        if empty_successful and not any(_get_attr(r, "row_count", 0) > 0 for r in results):
            data_context += (
                "\n\nNOTA IMPORTANTE: Todas las consultas SQL se ejecutaron correctamente, "
                "pero ninguna devolvió registros para los filtros indicados. "
                "Responde con honestidad que no hay datos disponibles para el período o criterios solicitados."
            )

    # ------------------------------------------------------------------
    # 4. Invocar LLM
    # ------------------------------------------------------------------
    response = LLM.invoke([system, HumanMessage(content=data_context)])

    # ------------------------------------------------------------------
    # 5. Retornar estado
    # ------------------------------------------------------------------
    return {
        "final_answer": response.content,
        "last_agent": "analyst",
        "messages": [AIMessage(content=response.content, name="analyst")]
    }
