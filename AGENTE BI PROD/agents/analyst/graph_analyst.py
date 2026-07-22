# agents/analyst/graph_analyst.py
import json
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.llm import LLM

# Import necesario para @traceable
from langsmith import traceable


def _build_sql_context(results: list, question: str) -> str:
    """Construye el contexto crudo de resultados SQL para el prompt."""
    if not results:
        return "No hay resultados SQL disponibles."

    tasks_context = []
    for contract in results:
        task_ctx = f"""
--- TAREA {contract.task_id} ---
Estrategia: {getattr(contract, 'execution_strategy', 'N/A')}
Vista usada: {getattr(contract, 'preferred_view', 'N/A')}
SQL ejecutado: {contract.generated_sql}
Filas: {contract.row_count}
Columnas: {contract.columns}
Datos: {json.dumps(contract.rows, indent=2, ensure_ascii=False, default=str)}
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
            f"| {r['fecha']} | {r['prediccion']} | {r['prediccion_con_buffer']} |"
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

    primer = forecast_results[0]
    metricas = primer.get("metricas", {})
    metricas_str = ", ".join([f"{k}={v:.3f}" for k, v in metricas.items()])

    return f"""
Pregunta original: {question}

El sistema ha generado un pronóstico de demanda con los siguientes datos:

- Producto: {primer['producto']}
- Sede: {primer['sede']}
- Días pronosticados: {len(forecast_results)}
- Modelo usado: {primer.get('modelo_version', 'N/A')}
- Métricas del modelo: {metricas_str}
- Buffer de seguridad (safety stock): {primer.get('safety_stock', 0):.1f} unidades

Tabla de predicciones:
{_format_forecast_table(forecast_results)}

1. Una breve introducción indicando qué se predijo (producto, sucursal y horizonte de tiempo).
2. Un análisis de la tendencia general de la demanda.
3. Destacar los días con mayor demanda y los días con menor demanda.
4. Explicar qué representa el buffer de seguridad y recomendar utilizarlo para planificar inventario o producción.
5. Finalizar indicando que la tabla contiene el detalle completo de las predicciones.
6. Mostrar la tabla al final.

No inventes información que no esté presente en los datos.
No repitas literalmente los valores de la tabla dentro del texto salvo para destacar máximos y mínimos.
"""


@traceable(name="Analyst: Generate Final Answer")
def analyst_node(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state["question"]
    results = state.get("sql_results", [])
    research_findings = state.get("research_findings")
    plan = state.get("plan")
    forecast_results = state.get("forecast_results")
    forecast_error = state.get("forecast_error")

    # ------------------------------------------------------------------
    # 0. Manejo de demand forecast
    # ------------------------------------------------------------------
    if plan and getattr(plan, "question_type", None) == "demand_forecast":
        system = SystemMessage(content="""
Eres un Analista de Negocio Senior de una cadena de cafeterías. Te especializas en interpretar pronósticos de demanda para equipos de operaciones y gerentes de tienda.
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
    errors = [r for r in results if r.status == "error"]
    if errors and not research_findings:
        err_msgs = "; ".join([f"Tarea {e.task_id}: {e.error_message}" for e in errors])
        return {
            "final_answer": f"Encontré problemas técnicos en algunas consultas: {err_msgs}",
            "last_agent": "analyst",
            "messages": [AIMessage(content="[Analyst] Reportando errores técnicos.")]
        }

    # ------------------------------------------------------------------
    # 3. Prompt según disponibilidad de research_findings
    # ------------------------------------------------------------------
    if research_findings:
        # ========== FLUJO DEEP RESEARCH ==========
        system = SystemMessage(content="""
Eres un Analista de Negocio Senior. Recibes un informe de investigación profundo generado internamente por un Researcher que consultó múltiples métricas de la base de datos de la empresa.

TU TRABAJO:
1. Pulir el informe y responder directamente la pregunta del usuario.
2. Usar los resultados SQL individuales (datos crudos) para refinar, verificar o matizar conclusiones del informe.
3. NO inventes datos. Si hay gaps o queries fallidas, menciónalos honestamente.
4. Estructura la respuesta en: Resumen Ejecutivo, Hallazgos Clave, Métricas y Comparativas, Conclusiones y Recomendaciones Accionables.
5. Mantén tono ejecutivo, claro y orientado a la toma de decisiones.
6. DONDE sea apropiado, usa lenguaje conversacional: "Según el análisis...", "Esto sugiere que...", "Mi recomendación es...".
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
Debes redactar UNA respuesta coherente que integre todos los hallazgos.

REGLAS:
- NO inventes datos.
- Integra los resultados de cada tarea en una narrativa fluida.
- Responde directamente a la pregunta del usuario.
- Destaca KPIs, comparaciones y tendencias cuando existan.
- Usa lenguaje natural y conversacional, no solo listas de números.
""")

        data_context = _build_sql_context(results, question)

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
