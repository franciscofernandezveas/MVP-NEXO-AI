# agents/analyst/graph_analyst.py
import json
from typing import Any, Dict, List

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


def _build_sql_context(results: list, question: str) -> str:
    """Construye el contexto crudo de resultados SQL para el prompt."""
    if not results:
        return "No hay resultados SQL disponibles."

    tasks_context = []
    for contract in results:
        task_ctx = f"""
--- TAREA {_get_attr(contract, 'task_id', '?')} ---
Estrategia: {_get_attr(contract, 'execution_strategy', 'N/A')}
Vista usada: {_get_attr(contract, 'preferred_view', 'N/A')}
SQL ejecutado: {_get_attr(contract, 'generated_sql', '')}
Filas: {_get_attr(contract, 'row_count', 0)}
Columnas: {_get_attr(contract, 'columns', [])}
Datos: {json.dumps(_get_attr(contract, 'rows', []), indent=2, ensure_ascii=False, default=str)}
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
    errors = [r for r in results if _get_attr(r, "status") == "error"]
    if errors and not research_findings:
        err_msgs = "; ".join([
            f"Tarea {_get_attr(e, 'task_id', '?')}: {_get_attr(e, 'error_message', '')}"
            for e in errors
        ])
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
Eres un Analista de Negocio Senior. Redactas la respuesta final para el usuario.

=== PREGUNTA DEL USUARIO ===
{user_question}

=== PLAN ===
{plan_json}

=== RESULTADOS SQL ===
{sql_results_json}

=== HALLAZGOS DE INVESTIGACIÓN (si aplica) ===
{research_findings}

=== INSTRUCCIONES ===
1. Responde directamente a la pregunta. No expliques el proceso interno.
2. Todo número o comparación debe citar su origen con [tarea: tX].
3. NO inventes datos. Si una query devolvió status="error" o "unrecoverable", di "No contamos con ese dato" y explica el gap.
4. Estructura obligatoria:
   - Resumen Ejecutivo (3-5 bullets, lenguaje conversacional).
   - Hallazgos Clave (con evidencia SQL).
   - Métricas relevantes (valores exactos).
   - Limitaciones / Gaps (si aplica).
   - Recomendaciones Accionables (máximo 5, con nivel de confianza).
5. Si hay proyecciones, etiquétalas como "Proyección" y menciona incertidumbre.
6. Tono ejecutivo, claro y orientado a decisiones.

=== FORMATO DE SALIDA ===
{
  "confidence": "alta|media|baja",
  "gaps": ["..."],
  "citations": ["t1: total_ventas = 1.234.567", "t2: sede_merced = ..."],
  "response_text": "..."
}

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
