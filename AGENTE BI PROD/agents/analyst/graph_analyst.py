# agents/analyst/graph_analyst.py
# -------------------------------------------------
# Cambios aplicados (revisión de honradez de next_agent_instruction):
#  REV-1 🔴 Branch 0A: needs_followup/clarificación ANTES de forecast y data-checks
#  REV-2 🔴 Branch forecast inyecta la instrucción del supervisor en el contexto
#  REV-3 🔴 Early-exit "sin datos" respeta la instrucción (una llamada LLM guiada)
#  REV-4 🟡 response_mode estructurado (sniffing de markers solo como fallback legacy)
#  REV-5 🟡 messages siempre contiene final_text (consistencia con contexto 2.1)
#  REV-6 🟡 Mensaje de error LLM genérico (no asume overflow de tokens)
#
# REQUIERE (compañero opcional pero recomendado):
#  - OrchestratorState.response_mode: Optional[str]
#  - safe_supervisor_node / _build_dynamic_instruction setean response_mode
#    ("partial" | "educated_guess" | "clarification") junto a la instrucción
# -------------------------------------------------

import json
import logging
import math
from typing import Any, Dict, List, Optional
from datetime import datetime, date

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.llm import LLM
from langsmith import traceable

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers genéricos
# ------------------------------------------------------------------
def _get_attr(obj: Any, attr: str, default: Any = None) -> Any:
    """Lee atributo o clave, soportando objetos Pydantic y dicts."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _sentence(text: str) -> str:
    s = (text or "").strip()
    if s and s[-1] not in ".!?":
        return s + "."
    return s


def _fmt_value(val: Any) -> str:
    """Formatea valores para tablas Markdown con locale español/chileno."""
    if val is None:
        return ""
    if isinstance(val, bool):
        return "Sí" if val else "No"
    if isinstance(val, (int, float)):
        if isinstance(val, float):
            if math.isnan(val):
                return "N/A"
            if math.isinf(val):
                return "∞"
        num_str = f"{val:,.2f}" if isinstance(val, float) else f"{val:,}"
        return num_str.replace(",", "X").replace(".", ",").replace("X", ".")
    if isinstance(val, (datetime, date)):
        return val.isoformat()
    return str(val)


# ------------------------------------------------------------------
# Compactación inteligente de resultados SQL
# ------------------------------------------------------------------
BASE_MAX_ROWS_PER_TASK = 20
ABSOLUTE_MAX_ROWS_PER_TASK = 30


def _compute_max_rows(n_tasks: int) -> int:
    if n_tasks <= 1:
        return ABSOLUTE_MAX_ROWS_PER_TASK
    if n_tasks <= 3:
        return BASE_MAX_ROWS_PER_TASK
    limit = max(6, int(BASE_MAX_ROWS_PER_TASK * 3 / n_tasks))
    return min(limit, ABSOLUTE_MAX_ROWS_PER_TASK)


def _format_rows_markdown(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    if not rows or not columns:
        return ""

    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---" for _ in columns]) + "|"
    lines = [header, separator]

    for row in rows:
        values = [_fmt_value(row.get(col, "")) for col in columns]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def _compute_numeric_summary(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    summaries = []
    for col in columns:
        values = []
        for row in rows:
            v = row.get(col)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                values.append(v)

        if not values:
            continue

        try:
            col_min = min(values)
            col_max = max(values)
            col_avg = sum(values) / len(values)
            summaries.append(
                f"- {col}: min={_fmt_value(col_min)}, "
                f"max={_fmt_value(col_max)}, "
                f"promedio={_fmt_value(col_avg)}, "
                f"muestras={len(values)}"
            )
        except Exception:
            continue

    if not summaries:
        return ""

    return "Resumen numérico de la muestra completa:\n" + "\n".join(summaries)


def _compact_task_summary(contract: Any, idx: int, max_rows: int) -> str:
    task_id = _get_attr(contract, "task_id", str(idx + 1))
    status = _get_attr(contract, "status", "unknown")
    can_answer = _get_attr(contract, "can_answer", False)
    row_count = _get_attr(contract, "row_count", 0)
    rows = _get_attr(contract, "rows", []) or []
    columns = _get_attr(contract, "columns", []) or []
    warnings = _get_attr(contract, "warnings", []) or []
    error_message = _get_attr(contract, "error_message", "")
    preferred_view = _get_attr(contract, "preferred_view", "N/A")
    execution_strategy = _get_attr(contract, "execution_strategy", "N/A")
    generated_sql = _get_attr(contract, "generated_sql", "")
    reasoning = _get_attr(contract, "reasoning", "")
    depends_on = _get_attr(contract, "depends_on", []) or []

    is_truncated = len(rows) > max_rows
    sample_rows = rows[:max_rows]

    parts = [
        f"=== Subtarea {task_id} ===",
        f"Depende de subtareas: {depends_on if depends_on else 'Ninguna'}",
        f"Estado: {status} | Puede responder: {'Sí' if can_answer else 'No'} | Filas totales: {row_count}",
        f"Vista principal: {preferred_view} | Estrategia: {execution_strategy}",
    ]

    if generated_sql:
        parts.append(f"SQL ejecutado: {generated_sql}")

    if reasoning:
        parts.append(f"Razonamiento técnico: {reasoning}")

    if warnings:
        parts.append("Advertencias:\n" + "\n".join(f"  - {w}" for w in warnings))

    if status == "error" and error_message:
        parts.append(f"Error: {error_message}")

    if status in ("success", "no_data") and row_count == 0:

        parts.append(
            "Nota: consulta ejecutada correctamente pero sin registros para los filtros indicados."
        )

    if sample_rows and columns:
        parts.append(f"Muestra de datos ({len(sample_rows)} de {row_count} filas):")
        parts.append(_format_rows_markdown(sample_rows, columns))

        if is_truncated:
            parts.append(
                f"[... Se truncaron {row_count - max_rows} filas para eficiencia de tokens ...]"
            )
            numeric_summary = _compute_numeric_summary(rows, columns)
            if numeric_summary:
                parts.append(numeric_summary)

    elif not sample_rows:
        parts.append("Sin registros devueltos.")

    return "\n".join(parts)


def _build_sql_context(results: list, question: str) -> str:
    if not results:
        return "No hay resultados SQL disponibles."

    max_rows = _compute_max_rows(len(results))
    tasks_context = [
        _compact_task_summary(r, idx, max_rows) for idx, r in enumerate(results)
    ]

    return f"""
Pregunta original: {question}

Resultados de {len(results)} subtarea(s) SQL (límite de muestra: {max_rows} filas por subtarea):
{chr(10).join(tasks_context)}
"""


def _format_forecast_table(forecast_results: List[Dict[str, Any]]) -> str:
    lineas = [
        "| Fecha | Predicción | Con buffer |",
        "|-------|-----------:|-----------:|",
    ]
    for r in forecast_results:
        lineas.append(
            f"| {r.get('fecha', 'N/A')} | {_fmt_value(r.get('prediccion'))} | {_fmt_value(r.get('prediccion_con_buffer'))} |"
        )
    return "\n".join(lineas)


def _build_forecast_context(
    question: str,
    forecast_results: List[Dict[str, Any]],
    forecast_error: Optional[str] = None,
    instruction: Optional[str] = None,  # REV-2
) -> str:
    # REV-2: la instrucción del supervisor viaja también en el camino forecast
    instruction_block = (
        f"\n--- INSTRUCCIÓN PRIORITARIA DEL SUPERVISOR ---\n{instruction}\n"
        if instruction else ""
    )

    if forecast_error:
        return f"""
Pregunta original: {question}
{instruction_block}
El sistema intentó generar un pronóstico de demanda, pero ocurrió un error técnico:
{forecast_error}

Redacta una respuesta clara y profesional para el usuario, explicando que no fue posible generar el pronóstico y sugiriendo posibles causas (falta de datos históricos, producto o sede no encontrados, problema técnico temporal).
"""

    if not forecast_results:
        return f"""
Pregunta original: {question}
{instruction_block}
El sistema intentó generar un pronóstico de demanda, pero no se obtuvieron resultados numéricos.
Redacta una respuesta honesta indicando que no hay datos suficientes para proyectar la demanda.
"""

    primer = forecast_results[0]
    metricas = _get_attr(primer, "metricas", {})
    metricas_str = ", ".join([f"{k}={_fmt_value(v)}" for k, v in metricas.items()])

    return f"""
Pregunta original: {question}
{instruction_block}
El sistema ha generado un pronóstico de demanda con los siguientes datos:

- Producto: {_get_attr(primer, 'producto', 'N/A')}
- Sede: {_get_attr(primer, 'sede', 'N/A')}
- Días pronosticados: {len(forecast_results)}
- Modelo usado: {_get_attr(primer, 'modelo_version', 'N/A')}
- Métricas del modelo: {metricas_str}
- Buffer de seguridad (safety stock): {_fmt_value(_get_attr(primer, 'safety_stock', 0))} unidades

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


# ------------------------------------------------------------------
# Detección de modo de respuesta
# ------------------------------------------------------------------
def _is_educated_guess_instruction(instruction: Optional[str]) -> bool:
    """LEGACY (REV-4): fallback cuando el supervisor no envía response_mode."""
    if not instruction:
        return False
    markers = [
        "educated guess", "estimación razonada", "respuesta parcial",
        "no se pudo completar", "límites", "gaps", "aproximación",
        "mejor estimación", "inferencia razonada"
    ]
    return any(marker in instruction.lower() for marker in markers)


def _is_partial_answer_instruction(instruction: Optional[str]) -> bool:
    """LEGACY (REV-4): fallback cuando el supervisor no envía response_mode."""
    if not instruction:
        return False
    markers = [
        "respuesta parcial", "algunas consultas no pudieron",
        "datos disponibles", "incompleto", "parcial"
    ]
    return any(marker in instruction.lower() for marker in markers)


def _resolve_response_mode(state: Dict[str, Any], instruction: Optional[str], results: list) -> str:
    """
    REV-4: prioriza la señal estructurada `response_mode` del supervisor.
    Fallback: sniffing de markers sobre la instrucción (legacy).
    Retorna: "educated_guess" | "partial" | "normal"
    """
    structured = state.get("response_mode")
    if structured in ("educated_guess", "partial", "normal"):
        return structured
    if structured:
        logger.warning(f"[Analyst] response_mode desconocido '{structured}', usando sniffing legacy")

    if _is_educated_guess_instruction(instruction):
        return "educated_guess"
    if _is_partial_answer_instruction(instruction) or _has_partial_data(results):
        return "partial"
    return "normal"


def _has_partial_data(results: list) -> bool:
    if not results:
        return False

    has_success_with_data = any(
        _get_attr(r, "status") == "success"
        and _get_attr(r, "row_count", 0) > 0
        for r in results
    )

    has_failure_or_empty = any(
        _get_attr(r, "status") == "error"
        or _get_attr(r, "status") == "no_data"
        or (
            _get_attr(r, "status") == "success"
            and _get_attr(r, "row_count", 0) == 0
        )
        for r in results
    )

    return has_success_with_data and has_failure_or_empty


# ------------------------------------------------------------------
# REV-1: Modo aclaración (needs_followup) — PRIORIDAD sobre todo lo demás
# ------------------------------------------------------------------
def _build_clarification_system() -> SystemMessage:
    return SystemMessage(content="""
Eres un asistente BI conversacional de una cadena de cafeterías.
El sistema NO pudo construir un plan ejecutable para la pregunta del usuario
y necesita una aclaración ANTES de consultar datos.

REGLAS ABSOLUTAS:
1. NUNCA se consultó nada: NO presentes datos, NO inventes cifras, NO des respuestas parciales.
2. Pide la aclaración de forma breve, cordial y específica (máximo 2-3 campos).
3. Si es útil, ofrece 1-2 ejemplos concretos de cómo reformular la consulta
   (mencionando sede, producto, periodo o métrica según lo que falte).
4. Tono cercano y profesional. Sin tablas. Máximo ~120 palabras.
5. Termina con una invitación clara a responder.
""")


def _build_clarification_context(question: str, plan: Any, instruction: Optional[str]) -> str:
    followup_reason = _get_attr(plan, "followup_reason", "") or ""
    missing = _get_attr(plan, "missing_information", []) or []

    parts = [
        f"Pregunta del usuario: {question}",
        "",
        f"Motivo por el que no se puede responder aún: {followup_reason or 'no especificado'}",
    ]
    if missing:
        parts.append(f"Información que falta: {', '.join(missing)}")
    if instruction:
        parts.append(f"Instrucción del supervisor: {instruction}")
    parts.append("")
    parts.append("Redacta la petición de aclaración para el usuario.")
    return "\n".join(parts)


def _deterministic_clarification_fallback(plan: Any) -> str:
    """Fallback sin LLM (idéntico en espíritu al guard (c) del supervisor)."""
    followup_reason = _get_attr(plan, "followup_reason", "") or ""
    missing = _get_attr(plan, "missing_information", []) or []

    answer = "No pude generar un plan ejecutable para tu consulta."
    if followup_reason:
        answer += f" {_sentence(followup_reason)}"
    if missing:
        answer += f" Información faltante: {', '.join(missing)}."
    return answer


# ------------------------------------------------------------------
# Prompts de modo
# ------------------------------------------------------------------
def _build_educated_guess_system() -> SystemMessage:
    return SystemMessage(content="""
Eres un Analista de Negocio Senior. El sistema NO pudo completar la consulta de forma definitiva,
pero cuenta con datos parciales. Tu trabajo es generar una **estimación razonada (educated guess)**
que sea útil para el usuario sin inventar información.

REGLAS ABSOLUTAS:
1. CERO ALUCINACIONES: Usa ÚNICAMENTE los datos disponibles. Si no tienes un dato, dilo explícitamente.
2. CUANTIFICA LA INCERTIDUMBRE: Usa frases como "aproximadamente", "alrededor de", "entre X e Y",
   "no podemos afirmar con certeza, pero...". Nunca presentes estimaciones como hechos definitivos.
3. EXPLICA LOS LÍMITES: En una sección final titulada **Limitaciones y supuestos**, explica claramente:
   - Qué información faltó.
   - Por qué no se pudo obtener.
   - Qué supuestos hiciste para dar una respuesta útil.
4. NO RECALCULES: No hagas cálculos complejos con datos parciales. Si haces una extrapolación simple,
   indica que es una aproximación conservadora.
5. FORMATO:
   - Comienza con una respuesta directa y honesta.
   - Presenta los datos disponibles con tablas Markdown si aplica.
   - Finaliza con la sección de limitaciones.
6. TONO: Profesional, cauteloso y útil. El usuario debe entender qué sabe el sistema y qué no.
""")


def _build_partial_answer_system() -> SystemMessage:
    return SystemMessage(content="""
Eres un Analista de Negocio Senior. El sistema obtuvo datos PARCIALES para responder la pregunta.
Algunas consultas funcionaron y otras fallaron o no devolvieron registros.

REGLAS ABSOLUTAS:
1. CERO ALUCINACIONES: Usa ÚNICAMENTE los datos que el SQL devolvió correctamente.
2. TRANSPARENCIA: Menciona claramente qué partes de la pregunta no pudieron responderse.
3. RESPUESTA DIRECTA: Responde lo que SÍ se puede responder con los datos disponibles.
4. NO RECALCULES: No intentes completar los gaps con cálculos manuales.
5. FORMATO:
   - Empieza con una respuesta concisa.
   - Muestra los datos disponibles con tablas Markdown breves.
   - Incluye una sección **Limitaciones** que enumere qué no se pudo obtener y por qué.
6. TONO: Profesional y honesto. Es mejor decir "no lo sé" que inventar.
""")


def _build_normal_answer_system(multi_query: bool = False, research_mode: bool = False) -> SystemMessage:
    multi_query_section = ""
    if multi_query:
        multi_query_section = """
--- GUÍA AVANZADA PARA MULTI-QUERY Y TAREAS COMPLEJAS ---
La respuesta del usuario requirió descomponer el problema en múltiples consultas independientes o encadenadas.
Tus responsabilidades para integrar esta información son:

1. **Correlación Cruzada:** Conecta los hallazgos de cada subtarea. Ejemplos:
   - Si Subtarea 1 consultó ventas de un producto por sede y Subtarea 2 consultó stock/merma del mismo producto, explica la relación entre ambas.
   - Si Subtarea A obtuvo un ranking/top-N y Subtarea B detalló esos mismos ítems, usa A para el panorama y B para el detalle.
2. **Coherencia Global:** Presenta una narrativa unificada. No separes la respuesta en compartimentos estancos por cada subtarea a menos que el usuario lo haya pedido explícitamente.
3. **Dependencias:** Si una subtarea depende de otra (campo `depends_on`), respeta ese orden lógico en tu explicación (primero causa, luego consecuencia o detalle).
4. **Transparencia en Vacíos:** Si una subtarea falló o no dio resultados, indícalo claramente, pero no dejes que eso opaque los datos exitosos.
5. **No Compares Manzanas con Naranjas:** No combines métricas de distinta naturaleza como si fueran la misma cosa (por ejemplo, ventas totales vs ticket promedio). Sé explícito sobre qué representa cada métrica.
"""

    if research_mode:
        content = f"""
Eres un Consultor de Negocio Senior. Recibes un informe de investigación profunda generado por un agente interno (Researcher) respaldado por datos crudos de múltiples consultas SQL.

{multi_query_section}

ESTRUCTURA OBLIGATORIA DEL INFORME:
- **Resumen Ejecutivo:** Conclusión principal en 1 o 2 frases.
- **Hallazgos Clave:** Puntos más destacados apoyados por las métricas.
- **Desglose y Comparativas:** Análisis de las dimensiones evaluadas.
- **Conclusiones y Recomendaciones:** Acciones prácticas sugeridas para el negocio.

REGLAS ABSOLUTAS:
1. CERO ALUCINACIONES: Usa ÚNICAMENTE los datos provistos. Si una consulta devolvió 0 filas, indícalo sin inventar explicaciones.
2. NO RECALCULES: No calcules porcentajes, sumas o promedios manuales si el SQL no los devolvió ya calculados.
3. VERIFICA EL INFORME: Contrasta el informe del Researcher con los "Datos crudos de apoyo (SQL results)". Si hay discrepancias o queries fallidas, menciónalas con transparencia. Prevalecen los datos crudos sobre el informe si hay contradicción.
4. FORMATO EJECUTIVO: Usa **negritas** para resaltar KPIs clave y tablas Markdown para comparaciones de múltiples valores.
5. TONO: Profesional, persuasivo y orientado a la toma de decisiones.
"""
    else:
        content = f"""
Eres un Analista de Negocio Senior. Recibes datos estructurados de una o varias consultas SQL independientes.
Debes redactar UNA respuesta coherente que integre todos los hallazgos y responda directamente la pregunta del usuario.

{multi_query_section}

REGLAS ABSOLUTAS:
1. CERO ALUCINACIONES: Utiliza ÚNICAMENTE los datos provistos. Si una consulta devolvió 0 filas, indícalo claramente: "No se registraron registros para los criterios solicitados".
2. NO RECALCULES: No calcules porcentajes, sumas o promedios manuales. Cíñete a las métricas que el SQL ya devolvió.
3. RESPUESTA DIRECTA: Comienza con la respuesta directa a la pregunta en la primera oración.
4. FORMATO EJECUTIVO:
   - Usa **negritas** para resaltar KPIs clave (ej. **$1.250.000**, **14%**).
   - Si hay comparación de múltiples valores o dimensiones, usa tablas Markdown breves.
   - No escribas párrafos gigantescos llenos de números crudos.
5. TONO: Profesional, analítico y conversacional orientado a decisiones ("Esto sugiere que...", "El volumen se concentra en...").
6. ADVERTENCIAS: Si los resultados incluyen warnings del sistema, menciónalos solo si afectan la interpretación de los datos.
7. INSTRUCCIÓN DEL SUPERVISOR: Si el contexto incluye una "INSTRUCCIÓN DEL SUPERVISOR", esta TIENE PRIORIDAD sobre las reglas de formato por defecto.
"""

    return SystemMessage(content=content)


# ------------------------------------------------------------------
# Nodo principal
# ------------------------------------------------------------------
@traceable(name="Analyst: Generate Final Answer")
def analyst_node(state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    question = state.get("question", "")
    results = state.get("sql_results", []) or []
    research_findings = state.get("research_findings")
    plan = state.get("plan")
    forecast_results = state.get("forecast_results")
    forecast_error = state.get("forecast_error")

    instruction = state.get("next_agent_instruction") or state.get("supervisor_instruction")
    is_multi_query = bool(plan and _get_attr(plan, "question_type") == "multi_query")

    if instruction:
        logger.info(f"[Analyst] Instrucción del supervisor: {instruction[:200]}...")

    # ------------------------------------------------------------------
    # 0A. REV-1: needs_followup → petición de aclaración
    #     VA PRIMERO: un plan demand_forecast sin params o un plan unknown
    #     caerían en los branches de forecast/sin-datos y la instrucción
    #     de aclaración nunca llegaría al usuario.
    # ------------------------------------------------------------------
    if plan and _get_attr(plan, "needs_followup", False):
        logger.info(
            f"[Analyst] Plan con needs_followup. Generando petición de aclaración. "
            f"Motivo: {(_get_attr(plan, 'followup_reason', '') or '')[:120]}"
        )
        try:
            response = LLM.invoke([
                _build_clarification_system(),
                HumanMessage(content=_build_clarification_context(question, plan, instruction)),
            ])
            final_text = response.content
        except Exception as e:
            logger.error(f"[Analyst] Error en LLM (clarificación): {e}")
            final_text = _deterministic_clarification_fallback(plan)

        return {
            "final_answer": final_text,
            "last_agent": "analyst",
            "messages": [AIMessage(content=final_text, name="analyst")],  # REV-5
            "next_agent_instruction": None,
        }

    # ------------------------------------------------------------------
    # 0. Manejo de demand forecast (REV-2: instrucción inyectada)
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
- PRIORIDAD: Si el contexto incluye una "INSTRUCCIÓN PRIORITARIA DEL SUPERVISOR", síguela antes que la estructura por defecto.
""")
        data_context = _build_forecast_context(
            question, forecast_results, forecast_error, instruction=instruction  # REV-2
        )

        try:
            response = LLM.invoke([system, HumanMessage(content=data_context)])
            final_text = response.content
        except Exception as e:
            logger.error(f"[Analyst] Error en LLM (forecast): {e}")
            final_text = "Ocurrió un error al procesar el pronóstico de demanda. Por favor, intenta nuevamente."

        return {
            "final_answer": final_text,
            "last_agent": "analyst",
            "messages": [AIMessage(content=final_text, name="analyst")],
            "next_agent_instruction": None,
        }

    # ------------------------------------------------------------------
    # 1. REV-3: Sin datos — la instrucción del supervisor YA NO se ignora
    # ------------------------------------------------------------------
    if not results and not research_findings:
        if instruction:
            # El supervisor pidió algo específico (ej. explicar por qué no hay
            # datos, generar respuesta explicativa): una llamada LLM guiada.
            logger.info("[Analyst] Sin datos, pero hay instrucción del supervisor. LLM guiado.")
            system = SystemMessage(content="""
Eres un Analista de Negocio Senior. No hay resultados SQL ni hallazgos disponibles.
Obedece la INSTRUCCIÓN DEL SUPERVISOR como prioridad absoluta. Sé breve, honesto
y accionable. No inventes datos.
""")
            context = f"Pregunta del usuario: {question}\n\nInstrucción del supervisor: {instruction}"
            try:
                response = LLM.invoke([system, HumanMessage(content=context)])
                final_text = response.content
            except Exception as e:
                logger.error(f"[Analyst] Error en LLM (sin datos + instrucción): {e}")
                final_text = "No fue posible obtener datos para responder tu consulta."
        else:
            final_text = "No fue posible obtener datos para responder tu consulta."

        return {
            "final_answer": final_text,
            "last_agent": "analyst",
            "messages": [AIMessage(content=final_text, name="analyst")],  # REV-5: publicar la respuesta real, no un log
            "next_agent_instruction": None,
        }

    # ------------------------------------------------------------------
    # 2. Detección de errores persistentes en SQL
    # ------------------------------------------------------------------
    errors = [r for r in results if _get_attr(r, "status") == "error"]
    successful = [r for r in results if _get_attr(r, "status") == "success"]

    response_mode = _resolve_response_mode(state, instruction, results)  # REV-4

    if errors and not successful and not research_findings and response_mode != "educated_guess":
        err_msgs = "; ".join([
            f"Tarea {_get_attr(e, 'task_id', '?')}: {_get_attr(e, 'error_message', '')}"
            for e in errors
        ])

        system = SystemMessage(content="""
Eres un Analista de Negocio Senior. Las consultas SQL fallaron todas. Redacta una respuesta breve, profesional y útil para el usuario.
No incluyas detalles técnicos internos, stack traces ni nombres de vistas. Explica de forma general qué ocurrió y qué puede intentar.
""")
        context = f"Pregunta: {question}\nError técnico interno: {err_msgs}"
        if instruction:
            context += f"\nInstrucción del supervisor: {instruction}"

        try:
            response = LLM.invoke([system, HumanMessage(content=context)])
            final_text = response.content
        except Exception as e:
            logger.error(f"[Analyst] Error en LLM (error técnico): {e}")
            final_text = "Hubo un problema técnico al consultar los datos. Por favor, verifica la pregunta o intenta más tarde."

        return {
            "final_answer": final_text,
            "last_agent": "analyst",
            "messages": [AIMessage(content=final_text, name="analyst")],
            "next_agent_instruction": None,
        }

    # ------------------------------------------------------------------
    # 3. Selección de prompt según modo de respuesta
    # ------------------------------------------------------------------
    research_mode = bool(research_findings)
    empty_successful = [r for r in successful if _get_attr(r, "row_count", 0) == 0]

    if response_mode == "educated_guess":
        system = _build_educated_guess_system()
        mode_label = "EDUCATED_GUESS"
    elif response_mode == "partial":
        system = _build_partial_answer_system()
        mode_label = "PARTIAL_ANSWER"
    else:
        system = _build_normal_answer_system(multi_query=is_multi_query, research_mode=research_mode)
        mode_label = "FINAL_ANSWER"

    multi_query_note = ""
    if is_multi_query:
        multi_query_note = (
            f"\n--- NOTA DE MULTI-QUERY ---\n"
            f"La pregunta fue dividida en {len(results)} subtareas SQL independientes. "
            f"Integra sus resultados en una respuesta coherente. "
            f"Menciona explícitamente si alguna subtarea no aportó datos.\n"
        )

    empty_note = ""
    if empty_successful and not any(_get_attr(r, "row_count", 0) > 0 for r in results):
        empty_note = (
            "\n\nNOTA IMPORTANTE: Todas las consultas SQL se ejecutaron correctamente, "
            "pero ninguna devolvió registros para los filtros indicados. "
            "Responde con honestidad que no hay datos disponibles para el período o criterios solicitados."
        )

    base_context = f"""
Pregunta original: {question}

--- INSTRUCCIÓN DEL SUPERVISOR (prioritaria) ---
{instruction or "Ninguna instrucción adicional."}

--- MODO DE RESPUESTA ---
{mode_label}
{multi_query_note}
"""

    if research_mode:
        data_context = base_context + f"""
--- INFORME DEL RESEARCHER ---
{research_findings}

--- DATOS CRUDOS DE APOYO (SQL results) ---
{_build_sql_context(results, question)}
{empty_note}
"""
    else:
        data_context = base_context + f"""
{_build_sql_context(results, question)}
{empty_note}
"""

    # ------------------------------------------------------------------
    # 4. Invocar LLM
    # ------------------------------------------------------------------
    try:
        response = LLM.invoke([system, HumanMessage(content=data_context)])
        final_text = response.content
    except Exception as e:
        # REV-6: no asumir overflow de tokens; mensaje genérico honesto
        logger.error(f"[Analyst] Error en LLM: {e}")
        final_text = (
            "Ocurrió un error al sintetizar los datos para tu consulta. "
            "Por favor, intenta nuevamente; si el problema persiste, "
            "reformula la pregunta de forma más específica."
        )

    return {
        "final_answer": final_text,
        "last_agent": "analyst",
        "messages": [AIMessage(content=final_text, name="analyst")],
        "next_agent_instruction": None,
    }
