import re
import json
import logging
from typing import Any, Dict, List, Optional, Literal, NotRequired, Tuple
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from core.llm import LLM
from core.database import execute_sql_query
from core.contracts import SQLContract
from core.sql_utils import extract_views_used, extract_columns_used
from core.harness import BusinessMemory, is_view_allowed

_biz_mem = BusinessMemory.from_file()

logger = logging.getLogger("bi_orchestrator")


class SQLAgentState(TypedDict):
    question: str
    payload: Dict[str, Any]
    messages: List[Any]
    schema_info: str
    semantic_context: str
    allowed_views: List[str]
    preferred_view: NotRequired[Optional[str]]
    generated_sql: str
    query_result: Optional[List[Dict[str, Any]]]
    error_message: str
    contract: Optional[SQLContract]
    attempts: int
    schema_used: NotRequired[List[str]]


def _normalize(text: str) -> str:
    if not text:
        return ""
    return text.lower().strip().replace("_", " ").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")


def _build_sql_catalog(allowed_views: List[str]) -> Dict[str, Any]:
    catalog: Dict[str, Any] = {}
    for view_full_name in allowed_views:
        view_name = view_full_name.replace("semantic.", "").strip()
        view_info = _biz_mem.get_view(view_name)
        if not view_info:
            continue
        catalog[view_full_name] = {
            "tipo": view_info.tipo,
            "descripcion": view_info.descripcion,
            "granularidad": view_info.granularidad,
            "filtro_fecha": view_info.filtro_fecha,
            "metricas": list(view_info.metricas.keys()),
            "columnas_fecha": view_info.columnas_fecha,
            "notas": view_info.notas,
        }
    return catalog


def _column_exists_in_view(view_name: str, column_name: str) -> bool:
    clean_view = view_name.replace("semantic.", "").strip()
    view_info = _biz_mem.get_view(clean_view)
    if not view_info:
        return False

    requested = _normalize(column_name)
    available_cols: List[str] = list(view_info.metricas.keys()) + view_info.columnas_fecha
    available_normalized = [_normalize(c) for c in available_cols]

    for avail in available_normalized:
        if requested in avail or avail in requested:
            return True

    semantic_map = {
        "producto": ["producto", "descripcion", "descripción", "nombre_producto", "articulo", "artículo", "sku"],
        "sucursal": ["sucursal", "nombre_sede", "sede", "local", "tienda", "plaza", "ubicacion", "ubicación"],
        "categoria": ["categoria", "categoría", "categoria_nueva"],
        "subcategoria": ["subcategoria", "subcategoría"],
        "fecha": ["fecha", "fecha_completa", "fecha_venta", "mes", "anio", "año"],
        "venta_total": ["venta_total", "ventas", "ventas_totales", "subtotal_diario", "ingreso"],
        "unidades": ["unidades", "cantidad", "unidades_totales", "unidades_vendidas", "unidades_fidelizacion", "unidades_cortesia"],
        "transacciones": ["transacciones", "total_transacciones", "numero_transacciones"],
        "ticket_promedio": ["ticket_promedio"],
    }

    variants = semantic_map.get(requested, [requested])
    for variant in variants:
        v = _normalize(variant)
        for avail in available_normalized:
            if v in avail or avail in v:
                return True

    return False


def _validate_columns_in_sql(sql: str) -> Tuple[bool, str]:
    used_views = extract_views_used(sql)
    if not used_views:
        return True, ""

    for view_name in used_views:
        cols = extract_columns_used(sql)
        invalid_cols = []
        for col in cols:
            if not _column_exists_in_view(view_name, col):
                if col.lower() in ["as", "by", "on", "and", "or", "not", "sum", "count", "avg", "min", "max"]:
                    continue
                invalid_cols.append(col)

        if invalid_cols:
            clean_view = view_name.replace("semantic.", "")
            view_info = _biz_mem.get_view(clean_view)
            available = list(view_info.metricas.keys()) + view_info.columnas_fecha if view_info else []
            return False, (
                f"ERROR_INSALVABLE: La vista '{view_name}' no contiene las columnas "
                f"{invalid_cols}. Columnas VÁLIDAS: {available}. "
                f"No intentes 'corregir' usando otra columna."
            )

    return True, ""


def _build_sql_system_prompt(catalogo_detallado: Dict[str, Any], task_json: str, error_history: str) -> str:
    catalogo_str = json.dumps(catalogo_detallado, indent=2, ensure_ascii=False)

    prompt_template = """
Eres un Data Engineer senior experto en PostgreSQL. Generas UNA query SQL SELECT válida para responder una tarea del Planner.

=== CATÁLOGO ESTRUCTURADO DE VISTAS AUTORIZADAS ===
{catalogo_str}

=== TAREA A RESOLVER (SQLPayload) ===
{task_json}

=== ERRORES PREVIOS ===
{error_history}

=== PROTOCOLO OBLIGATORIO DE VALIDACIÓN ===
1. Extrae las columnas requeridas: task.metrics + task.dimensions + task.filters[*].column.
2. Para cada columna, verifica que exista con su nombre EXACTO en task.preferred_view (o la primera de task.candidate_views).
3. Si alguna columna NO existe:
   - Devuelve status="unrecoverable".
   - En reason_for_view_choice escribe: "Columna X no existe en semantic.vw_...".
   - No inventes alias ni columnas alternativas.
4. Si todas existen, escribe la query.

=== REGLAS ABSOLUTAS ===
1. Usa ÚNICAMENTE `semantic.<vista>`. Prohibido cualquier otro esquema.
2. No uses columnas que no estén en el catálogo de arriba.
3. Para filtros de texto (sede, local, producto, categoria) usa ILIKE o LOWER(col) = LOWER('valor'). Nunca = directo.
4. Usa task.date_range para el WHERE de fecha. Aplica DATE_TRUNC según grain si es necesario.
5. Incluye GROUP BY por todas las dimensiones no agregadas.
6. No uses SELECT *.
7. No uses LIMIT a menos que se pida TOP N explícitamente.
8. Prohibido DELETE, DROP, INSERT, UPDATE, TRUNCATE.
9. Si la tarea es demand_forecast, genera solo la query base de serie histórica. El forecast se hará en el nodo forecaster.

=== FORMATO DE SALIDA (SQLContract) ===
Devuelve ÚNICAMENTE JSON válido con esta estructura exacta:

{{
  "task_id": "t1",
  "status": "success|error|partial|needs_clarification|unrecoverable",
  "generated_sql": "SELECT ...",
  "columns": ["col1", "col2"],
  "rows": [],
  "row_count": 0,
  "error_message": null,
  "schema_used": ["semantic"],
  "can_answer": true,
  "reasoning": "Breve explicación técnica",
  "needs_followup": false,
  "warnings": [],
  "allowed_views": ["semantic.vw_..."],
  "preferred_view": "semantic.vw_...",
  "semantic_context_used": "Resumen del catálogo usado",
  "query_confidence": 1.0,
  "reason_for_view_choice": "Vista X elegida porque contiene todas las columnas requeridas"
}}

Si la validación falla:
{{
  "status": "unrecoverable",
  "generated_sql": null,
  "can_answer": false,
  "query_confidence": 0.0,
  "reason_for_view_choice": "Columna X no existe en semantic.vw_...",
  "error_message": "Columna X no existe en semantic.vw_..."
}}
"""
    return prompt_template.format(
        catalogo_str=catalogo_str,
        task_json=task_json,
        error_history=error_history
    )


def sql_fetch_schema(state: SQLAgentState) -> Dict[str, Any]:
    if state.get("schema_info") and state["schema_info"].strip():
        schema = state["schema_info"]
        logger.debug("[SQL] Usando schema inyectado por orquestador (filtrado)")
    else:
        logger.warning("[SQL] No hay schema_info inyectado. Omitiendo fallback a BD.")
        schema = "Schema no disponible. Usar catálogo semántico y allowed_views."
    return {
        "schema_info": schema,
        "messages": state.get("messages", []) + [AIMessage(content="[SQL] Schema listo.")]
    }


def sql_generate_query(state: SQLAgentState) -> Dict[str, Any]:
    preferred = state.get("preferred_view")
    allowed = state.get("allowed_views", [])
    payload = state.get("payload")

    if not payload:
        return {
            "generated_sql": "",
            "error_message": "No se encontró payload en el estado del subgrafo SQL.",
            "attempts": state.get("attempts", 0) + 1,
            "messages": state.get("messages", []) + [AIMessage(content="[SQL] Error: falta payload")]
        }

    catalogo_detallado = _build_sql_catalog(allowed)

    task_json = json.dumps(payload, indent=2, ensure_ascii=False)
    error_history = state.get("error_message") or "Ninguno"

    system_prompt = _build_sql_system_prompt(catalogo_detallado, task_json, error_history)
    system = SystemMessage(content=system_prompt)

    ctx = f"""
TAREA DEL PAYLOAD:
{task_json}

CATÁLOGO ESTRUCTURADO DE VISTAS (USA SOLO ESTAS COLUMNAS):
{json.dumps(catalogo_detallado, indent=2, ensure_ascii=False)}

VISTA PREFERIDA: {preferred or 'Ninguna'}
VISTAS PERMITIDAS RECOMENDADAS (allowed_views): {allowed}

CONTEXTO SEMÁNTICO DE NEGOCIO:
{state.get('semantic_context', 'No disponible')}

SCHEMA TÉCNICO (solo vistas permitidas):
{state.get('schema_info', 'No cargado')}

ERROR PREVIO (si existe):
{state.get('error_message', 'Ninguno')}
"""
    human = HumanMessage(content=ctx)

    messages = state.get("messages", [])
    if len(messages) > 4:
        messages = messages[-4:]
        logger.debug("[SQL] Historial truncado a 4 mensajes")

    response = LLM.invoke([system] + messages + [human])
    content = response.content

    match = re.search(r"```sql\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
    sql_extracted = match.group(1).strip() if match else content.strip()

    sql_upper = sql_extracted.upper()
    forbidden_cmds = ["DELETE", "DROP", "INSERT", "UPDATE", "TRUNCATE"]
    if any(cmd in sql_upper for cmd in forbidden_cmds):
        return {
            "generated_sql": "",
            "error_message": "Seguridad: comando DML/DDL detectado.",
            "attempts": state.get("attempts", 0) + 1,
            "messages": messages + [response, AIMessage(content="[SQL] Bloqueado por seguridad")]
        }

    if "ERROR_INSALVABLE" in content.upper():
        return {
            "generated_sql": "",
            "error_message": "Ninguna vista permitida resuelve la consulta. Se necesita revisión humana.",
            "attempts": state.get("attempts", 0) + 1,
            "messages": messages + [response, AIMessage(content="[SQL] Insalvable por vistas faltantes")]
        }

    used_views = extract_views_used(sql_extracted)

    if used_views:
        invalid_views = [v for v in used_views if not is_view_allowed(v, _biz_mem)]
        if invalid_views:
            err_msg = (
                f"SECURITY: Vistas no documentadas en AGENTS.md: {invalid_views}. "
                f"Catálogo disponible: {', '.join(_biz_mem.list_views())}"
            )
            logger.warning(f"[SQL] {err_msg}")
            return {
                "generated_sql": "",
                "error_message": err_msg,
                "attempts": state.get("attempts", 0) + 1,
                "messages": messages + [response, AIMessage(content=f"[SQL] {err_msg}")]
            }

        allowed_set = {v.lower().replace("semantic.", "") for v in allowed}
        for v in used_views:
            clean = v.lower().replace("semantic.", "")
            if clean not in allowed_set:
                logger.info(
                    f"[SQL] Vista '{v}' está en el catálogo pero fuera de allowed_views "
                    f"recomendadas. Shortlist: {allowed}"
                )
    else:
        if sql_extracted and re.search(r'\bselect\b', sql_extracted, re.IGNORECASE):
            logger.warning("[SQL] Query SELECT sin vistas semantic. detectadas.")

    is_valid, column_error = _validate_columns_in_sql(sql_extracted)
    if not is_valid:
        logger.warning(f"[SQL] {column_error}")
        return {
            "generated_sql": "",
            "error_message": column_error,
            "attempts": state.get("attempts", 0) + 1,
            "messages": messages + [response, AIMessage(content=f"[SQL] {column_error}")]
        }

    return {
        "generated_sql": sql_extracted,
        "attempts": state.get("attempts", 0) + 1,
        "messages": messages + [response]
    }


def sql_execute_query(state: SQLAgentState) -> Dict[str, Any]:
    sql = state.get("generated_sql", "")
    if not sql:
        return {
            "query_result": None,
            "error_message": state.get("error_message") or "No SQL generado"
        }

    rows, columns, error = execute_sql_query(sql)

    if error:
        return {
            "query_result": None,
            "error_message": error,
            "messages": state.get("messages", []) + [AIMessage(content=f"[SQL] Error DB: {error}")]
        }

    return {
        "query_result": rows,
        "error_message": "",
        "messages": state.get("messages", []) + [AIMessage(content=f"[SQL] Ejecutado. Filas: {len(rows) if rows else 0}")]
    }


def _is_recoverable_db_error(error: str) -> bool:
    if not error:
        return False
    e = error.lower()
    if "column" in e and "does not exist" in e:
        return True
    if any(x in e for x in ["syntax error", "invalid input syntax", "operator does not exist", "ambiguous column"]):
        return True
    if any(x in e for x in ["relation", "undefined_table", "does not exist"]) and "column" not in e:
        return False
    return False


def _enrich_error_with_valid_columns(error: str, sql: str) -> str:
    if not error or "UndefinedColumn" not in error:
        return error
    used_views = extract_views_used(sql)
    enrichments = []
    for view_name in used_views:
        clean = view_name.replace("semantic.", "").strip()
        view_info = _biz_mem.get_view(clean)
        if view_info:
            available = list(view_info.metricas.keys()) + view_info.columnas_fecha
            enrichments.append(f"\nColumnas VÁLIDAS para {view_name}: {available}")
    if enrichments:
        return error + "\n" + "\n".join(enrichments)
    return error


def sql_validate_and_package(state: SQLAgentState) -> Dict[str, Any]:
    rows_raw = state.get("query_result")
    err = state.get("error_message", "")
    sql = state.get("generated_sql", "")
    allowed = state.get("allowed_views", [])
    semantic_ctx = state.get("semantic_context", "")
    preferred = state.get("preferred_view")
    attempts = state.get("attempts", 0)

    columns: List[str] = []
    rows_norm: List[Dict[str, Any]] = []
    warnings: List[str] = []
    status = "success"
    reason = ""
    needs_followup = False

    if err and "UndefinedColumn" in err:
        err = _enrich_error_with_valid_columns(err, sql)

    if "SECURITY" in (err or "") or "Ninguna vista permitida" in (err or "") or "ERROR_INSALVABLE" in (err or ""):
        status = "error"
        needs_followup = True
        reason = f"Violación de política o insalvable: {err}"
        warnings.append(reason)
    elif attempts >= 3 and err:
        status = "error"
        needs_followup = True
        reason = f"Máximo de reintentos alcanzado. Último error: {err}"
        warnings.append(reason)
    elif err:
        status = "error"
        needs_followup = not _is_recoverable_db_error(err)
        reason = f"Error SQL/DB: {err}"
    else:
        if isinstance(rows_raw, list) and len(rows_raw) > 0:
            columns = list(rows_raw[0].keys())
            rows_norm = rows_raw
            reason = "Query válida y con datos"
        else:
            warnings.append("La query ejecutó pero no devolvió filas.")
            reason = "Query ejecutada sin resultados"

    can_answer = bool(rows_norm) and not err and not needs_followup

    contract = SQLContract(
        status=status,
        generated_sql=sql,
        columns=columns,
        rows=rows_norm,
        row_count=len(rows_norm),
        error_message=err or None,
        schema_used=["semantic"],
        allowed_views=allowed,
        preferred_view=preferred,
        semantic_context_used=semantic_ctx[:500] + "..." if len(semantic_ctx) > 500 else semantic_ctx,
        query_confidence=0.95 if status == "success" else 0.0,
        needs_followup=needs_followup,
        reason_for_view_choice=reason,
        can_answer=can_answer,
        reasoning=reason,
        warnings=warnings,
    )

    return {"contract": contract}


def sql_route_retry(state: SQLAgentState) -> Literal["generate_query", "validate_package"]:
    err = state.get("error_message", "")
    attempts = state.get("attempts", 0)

    if not err:
        return "validate_package"
    if attempts >= 3:
        return "validate_package"
    if "SECURITY" in err or "Ninguna vista permitida" in err or "ERROR_INSALVABLE" in err:
        return "validate_package"
    if not _is_recoverable_db_error(err):
        return "validate_package"
    return "generate_query"


sql_builder = StateGraph(SQLAgentState)
sql_builder.add_node("fetch_schema", sql_fetch_schema)
sql_builder.add_node("generate_query", sql_generate_query)
sql_builder.add_node("execute_query", sql_execute_query)
sql_builder.add_node("validate_package", sql_validate_and_package)

sql_builder.add_edge(START, "fetch_schema")
sql_builder.add_edge("fetch_schema", "generate_query")
sql_builder.add_edge("generate_query", "execute_query")
sql_builder.add_conditional_edges(
    "execute_query",
    sql_route_retry,
    {
        "generate_query": "generate_query",
        "validate_package": "validate_package"
    }
)
sql_builder.add_edge("validate_package", END)

SQL_SUBGRAPH = sql_builder.compile()
