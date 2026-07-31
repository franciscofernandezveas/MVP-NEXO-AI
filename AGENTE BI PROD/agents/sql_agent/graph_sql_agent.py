import re
import json
import logging
from typing import Any, Dict, List, Optional, Literal, NotRequired, Tuple
from typing_extensions import TypedDict
from datetime import datetime, timedelta

import sqlparse
from sqlparse.sql import Identifier, IdentifierList
from sqlparse.tokens import Keyword

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from core.llm import LLM
from core.database import execute_sql_query
from core.contracts import SQLContract
from core.sql_utils import extract_views_used
from core.semantic_retriever import (
    column_exists_in_view,
    resolve_column,
    get_view_columns,
)
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


# ------------------------------------------------------------------
# Catálogo y utilidades
# ------------------------------------------------------------------
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


def _detect_date_column(view_name: Optional[str]) -> Optional[str]:
    if not view_name:
        return None
    for col in get_view_columns(view_name):
        if "fecha" in col.lower():
            return col
    return None


def _time_window_to_sql(date_column: str, time_window: Optional[str]) -> Optional[str]:
    if not time_window or not date_column:
        return None

    tw = time_window.lower().strip()
    today = datetime.now().date()

    m = re.match(r"last_(\d+)_days?", tw)
    if m:
        n = int(m.group(1))
        start = (datetime.now().date() - timedelta(days=n)).isoformat()
        return f"{date_column} >= '{start}' AND {date_column} <= '{today}'"

    m = re.match(r"next_(\d+)_days?", tw)
    if m:
        n = int(m.group(1))
        end = (datetime.now().date() + timedelta(days=n)).isoformat()
        return f"{date_column} >= '{today}' AND {date_column} <= '{end}'"

    if tw == "current_month":
        return f"DATE_TRUNC('month', {date_column}) = DATE_TRUNC('month', CURRENT_DATE)"
    if tw == "previous_month":
        return f"DATE_TRUNC('month', {date_column}) = DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')"
    if tw in ("current_year", "ytd"):
        return f"EXTRACT(YEAR FROM {date_column}) = EXTRACT(YEAR FROM CURRENT_DATE)"

    return None


def _strategy_guidance(strategy: str, date_col: Optional[str]) -> str:
    d = date_col or "fecha"
    guides = {
        "single_view": "SELECT dim, metric FROM preferred_view WHERE ... GROUP BY dim",
        "by_branch": f"SELECT nombre_sede, metric FROM preferred_view WHERE ... GROUP BY nombre_sede ORDER BY metric DESC",
        "by_product": f"SELECT producto, metric FROM preferred_view WHERE ... GROUP BY producto ORDER BY metric DESC",
        "monthly": f"SELECT DATE_TRUNC('month', {d}) AS mes, metric FROM preferred_view WHERE ... GROUP BY mes ORDER BY mes",
        "compare_periods": f"WITH current AS (SELECT ... FROM preferred_view WHERE {d} >= periodo_actual), previous AS (...) SELECT ...",
        "historical": f"SELECT {d}, metric FROM preferred_view WHERE {d} >= ... ORDER BY {d}",
    }
    return guides.get(strategy, guides["single_view"])


def _filters_to_sql_description(filters: List[Dict[str, Any]]) -> str:
    if not filters:
        return "Sin filtros estructurados"
    lines = []
    text_cols = {
        "nombre_sede", "sede", "sucursal", "local", "tienda", "plaza",
        "producto", "descripcion", "descripción", "categoria", "categoría"
    }
    for f in filters:
        col = f.get("column", "")
        op = f.get("operator", "=")
        val = f.get("value")
        vt = f.get("value_type", "string")
        if op.upper() == "IN" and isinstance(val, list):
            lines.append(f"{col} IN ({', '.join(repr(v) for v in val)})")
        elif col in text_cols or vt == "string":
            lines.append(f"LOWER({col}) = LOWER('{val}')  -- o: {col} ILIKE '%{val}%'")
        else:
            lines.append(f"{col} {op} {val}")
    return "\n".join(lines)


def _extract_table_aliases(sql: str) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    parsed = sqlparse.parse(sql)
    for stmt in parsed:
        tokens = list(stmt.tokens)
        for idx, token in enumerate(tokens):
            if token.ttype is Keyword and token.value.upper() in ("FROM", "JOIN"):
                nxt_idx = _next_real_token(tokens, idx)
                if nxt_idx is not None:
                    _parse_from_token(tokens[nxt_idx], aliases)
    return aliases


def _next_real_token(tokens, idx):
    for i in range(idx + 1, len(tokens)):
        t = tokens[i]
        if not t.is_whitespace and t.ttype not in (
            sqlparse.tokens.Comment.Single, sqlparse.tokens.Comment.Multiline
        ):
            return i
    return None


def _parse_from_token(token, aliases):
    if isinstance(token, IdentifierList):
        for ident in token.get_identifiers():
            _parse_single_identifier(ident, aliases)
    elif isinstance(token, Identifier):
        _parse_single_identifier(token, aliases)


def _parse_single_identifier(identifier, aliases):
    parts = [t for t in identifier.tokens if not t.is_whitespace and t.value.upper() != "AS"]
    if not parts:
        return
    name = parts[0].value.strip()
    alias = parts[-1].value.strip() if len(parts) > 1 else name
    if name.startswith("semantic."):
        real = name.replace("semantic.", "")
        aliases[alias] = real
        aliases[real] = real


def _validate_columns_in_sql(sql: str) -> Tuple[bool, str]:
    aliases = _extract_table_aliases(sql)
    if not aliases:
        return True, ""

    functions = {
        "sum", "count", "avg", "min", "max", "round", "coalesce", "nullif",
        "date_trunc", "extract", "lower", "upper", "trim", "length"
    }
    invalid: List[str] = []

    parsed = sqlparse.parse(sql)
    for stmt in parsed:
        for token in stmt.flatten():
            if token.ttype is None:
                val = token.value.strip()
                if "." in val:
                    alias, col = val.split(".", 1)
                    alias = alias.strip()
                    col = col.strip()
                else:
                    if len(aliases) == 1:
                        alias = list(aliases.keys())[0]
                        col = val
                    else:
                        continue

                if col.lower() in functions:
                    continue

                view_name = aliases.get(alias)
                if not view_name:
                    continue

                if not column_exists_in_view(f"semantic.{view_name}", col):
                    invalid.append(f"semantic.{view_name}.{col}")

    if invalid:
        return False, f"ERROR_COLUMNAS_INVALIDAS: {', '.join(invalid)}. Usa solo columnas del catálogo semántico."
    return True, ""


def _sql_uses_preferred_view(sql: str, preferred: Optional[str]) -> bool:
    if not preferred:
        return True
    clean = preferred.replace("semantic.", "").strip()
    pattern = re.compile(rf"\bsemantic\.{re.escape(clean)}\b", re.IGNORECASE)
    return bool(pattern.search(sql))


# ------------------------------------------------------------------
# Nodos del grafo SQL
# ------------------------------------------------------------------
def sql_fetch_schema(state: SQLAgentState) -> Dict[str, Any]:
    if state.get("schema_info") and state["schema_info"].strip():
        schema = state["schema_info"]
        logger.debug("[SQL] Usando schema inyectado por orquestador (filtrado)")
    else:
        logger.warning("[SQL] No hay schema_info inyectado.")
        schema = "Schema no disponible. Usar catálogo semántico y allowed_views."
    return {
        "schema_info": schema,
        "messages": state.get("messages", []) + [AIMessage(content="[SQL] Schema listo.")]
    }


def sql_generate_query(state: SQLAgentState) -> Dict[str, Any]:
    payload = state.get("payload", {})

    # PRIORIDAD: payload de la subtarea
    preferred = payload.get("preferred_view") or state.get("preferred_view")
    allowed = payload.get("candidate_views") or state.get("allowed_views", [])

    if not allowed:
        return _sql_error(state, None, "No hay vistas permitidas para esta subtarea.")

    relevant_views = ([preferred] if preferred else []) + [v for v in allowed if v != preferred]
    relevant_views = list(dict.fromkeys(relevant_views))

    catalogo_detallado = _build_sql_catalog(relevant_views)

    # Mapeo de columnas reales
    column_mapping = {}
    if preferred:
        for col in (payload.get("metrics", []) + payload.get("dimensions", [])):
            real = resolve_column(preferred, col)
            column_mapping[col] = real or f"NO_ENCONTRADA:{col}"

    date_col = _detect_date_column(preferred)
    time_window_sql = _time_window_to_sql(date_col, payload.get("time_window")) if date_col else None
    strategy_guidance = _strategy_guidance(payload.get("execution_strategy", "single_view"), date_col)

    system = SystemMessage(content=f"""
Eres un Data Engineer senior experto en PostgreSQL. Debes obedecer ESTRICTAMENTE el plan del Planner.

REGLAS ABSOLUTAS:
1. Usa EXCLUSIVAMENTE vistas bajo el esquema `semantic.`.
2. La vista OBLIGATORIA es: `{preferred or "NINGUNA - elige una de allowed_views"}`. Si no puedes resolver la query con ella, responde `ERROR_INSALVABLE`.
3. Usa SOLO columnas listadas en `metricas` o `columnas_fecha` del catálogo.
4. Mapea cada métrica/dimensión del plan a la `columna_real` indicada en `mapeo_de_columnas`.
5. Para filtros de texto (sede, producto, categoría) usa `ILIKE` o `LOWER(x) = LOWER(y)`. NUNCA `=` directo.
6. La query debe responder EXACTAMENTE a: `{payload.get("task", "")}`.
7. Estrategia de ejecución: `{payload.get("execution_strategy", "single_view")}`. Guía: {strategy_guidance}
8. Ventana temporal: `{payload.get("time_window", "ninguna")}`. Traducción sugerida: {time_window_sql or "N/A"}
9. PROHIBIDO: DELETE, DROP, INSERT, UPDATE, TRUNCATE.
10. Devuelve UNA query SQL SELECT entre ```sql ... ```.

ANTES del SQL, escribe tu plan de traducción:
- Vista elegida: ...
- Columnas disponibles: ...
- Mapeo métricas/dimensiones: ...
- Filtros estructurados a WHERE: ...
- Estrategia aplicada: ...

NO respondas en lenguaje natural fuera del plan de traducción y el bloque SQL.
""")

    human_content = f"""
PAYLOAD DEL PLANNER:
{json.dumps(payload, indent=2, ensure_ascii=False)}

CATÁLOGO DE VISTAS RELEVANTES:
{json.dumps(catalogo_detallado, indent=2, ensure_ascii=False)}

MAPEO DE COLUMNAS:
{json.dumps(column_mapping, indent=2, ensure_ascii=False)}

FILTROS ESTRUCTURADOS A TRADUCIR A WHERE:
{_filters_to_sql_description(payload.get("filters", []))}

VENTANA TEMPORAL:
{payload.get("time_window", "ninguna")} -> {time_window_sql or "N/A"}

PREGUNTA ORIGINAL:
{state.get("question", "")}

ERROR PREVIO:
{state.get("error_message", "Ninguno")}
"""

    messages = state.get("messages", [])
    if len(messages) > 4:
        messages = messages[-4:]

    response = LLM.invoke([system] + messages + [HumanMessage(content=human_content)])
    content = response.content

    match = re.search(r"```sql\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
    sql_extracted = match.group(1).strip() if match else content.strip()

    # 1. Bloqueo DML/DDL
    sql_upper = sql_extracted.upper()
    forbidden_cmds = ["DELETE", "DROP", "INSERT", "UPDATE", "TRUNCATE"]
    if any(cmd in sql_upper for cmd in forbidden_cmds):
        return _sql_error(state, response, "Seguridad: comando DML/DDL detectado.")

    # 2. Insalvable
    if "ERROR_INSALVABLE" in content.upper():
        return _sql_error(state, response, "Ninguna vista permitida resuelve la consulta. Se necesita revisión humana.")

    # 3. Debe usar preferred_view
    if not _sql_uses_preferred_view(sql_extracted, preferred):
        err = f"La query debe usar la vista obligatoria '{preferred}'. SQL generado no la referencia."
        return _sql_error(state, response, err)

    # 4. Vistas documentadas
    used_views = extract_views_used(sql_extracted)
    if used_views:
        invalid_views = [v for v in used_views if not is_view_allowed(v, _biz_mem)]
        if invalid_views:
            err = f"SECURITY: Vistas no documentadas en AGENTS.md: {invalid_views}"
            return _sql_error(state, response, err)
    else:
        if sql_extracted and re.search(r'\bselect\b', sql_extracted, re.IGNORECASE):
            logger.warning("[SQL] Query SELECT sin vistas semantic. detectadas.")

    # 5. Columnas válidas por ámbito
    is_valid, column_error = _validate_columns_in_sql(sql_extracted)
    if not is_valid:
        return _sql_error(state, response, column_error)

    return {
        "generated_sql": sql_extracted,
        "attempts": state.get("attempts", 0) + 1,
        "messages": messages + [response]
    }


def _sql_error(state, response, error_message: str) -> Dict[str, Any]:
    logger.warning(f"[SQL] {error_message}")
    msgs = state.get("messages", [])
    if response is not None:
        msgs = msgs + [response]
    return {
        "generated_sql": "",
        "error_message": error_message,
        "attempts": state.get("attempts", 0) + 1,
        "messages": msgs + [AIMessage(content=f"[SQL] {error_message}")]
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
    payload = state.get("payload", {})
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
        preferred_view=payload.get("preferred_view") or preferred,
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
