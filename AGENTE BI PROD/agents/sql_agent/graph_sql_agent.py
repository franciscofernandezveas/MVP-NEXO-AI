import re
import json
import logging
from typing import Any, Dict, List, Optional, Literal, NotRequired, Tuple
from typing_extensions import TypedDict
from datetime import datetime, timedelta

import sqlparse
from sqlparse.sql import Identifier, IdentifierList
from sqlparse.tokens import Keyword, Token

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from core.llm import LLM
from core.database import execute_sql_query, get_semantic_schema_for_views
from core.contracts import SQLContract
from core.sql_utils import extract_views_used
from core.harness import is_view_allowed

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
# Extracción de esquema real
# ------------------------------------------------------------------
def _extract_columns_from_ddl(ddl: str) -> List[str]:
    """
    Extrae nombres de columnas reales desde un DDL-like.
    Soporta formato:
      col_name TYPE,
      col_name TYPE NOT NULL,
    """
    if not ddl:
        return []
    cols = []
    # Capturar bloques CREATE TABLE/VIEW ... (
    blocks = re.findall(
        r"CREATE\s+(?:TABLE|VIEW)\s+[\w\.]+\s*\((.*?)\);",
        ddl,
        re.DOTALL | re.IGNORECASE
    )
    for block in blocks:
        for line in block.split("\n"):
            line = line.strip()
            if not line or line.startswith("--"):
                continue
            # Quitar NOT NULL, NULL, DEFAULT...
            line = re.sub(r"\s+(?:NOT\s+NULL|NULL|DEFAULT\s+.*)$", "", line, flags=re.IGNORECASE)
            m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s+\w+", line)
            if m:
                cols.append(m.group(1))
    return cols


def _get_real_columns(schema_info: str, preferred_view: Optional[str]) -> List[str]:
    """Devuelve columnas reales parseadas del esquema."""
    return _extract_columns_from_ddl(schema_info)


# ------------------------------------------------------------------
# Mapeo semántico local (sinónimos comunes)
# ------------------------------------------------------------------
def _normalize(text: str) -> str:
    if not text:
        return ""
    return text.lower().strip().replace("_", " ").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")


_COLUMN_SYNONYMS = {
    "sede": ["nombre_sede", "sucursal", "local", "tienda", "plaza", "sucursales"],
    "nombre_sede": ["sucursal", "local", "tienda", "plaza", "sede"],
    "sucursal": ["nombre_sede", "local", "tienda", "plaza", "sede"],
    "producto": ["descripcion", "descripción", "nombre_producto", "articulo", "artículo", "sku"],
    "descripcion": ["producto", "nombre_producto"],
    "categoria": ["categoria_nueva", "categoría"],
    "fecha": ["fecha_completa", "fecha_venta", "mes", "anio", "año"],
    "venta_total": ["venta_total", "ventas", "ventas_totales", "total_ventas", "subtotal_diario", "ingreso"],
    "unidades": ["cantidad", "unidades_totales", "unidades_vendidas"],
    "transacciones": ["total_transacciones", "numero_transacciones", "transacciones_totales"],
    "ticket_promedio": ["ticket_promedio_sede"],
}


def _resolve_column_local(requested: str, real_columns: List[str]) -> Optional[str]:
    """Mapea un nombre semántico a una columna real."""
    requested_norm = _normalize(requested)
    real_norm = {c: _normalize(c) for c in real_columns}

    # Exacto
    if requested in real_columns:
        return requested

    # Normalizado exacto
    for c, n in real_norm.items():
        if requested_norm == n:
            return c

    # Subcadena
    for c, n in real_norm.items():
        if requested_norm in n or n in requested_norm:
            return c

    # Sinónimos
    synonyms = _COLUMN_SYNONYMS.get(requested_norm, [])
    for syn in synonyms:
        for c, n in real_norm.items():
            if syn == n:
                return c

    return None


def _column_exists_local(column_name: str, real_columns: List[str]) -> bool:
    return _resolve_column_local(column_name, real_columns) is not None


# ------------------------------------------------------------------
# Construcción del catálogo real
# ------------------------------------------------------------------
def _build_real_catalog(preferred_view: Optional[str], schema_info: str) -> Dict[str, Any]:
    catalog: Dict[str, Any] = {}
    if preferred_view:
        real_columns = _get_real_columns(schema_info, preferred_view)
        catalog[preferred_view] = {
            "columnas_reales": real_columns,
            "advertencia": "USA EXACTAMENTE ESTOS NOMBRES DE COLUMNA. Nada más. Nada menos.",
        }
    return catalog


# ------------------------------------------------------------------
# Utilidades de fecha y estrategia
# ------------------------------------------------------------------
def _detect_date_column(real_columns: List[str]) -> Optional[str]:
    for c in real_columns:
        if "fecha" in c.lower():
            return c
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

    m = re.match(r"(\d{4})-(\d{2})-(\d{2})_to_(\d{4})-(\d{2})-(\d{2})", tw)
    if m:
        return f"{date_column} >= '{m.group(1)}-{m.group(2)}-{m.group(3)}' AND {date_column} <= '{m.group(4)}-{m.group(5)}-{m.group(6)}'"

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
        "single_view": "SELECT dimensiones, metricas FROM preferred_view WHERE ... GROUP BY dimensiones",
        "by_branch": f"SELECT nombre_sede, metricas FROM preferred_view WHERE ... GROUP BY nombre_sede ORDER BY metricas DESC",
        "by_product": f"SELECT producto, metricas FROM preferred_view WHERE ... GROUP BY producto ORDER BY metricas DESC",
        "monthly": f"SELECT DATE_TRUNC('month', {d}) AS mes, metricas FROM preferred_view WHERE ... GROUP BY mes ORDER BY mes",
        "compare_periods": f"WITH current AS (SELECT ... FROM preferred_view WHERE {d} >= ...), previous AS (...) SELECT ...",
        "historical": f"SELECT {d}, metricas FROM preferred_view WHERE {d} >= ... ORDER BY {d}",
    }
    return guides.get(strategy, guides["single_view"])


def _translate_filters(filters: List[Dict[str, Any]], real_columns: List[str]) -> str:
    """Traduce filtros estructurados a SQL usando nombres de columna reales."""
    if not filters:
        return ""

    lines = []
    for f in filters:
        col_sem = f.get("column", "")
        col_real = _resolve_column_local(col_sem, real_columns)
        if not col_real:
            continue
        op = f.get("operator", "=")
        val = f.get("value")
        vt = f.get("value_type", "string")

        if op.upper() == "IN" and isinstance(val, list):
            lines.append(f"{col_real} IN ({', '.join(repr(v) for v in val)})")
        elif _is_text_column(col_real):
            lines.append(f"LOWER({col_real}) = LOWER('{val}')")
        else:
            lines.append(f"{col_real} {op} {val}")

    return " AND ".join(lines)


def _is_text_column(col_name: str) -> bool:
    text_patterns = ["sede", "sucursal", "local", "tienda", "plaza", "producto", "descripcion", "categoria", "subcategoria"]
    return any(p in col_name.lower() for p in text_patterns)


def _translate_metrics(metrics: List[str], real_columns: List[str]) -> str:
    """Traduce métricas semánticas a expresiones SQL con columnas reales."""
    exprs = []
    for m in metrics:
        real = _resolve_column_local(m, real_columns)
        if not real:
            continue
        if any(fn in m.lower() for fn in ["venta", "ingreso", "total", "subtotal"]):
            exprs.append(f"SUM({real}) AS total_{real}")
        elif any(fn in m.lower() for fn in ["unidades", "cantidad", "transacciones"]):
            exprs.append(f"SUM({real}) AS total_{real}")
        else:
            exprs.append(f"{real} AS {real}")
    if not exprs:
        return "COUNT(*) AS n"
    return ", ".join(exprs)


def _translate_dimensions(dimensions: List[str], real_columns: List[str]) -> str:
    exprs = []
    for d in dimensions:
        real = _resolve_column_local(d, real_columns)
        if real:
            exprs.append(real)
    return ", ".join(exprs) if exprs else ""


# ------------------------------------------------------------------
# Validación de columnas con sqlparse
# ------------------------------------------------------------------
def _extract_table_aliases(sql: str) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    try:
        parsed = sqlparse.parse(sql)
    except Exception as e:
        logger.warning(f"[SQL] sqlparse falló: {e}")
        return aliases

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
    elif isinstance(token, Token):
        value = token.value.strip()
        if value.startswith("semantic."):
            real = value.replace("semantic.", "")
            aliases[real] = real


def _parse_single_identifier(identifier, aliases):
    if isinstance(identifier, Token) and not isinstance(identifier, Identifier):
        name = identifier.value.strip()
        if name.startswith("semantic."):
            real = name.replace("semantic.", "")
            aliases[real] = real
        return

    try:
        parts = [t for t in identifier.tokens if not t.is_whitespace and t.value.upper() != "AS"]
    except AttributeError:
        name = identifier.value.strip() if hasattr(identifier, "value") else str(identifier)
        if name.startswith("semantic."):
            real = name.replace("semantic.", "")
            aliases[real] = real
        return

    if not parts:
        return
    name = parts[0].value.strip()
    alias = parts[-1].value.strip() if len(parts) > 1 else name
    if name.startswith("semantic."):
        real = name.replace("semantic.", "")
        aliases[alias] = real
        aliases[real] = real


def _validate_columns_in_sql(sql: str, real_columns: List[str]) -> Tuple[bool, str]:
    try:
        aliases = _extract_table_aliases(sql)
    except Exception as e:
        logger.warning(f"[SQL] No se pudo parsear aliases: {e}. Saltando validación.")
        return True, ""

    if not aliases:
        return True, ""

    functions = {
        "sum", "count", "avg", "min", "max", "round", "coalesce", "nullif",
        "date_trunc", "extract", "lower", "upper", "trim", "length"
    }
    invalid: List[str] = []

    try:
        parsed = sqlparse.parse(sql)
    except Exception as e:
        logger.warning(f"[SQL] No se pudo parsear SQL: {e}. Saltando validación.")
        return True, ""

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

                if col not in real_columns:
                    invalid.append(f"semantic.{view_name}.{col}")

    if invalid:
        return False, f"ERROR_COLUMNAS_INVALIDAS: {', '.join(invalid)}. Columnas VÁLIDAS: {real_columns}"
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
    payload = state.get("payload", {})
    preferred = payload.get("preferred_view") or state.get("preferred_view")

    # Si no hay schema_info inyectado, obtenerlo de la BD
    schema = state.get("schema_info", "")
    if not schema.strip() and preferred:
        try:
            schema = get_semantic_schema_for_views([preferred])
        except Exception as e:
            logger.warning(f"[SQL] No se pudo obtener schema real de {preferred}: {e}")

    if not schema.strip():
        logger.warning("[SQL] Schema real no disponible. Usar allowed_views y semantic_context.")
        schema = "-- Schema real no disponible"

    return {
        "schema_info": schema,
        "messages": state.get("messages", []) + [AIMessage(content="[SQL] Schema real listo.")]
    }


def sql_generate_query(state: SQLAgentState) -> Dict[str, Any]:
    payload = state.get("payload", {})
    preferred = payload.get("preferred_view") or state.get("preferred_view")
    allowed = payload.get("candidate_views") or state.get("allowed_views", [])

    if not allowed:
        return _sql_error(state, None, "No hay vistas permitidas para esta subtarea.")

    if not preferred:
        return _sql_error(state, None, "No hay preferred_view definida para esta subtarea.")

    # 1. Obtener columnas reales del schema_info
    schema_info = state.get("schema_info", "")
    real_columns = _get_real_columns(schema_info, preferred)

    if not real_columns:
        logger.warning(f"[SQL] No se pudieron extraer columnas reales de {preferred}. Schema: {schema_info[:200]}")
        # Fallback: usar allowed_views
        real_columns = []

    # 2. Mapeo de requerimientos a columnas reales
    column_mapping = {}
    unresolved = []
    for col in (payload.get("metrics", []) + payload.get("dimensions", [])):
        real = _resolve_column_local(col, real_columns)
        column_mapping[col] = real or f"NO_ENCONTRADA:{col}"
        if not real:
            unresolved.append(col)

    for f in (payload.get("filters", []) or []):
        real = _resolve_column_local(f.get("column", ""), real_columns)
        if real:
            column_mapping[f.get("column")] = real
        else:
            unresolved.append(f.get("column"))

    if unresolved:
        err = f"ERROR_INSALVABLE: Columnas no encontradas en {preferred}: {unresolved}. Columnas disponibles: {real_columns}"
        return _sql_error(state, None, err)

    # 3. Detectar fecha real
    date_col = _detect_date_column(real_columns)

    # 4. Pre-traducir para el prompt
    dims_expr = _translate_dimensions(payload.get("dimensions", []), real_columns)
    metrics_expr = _translate_metrics(payload.get("metrics", []), real_columns)
    filters_expr = _translate_filters(payload.get("filters", []), real_columns)
    time_window_sql = _time_window_to_sql(date_col, payload.get("time_window")) if date_col else None
    strategy_guidance = _strategy_guidance(payload.get("execution_strategy", "single_view"), date_col)

    catalogo_real = _build_real_catalog(preferred, schema_info)

    system = SystemMessage(content=f"""
Eres un Data Engineer senior experto en PostgreSQL. Debes generar SQL usando ÚNICAMENTE el esquema real proporcionado.

REGLAS ABSOLUTAS:
1. Usa EXCLUSIVAMENTE vistas bajo el esquema `semantic.`.
2. La vista OBLIGATORIA es: `{preferred}`. Toda referencia a tabla DEBE ser `{preferred}`.
3. Usa ÚNICAMENTE columnas de la lista `columnas_reales` del catálogo. NUNCA inventes nombres.
4. Mapeo de columnas reales: usa los nombres de la columna REAL, no los semánticos del plan.
5. Para columnas de texto (sede/sucursal/local/tienda/plaza/producto/categoria) usa `ILIKE` o `LOWER(x) = LOWER(y)`. NUNCA `=` directo.
6. PROHIBIDO: DELETE, DROP, INSERT, UPDATE, TRUNCATE.
7. Devuelve UNA query SQL SELECT entre ```sql ... ```.

ANTES del SQL, escribe tu plan de traducción:
- Vista elegida: ...
- Columnas disponibles: ...
- Mapeo requerimientos → columnas reales: ...
- SELECT proyectado: ...
- WHERE proyectado: ...
- Estrategia aplicada: ...
""")

    human_content = f"""
TAREA DEL PLANNER:
{json.dumps(payload, indent=2, ensure_ascii=False)}

ESQUEMA REAL DE LA VISTA:
{json.dumps(catalogo_real, indent=2, ensure_ascii=False)}

MAPEO REQUERIMIENTO → COLUMNA REAL:
{json.dumps(column_mapping, indent=2, ensure_ascii=False)}

SUGERENCIAS DE TRADUCCIÓN:
- SELECT dimensiones: {dims_expr or 'N/A'}
- SELECT métricas: {metrics_expr or 'N/A'}
- WHERE filtros: {filters_expr or 'N/A'}
- Ventana temporal ({payload.get('time_window', 'ninguna')}): {time_window_sql or 'N/A'}
- Estrategia: {payload.get('execution_strategy', 'single_view')} → {strategy_guidance}

PREGUNTA TÉCNICA DE LA SUBTAREA:
{state.get('question', '')}

ERROR PREVIO:
{state.get('error_message', 'Ninguno')}
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

    # 4. Columnas válidas contra esquema real
    is_valid, column_error = _validate_columns_in_sql(sql_extracted, real_columns)
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
        try:
            ddl = get_semantic_schema_for_views([view_name])
            cols = _extract_columns_from_ddl(ddl)
            enrichments.append(f"\nColumnas VÁLIDAS para {view_name}: {cols}")
        except Exception as e:
            logger.warning(f"[SQL] No se pudo enriquecer error con columnas de {view_name}: {e}")
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
