import re
import json
import logging
from typing import Any, Dict, List, Optional, Literal, NotRequired, Tuple, Set
from typing_extensions import TypedDict

import sqlparse
from sqlparse import tokens as T

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from core.llm import LLM
from core.database import execute_sql_query
from core.contracts import SQLContract
from core.sql_utils import extract_views_used

# ============================================================
# NUEVO: Seguridad basada en catálogo semántico (AGENTS.md)
# ============================================================
from core.harness import BusinessMemory, is_view_allowed

# Cargar catálogo semántico una sola vez al importar el módulo
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
    """Normaliza texto para comparación flexible de columnas."""
    if not text:
        return ""
    return (
        text.lower()
        .strip()
        .replace("_", " ")
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
    )


def _build_sql_catalog(allowed_views: List[str]) -> Dict[str, Any]:
    """
    Construye un catálogo estructurado de vistas permitidas con sus columnas.
    Incluye métricas y dimensiones/documentación para que el agente no alucine columnas.
    """
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
    """
    Verifica si una columna existe en la vista (métrica o columna de fecha/documentada).
    Soporta coincidencia flexible.
    """
    clean_view = view_name.replace("semantic.", "").strip()
    view_info = _biz_mem.get_view(clean_view)
    if not view_info:
        return False

    requested = _normalize(column_name)
    if not requested:
        return False

    # Métricas y columnas documentadas
    available_cols: List[str] = list(view_info.metricas.keys()) + view_info.columnas_fecha
    available_normalized = [_normalize(c) for c in available_cols]

    # Coincidencia exacta o subcadura
    for avail in available_normalized:
        if requested == avail or requested in avail or avail in requested:
            return True

    # Mapeos semánticos comunes
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
            if v == avail or v in avail or avail in v:
                return True

    return False


# ============================================================
# NUEVO: Extractor real de columnas con sqlparse
# ============================================================
SQL_RESERVED_WORDS = {
    "select", "from", "where", "group", "by", "order", "having", "join",
    "inner", "left", "right", "full", "outer", "on", "and", "or", "not",
    "in", "is", "null", "between", "like", "ilike", "limit", "offset",
    "as", "desc", "asc", "distinct", "all", "union", "except", "intersect",
    "cast", "case", "when", "then", "else", "end", "sum", "count", "avg",
    "min", "max", "over", "partition", "row_number", "rank", "dense_rank",
    "true", "false", "date", "interval", "extract", "to_char", "now", "current_date",
    # Palabras en español que aparecían como falsos positivos
    "el", "la", "los", "las", "un", "una", "del", "al", "por", "para",
    "con", "sin", "sobre", "entre", "desde", "hasta", "cada", "cual",
    "consulta", "query", "columnas", "columna", "rows", "filas", "vista",
    "tabla", "contiene", "contienen", "todas", "todos", "requeridas", "requerido",
    "elegida", "elegido", "utilizando", "usando", "porque", "razon", "razón",
    "error_message", "reasoning", "reason_for_view_choice", "allowed_views",
    "preferred_view", "semantic_context_used", "query_confidence", "schema_used",
    "can_answer", "needs_followup", "warnings", "junio", "julio", "agosto",
    "diario", "semanal", "mensual", "anual", "desagregado", "total", "ventas",
    "total_ventas", "calcula", "agrupando", "sumando", "rango", "fechas",
    "especificado", "especificada", "especificados", "especificadas", "que",
    "la", "del", "al",
}


def _extract_sql_identifiers(sql: str) -> List[str]:
    """
    Extrae identificadores reales de un SQL usando sqlparse.
    Ignora palabras reservadas, funciones, literales y alias comunes.
    """
    identifiers = []
    parsed = sqlparse.parse(sql)
    for statement in parsed:
        for token in statement.flatten():
            if token.is_whitespace:
                continue
            if token.ttype in (T.String.Single, T.String.Symbol, T.Number, T.Number.Integer, T.Number.Float):
                continue
            if token.ttype in T.Comment:
                continue
            value = token.value.strip('"').strip("'").strip()
            normalized = _normalize(value)
            if not normalized:
                continue
            if normalized in SQL_RESERVED_WORDS:
                continue
            # Ignorar funciones agregadas conocidas (ya están en reserved)
            if normalized in SQL_RESERVED_WORDS:
                continue
            identifiers.append(normalized)
    return identifiers


def _validate_columns_in_sql(sql: str) -> Tuple[bool, str]:
    """
    Valida que todas las columnas usadas en el SQL existan en las vistas documentadas.
    Devuelve (is_valid, error_message).
    """
    used_views = extract_views_used(sql)
    if not used_views:
        return True, ""

    all_identifiers = _extract_sql_identifiers(sql)

    for view_name in used_views:
        invalid_cols = []
        for col in all_identifiers:
            if not _column_exists_in_view(view_name, col):
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


def _sanitize_messages(messages: List[Any]) -> List[Any]:
    """
    Elimina/reemplaza respuestas anteriores que sean JSON puro,
    para evitar que el modelo las imite en el siguiente intento.
    """
    cleaned = []
    for m in messages:
        if isinstance(m, AIMessage) and isinstance(getattr(m, "content", ""), str):
            content = m.content.strip()
            if content.startswith("{") and content.endswith("}"):
                cleaned.append(
                    AIMessage(
                        content="[Respuesta previa inválida: se devolvió JSON en lugar de SQL. Recordar: solo bloque ```sql ... ```]"
                    )
                )
                continue
        cleaned.append(m)
    return cleaned


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

    catalogo_detallado = _build_sql_catalog(allowed)

    system = SystemMessage(content=f"""
Eres un Data Engineer senior experto en PostgreSQL.

CONTEXTO CRÍTICO:
- TÚ SOLO puedes acceder al esquema 'semantic.'.
- PROHIBIDO usar tablas de staging, raw, public o cualquier otro esquema.

REGLAS ABSOLUTAS:
1. Antes de escribir CUALQUIER query, consulta el CATÁLOGO ESTRUCTURADO DE VISTAS.
2. SOLO puedes usar columnas que aparezcan en 'metricas' o 'columnas_fecha' de la vista seleccionada.
3. SI una columna que necesitas NO está en el catálogo, NO la inventes. Escribe exactamente: ERROR_INSALVABLE
4. Usa ÚNICAMENTE vistas que estén en `allowed_views` o en el catálogo documentado.
5. Si `preferred_view` existe y tiene todas las columnas necesarias, ÚSALA.
6. Toda referencia a tabla DEBE ser: semantic.nombre_vista.
7. Genera UNA query SQL SELECT válida para PostgreSQL.
8. PROHIBIDO: DELETE, DROP, INSERT, UPDATE, TRUNCATE.
9. Si hay un error previo de ejecución, CORRÍGELO respetando las columnas VÁLIDAS del catálogo.

REGLA DE SALIDA OBLIGATORIA (LA MÁS IMPORTANTE):
- Devuelve ÚNICAMENTE un bloque de código SQL: ```sql ... ```
- NO devuelvas JSON, explicaciones, listas, ni ningún otro texto fuera del bloque SQL.
- Si no puedes generar SQL válido, escribe exactamente: ERROR_INSALVABLE

REGLA DE INTEGRIDAD DE COLUMNAS:
- Antes de usar una columna en SELECT, WHERE, GROUP BY, ORDER BY u ON:
  a) Verifica su nombre exacto en el catálogo.
  b) Si no existe, NO uses una columna "parecida". Devuelve ERROR_INSALVABLE.

REGLA DE DIMENSIONES DE TEXTO:
- Para columnas de texto como `nombre_sede`, usa `ILIKE` o `LOWER(column) = LOWER('valor')`.

NO respondas en lenguaje natural. Solo SQL o la marca ERROR_INSALVABLE.
""")

    payload = state.get("payload")
    if not payload:
        return {
            "generated_sql": "",
            "error_message": "No se encontró payload en el estado del subgrafo SQL.",
            "attempts": state.get("attempts", 0) + 1,
            "messages": state.get("messages", []) + [AIMessage(content="[SQL] Error: falta payload")]
        }

    ctx = f"""
TAREA DEL PAYLOAD:
{json.dumps(payload, indent=2, ensure_ascii=False)}

CATÁLOGO ESTRUCTURADO DE VISTAS (USA SOLO ESTAS COLUMNAS):
{json.dumps(catalogo_detallado, indent=2, ensure_ascii=False)}

VISTA PREFERIDA: {preferred or 'Ninguna'}
VISTAS PERMITIDAS: {allowed}

CONTEXTO SEMÁNTICO:
{state.get('semantic_context', 'No disponible')}

SCHEMA TÉCNICO:
{state.get('schema_info', 'No cargado')}

ERROR PREVIO:
{state.get('error_message', 'Ninguno')}
"""
    human = HumanMessage(content=ctx)

    # Sanitizar historial y truncar
    messages = _sanitize_messages(state.get("messages", []))
    if len(messages) > 4:
        messages = messages[-4:]
        logger.debug("[SQL] Historial truncado a 4 mensajes")

    response = LLM.invoke([system] + messages + [human])
    content = response.content.strip()

    # ============================================================
    # RECHAZAR JSON EXPLÍCITO
    # ============================================================
    if content.startswith("{") and content.endswith("}"):
        logger.warning("[SQL] Modelo devolvió JSON en lugar de SQL: %s", content[:300])
        return {
            "generated_sql": "",
            "error_message": "El modelo devolvió JSON en lugar de SQL. Se requiere solo un bloque ```sql ... ```.",
            "attempts": state.get("attempts", 0) + 1,
            "messages": messages + [AIMessage(content="[SQL] Error: respuesta fue JSON, no SQL")]
        }

    # ============================================================
    # EXIGIR BLOQUE SQL EXPLÍCITO
    # ============================================================
    match = re.search(r"```sql\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
    if not match:
        logger.warning("[SQL] No se encontró bloque SQL en la respuesta: %s", content[:300])
        return {
            "generated_sql": "",
            "error_message": "No se encontró bloque SQL ```sql ... ``` en la respuesta del modelo.",
            "attempts": state.get("attempts", 0) + 1,
            "messages": messages + [AIMessage(content="[SQL] Error: no se encontró bloque SQL")]
        }

    sql_extracted = match.group(1).strip()

    # Asegurar que sea un SELECT
    if not re.search(r'^\s*SELECT\b', sql_extracted, re.IGNORECASE):
        return {
            "generated_sql": "",
            "error_message": f"El bloque SQL no comienza con SELECT: {sql_extracted[:200]}",
            "attempts": state.get("attempts", 0) + 1,
            "messages": messages + [AIMessage(content="[SQL] Error: bloque no es SELECT")]
        }

    # Bloqueo DML/DDL
    sql_upper = sql_extracted.upper()
    forbidden_cmds = ["DELETE", "DROP", "INSERT", "UPDATE", "TRUNCATE"]
    if any(re.search(rf'\b{cmd}\b', sql_upper) for cmd in forbidden_cmds):
        return {
            "generated_sql": "",
            "error_message": "Seguridad: comando DML/DDL detectado.",
            "attempts": state.get("attempts", 0) + 1,
            "messages": messages + [response, AIMessage(content="[SQL] Bloqueado por seguridad")]
        }

    # Detección de insalvable
    if "ERROR_INSALVABLE" in content.upper():
        return {
            "generated_sql": "",
            "error_message": "Ninguna vista permitida resuelve la consulta. Se necesita revisión humana.",
            "attempts": state.get("attempts", 0) + 1,
            "messages": messages + [response, AIMessage(content="[SQL] Insalvable por vistas faltantes")]
        }

    # Validación de vistas permitidas
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

    # ============================================================
    # VALIDACIÓN DE COLUMNAS CON sqlparse
    # ============================================================
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
    """
    Determina si un error de DB justifica un reintento.
    Errores de formato del modelo o insalvables NO se reintentan.
    """
    if not error:
        return False
    e = error.lower()

    # NO recuperable: el modelo devolvió JSON, no encontró bloque SQL, o es insalvable
    if any(x in e for x in [
        "json en lugar de sql",
        "no se encontró bloque sql",
        "insalvable",
        "ninguna vista permitida resuelve",
        "el bloque sql no comienza con select",
    ]):
        return False

    # Errores de columna: recoverable si el prompt tiene catálogo
    if "column" in e and "does not exist" in e:
        return True

    # Errores de sintaxis u operadores: recoverable
    if any(x in e for x in ["syntax error", "invalid input syntax", "operator does not exist", "ambiguous column"]):
        return True

    # Errores de tabla/vista: no recoverable
    if any(x in e for x in ["relation", "undefined_table"]) and "column" not in e:
        return False

    return False


def _enrich_error_with_valid_columns(error: str, sql: str) -> str:
    """
    Si el error es UndefinedColumn, enriquece el mensaje con las columnas válidas
    de la vista afectada para que el reintento tenga contexto claro.
    """
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

    # Enriquecer error de columna con columnas válidas para el reintento
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
    if "El modelo devolvió JSON" in err or "No se encontró bloque SQL" in err:
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
