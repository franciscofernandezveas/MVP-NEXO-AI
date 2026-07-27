import re
import json
import logging
from typing import Any, Dict, List, Optional, Literal, NotRequired, Tuple
from typing_extensions import TypedDict

import sqlparse

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from core.llm import LLM
from core.database import execute_sql_query, get_semantic_schema_info, get_semantic_schema_for_views
from core.contracts import SQLContract
from core.sql_utils import extract_views_used

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
    Catálogo estructurado de vistas permitidas.
    Incluye métricas/documentación para que el LLM no alucine.
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


def _get_schema_info(allowed_views: List[str]) -> str:
    """
    Obtiene el DDL real del esquema para las vistas permitidas.
    Fallback al schema completo si el filtrado falla o está vacío.
    """
    try:
        schema = get_semantic_schema_for_views(allowed_views)
        if schema and schema.strip():
            return schema
    except Exception as e:
        logger.warning(f"[SQL] Fallo schema filtrado: {e}")
    
    try:
        return get_semantic_schema_info(max_objects=30)
    except Exception as e:
        logger.warning(f"[SQL] Fallo schema completo: {e}")
        return "-- Schema no disponible"


def _validate_sql_security(sql: str) -> Tuple[bool, str]:
    """
    Validación portable: solo seguridad.
    No hardcodea nombres de columnas. El LLM usa el schema real.
    """
    if not sql:
        return False, "SQL vacío"

    sql_upper = sql.upper()

    # Solo SELECT
    if not re.search(r'^\s*SELECT\b', sql, re.IGNORECASE):
        return False, "Solo se permiten consultas SELECT"

    # Bloquear DML/DDL
    forbidden = ["DELETE", "DROP", "INSERT", "UPDATE", "TRUNCATE", "CREATE", "ALTER"]
    for cmd in forbidden:
        if re.search(rf'\b{cmd}\b', sql_upper):
            return False, f"Comando prohibido detectado: {cmd}"

    # Verificar esquema semantic
    used_views = extract_views_used(sql)
    for view in used_views:
        if not view.startswith("semantic."):
            return False, f"Solo se permite el esquema 'semantic'. Vista inválida: {view}"

    return True, ""


def _sanitize_messages(messages: List[Any]) -> List[Any]:
    """
    Elimina respuestas JSON del historial para que el LLM no las imite.
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
    """
    Asegura que el estado tenga schema_info real.
    Si el orquestador no lo inyectó, lo cargamos aquí.
    """
    schema = state.get("schema_info", "")
    allowed_views = state.get("allowed_views", [])
    
    if not schema or not schema.strip():
        logger.warning("[SQL] schema_info vacío. Cargando schema real desde DB.")
        schema = _get_schema_info(allowed_views)
    else:
        logger.debug("[SQL] Usando schema_info inyectado por orquestador.")
    
    return {
        "schema_info": schema,
        "messages": state.get("messages", []) + [AIMessage(content="[SQL] Schema listo.")]
    }


def sql_generate_query(state: SQLAgentState) -> Dict[str, Any]:
    preferred = state.get("preferred_view")
    allowed = state.get("allowed_views", [])
    
    # Catálogo semántico (descripción de vistas)
    catalogo_detallado = _build_sql_catalog(allowed)
    
    # Schema técnico real (DDL con tipos de columnas)
    schema_tecnico = _get_schema_info(allowed)
    if state.get("schema_info") and state["schema_info"].strip():
        schema_tecnico = state["schema_info"]

    system = SystemMessage(content=f"""
Eres un Data Engineer senior experto en PostgreSQL.

CONTEXTO CRÍTICO:
- TÚ SOLO puedes acceder al esquema 'semantic.'.
- PROHIBIDO usar tablas de staging, raw, public o cualquier otro esquema.

REGLAS ABSOLUTAS:
1. Usa ÚNICAMENTE vistas que estén en `allowed_views` o en el catálogo documentado.
2. Toda referencia a tabla/vista DEBE ser: semantic.nombre_vista.
3. Genera UNA query SQL SELECT válida para PostgreSQL.
4. PROHIBIDO: DELETE, DROP, INSERT, UPDATE, TRUNCATE, CREATE, ALTER.
5. Si no puedes generar SQL válido, escribe exactamente: ERROR_INSALVABLE

REGLA DE MAPEO SEMÁNTICO (muy importante):
- Las métricas y dimensiones en el payload son CONCEPTUALES.
- Tú debes encontrar la columna FÍSICA equivalente en la vista elegida usando el SCHEMA TÉCNICO.
- Ejemplos de mapeos comunes:
  * "ventas" puede mapear a: ventas, venta_total, total_ventas, monto, ingreso...
  * "unidades" puede mapear a: unidades, unidades_totales, cantidad...
  * "sede" puede mapear a: sucursal, nombre_sede, local, tienda...
  * "producto" puede mapear a: producto, nombre_producto, descripcion...
  * "fecha" puede mapear a: fecha, fecha_completa, fecha_venta...
- No exijas nombres exactos. Cada vista puede llamar distinto a una misma métrica.
- Si la vista elegida ya expone una métrica agregada (ej: venta_total), NO la sumes manualmente.
- Si la columna es granular diaria (ej: ventas por fila), entonces SÍ usa SUM().

REGLA DE SELECCIÓN DE VISTA:
- Si `preferred_view` está definida y tiene todas las métricas/dimensiones necesarias, úsala.
- Si no, elige la vista más específica de `allowed_views` que tenga todo lo necesario.
- Si ninguna vista lo tiene todo, devuelve ERROR_INSALVABLE.

REGLA DE SALIDA OBLIGATORIA:
- Devuelve ÚNICAMENTE un bloque: ```sql ... ```
- NO devuelvas JSON, explicaciones ni otro texto.
- Si no puedes generar SQL válido, escribe exactamente: ERROR_INSALVABLE

REGLA DE DIMENSIONES DE TEXTO:
- Para columnas de texto como `nombre_sede`, usa `ILIKE` o `LOWER(column) = LOWER('valor')`.
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

CATÁLOGO SEMÁNTICO DE VISTAS:
{json.dumps(catalogo_detallado, indent=2, ensure_ascii=False)}

SCHEMA TÉCNICO REAL (DDL de vistas permitidas):
{schema_tecnico}

VISTA PREFERIDA: {preferred or 'Ninguna'}
VISTAS PERMITIDAS: {allowed}

CONTEXTO SEMÁNTICO DE NEGOCIO:
{state.get('semantic_context', 'No disponible')}

ERROR PREVIO DE EJECUCIÓN (si existe):
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

    # Rechazar JSON explícito
    if content.startswith("{") and content.endswith("}"):
        logger.warning("[SQL] Modelo devolvió JSON en lugar de SQL: %s", content[:300])
        return {
            "generated_sql": "",
            "error_message": "El modelo devolvió JSON en lugar de SQL. Se requiere solo un bloque ```sql ... ```.",
            "attempts": state.get("attempts", 0) + 1,
            "messages": messages + [AIMessage(content="[SQL] Error: respuesta fue JSON, no SQL")]
        }

    # Exigir bloque SQL
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

    # Validación de seguridad portable
    is_valid, security_error = _validate_sql_security(sql_extracted)
    if not is_valid:
        logger.warning(f"[SQL] {security_error}")
        return {
            "generated_sql": "",
            "error_message": security_error,
            "attempts": state.get("attempts", 0) + 1,
            "messages": messages + [response, AIMessage(content=f"[SQL] {security_error}")]
        }

    # Detección de insalvable
    if "ERROR_INSALVABLE" in content.upper():
        return {
            "generated_sql": "",
            "error_message": "Ninguna vista permitida resuelve la consulta. Se necesita revisión humana.",
            "attempts": state.get("attempts", 0) + 1,
            "messages": messages + [response, AIMessage(content="[SQL] Insalvable por vistas faltantes")]
        }

    # Validación de vistas permitidas contra catálogo
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
    Decide si un error de PostgreSQL merece reintento.
    Errores de seguridad/insalvables NO se reintentan.
    Errores de columna/sintaxis SÍ, porque el LLM puede corregir con el schema.
    """
    if not error:
        return False
    e = error.lower()

    if any(x in e for x in [
        "json en lugar de sql",
        "no se encontró bloque sql",
        "insalvable",
        "ninguna vista permitida resuelve",
        "el bloque sql no comienza con select",
        "comando prohibido",
        "solo se permite el esquema",
    ]):
        return False

    if "column" in e and "does not exist" in e:
        return True

    if any(x in e for x in ["syntax error", "invalid input syntax", "operator does not exist", "ambiguous column"]):
        return True

    if any(x in e for x in ["relation", "undefined_table"]) and "column" not in e:
        return False

    return False


def _enrich_error_with_valid_columns(error: str, sql: str) -> str:
    """
    Si el error es de columna inexistente, enriquece con las columnas
    disponibles de la vista afectada. El LLM las usará para corregir.
    """
    if not error:
        return error

    # Solo enriquecer errores de columna
    e_lower = error.lower()
    if not ("does not exist" in e_lower and "column" in e_lower):
        return error

    used_views = extract_views_used(sql)
    enrichments = []

    for view_name in used_views:
        clean = view_name.replace("semantic.", "").strip()
        view_info = _biz_mem.get_view(clean)
        if view_info:
            available = list(view_info.metricas.keys()) + view_info.columnas_fecha
            enrichments.append(f"\nColumnas disponibles en {view_name}: {available}")

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

    # Enriquecer error de columna con columnas disponibles
    if err:
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
