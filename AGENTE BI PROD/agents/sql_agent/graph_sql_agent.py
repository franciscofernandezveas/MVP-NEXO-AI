# core/graph_sql_agent.py
# -------------------------------------------------
# Versión alineada a la arquitectura (sin semantic_map):
#
#  - Validación de columnas = match EXACTO normalizado contra BusinessMemory.
#    No hay sinónimos ni fuzzy: si la columna no existe, se devuelve un error
#    estructurado (COLUMNAS_INVALIDAS) con la lista de columnas válidas, y el
#    reintento deja que el LLM se autocorrija con esa información.
#    La resolución semántica (término → columna) vive en retriever/planner.
#
# Cambios heredados P0/P1:
#  B3  Intento previo visible en el prompt (corrección accionable del supervisor)
#  C1  error_context ya no se dispara con error vacío en el primer intento
#  D1  Filtro DML con word boundaries ('updated_at' ya no bloquea)
#  D2  Extracción de columnas por AST (sqlglot, fail-open) — EXTRACT/ISODOW/… ok
#  P0  Contrato: views_used, attempts, columnas del driver incluso con 0 filas
#  P1  status="no_data" para 0 filas exitosas (corta el loop de re-ejecución)
#
# Compañeros necesarios:
#  - contracts.py: SQLContract.status incluye "no_data"; campos views_used, attempts
#  - orchestrator wrapper: inyecta previous_sql/error/row_count; acepta "no_data"
#  - graph_supervisor: status_ok incluye "no_data"
# -------------------------------------------------
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

# D2: extracción estructural de columnas (AST). Si falta, fallback léxico.
try:
    import sqlglot
    from sqlglot import exp as sqlglot_exp
    _HAS_SQLGLOT = True
except ImportError:
    _HAS_SQLGLOT = False

_biz_mem = BusinessMemory.from_file()

logger = logging.getLogger("bi_orchestrator")

if not _HAS_SQLGLOT:
    logger.warning("[SQL] sqlglot no instalado (pip install sqlglot). Fallback léxico activo.")

# D1: comandos prohibidos con word boundaries ("updated_at" ya no es falso positivo)
_FORBIDDEN_RE = re.compile(r"\b(?:DELETE|DROP|INSERT|UPDATE|TRUNCATE)\b")

# Marcadores de error con semántica de routing distinta:
#  - COLUMNAS_INVALIDAS   → recuperable (reintento con catálogo en contexto)
#  - ERROR_INSALVABLE     → terminal (lo declaró el propio LLM)
#  - SECURITY:            → terminal (violación de allowlist)
COL_ERR_PREFIX = "COLUMNAS_INVALIDAS"


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
    supervisor_instruction: NotRequired[Optional[str]]

    # B3: intento previo inyectado por el wrapper del orquestador
    previous_sql: NotRequired[str]
    previous_error: NotRequired[Optional[str]]
    previous_row_count: NotRequired[int]

    # P0: columnas devueltas por el driver (útiles incluso con 0 filas)
    query_columns: NotRequired[List[str]]


# ------------------------------------------------------------------
# Helpers genéricos
# ------------------------------------------------------------------
def _normalize(text: str) -> str:
    """Higiene de strings (case/acentos/underscores). NO es un mapa semántico."""
    if not text:
        return ""
    return (
        text.lower().strip().replace("_", " ")
        .replace("á", "a").replace("é", "e").replace("í", "i")
        .replace("ó", "o").replace("ú", "u")
    )


def _get(obj: Any, attr: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _build_sql_catalog(allowed_views: List[str]) -> Dict[str, Any]:
    catalog: Dict[str, Any] = {}
    for view_full_name in allowed_views:
        view_info = _biz_mem.get_view(view_full_name.replace("semantic.", "").strip())
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


# ------------------------------------------------------------------
# Validación de columnas: EXACTA contra el catálogo (cortafuegos, no resolver)
# ------------------------------------------------------------------
def _column_exists_in_view(view_name: str, column_name: str) -> bool:
    """Match exacto normalizado. La semántica la resolvió planner/retriever."""
    view_info = _biz_mem.get_view(view_name.replace("semantic.", "").strip())
    if not view_info:
        return False
    available = {_normalize(k) for k in view_info.metricas.keys()}
    available |= {_normalize(c) for c in view_info.columnas_fecha}
    return _normalize(column_name) in available


_SQL_KEYWORDS_LEGACY = {
    # Fallback si no hay sqlglot: keywords + funciones frecuentes de PostgreSQL
    "as", "by", "on", "and", "or", "not", "in", "is", "like", "ilike", "between",
    "sum", "count", "avg", "min", "max", "round", "coalesce", "nullif", "cast",
    "case", "when", "then", "else", "end", "distinct", "null", "true", "false",
    "extract", "isodow", "dow", "epoch", "date", "date_trunc", "interval",
    "current_date", "current_timestamp", "day", "days", "month", "months",
    "year", "years", "hour", "hours", "week", "weeks",
    "limit", "offset", "order", "group", "having", "asc", "desc",
    "join", "left", "right", "inner", "outer", "where", "from", "select",
    "over", "partition", "concat", "abs", "ceil", "floor",
}


def _extract_referenced_columns(sql: str) -> List[str]:
    """Referencias de columna reales. AST excluye funciones/keywords/literales:
    EXTRACT(ISODOW FROM fecha) → ['fecha']; ya no bloquea las queries de horas."""
    if _HAS_SQLGLOT:
        try:
            tree = sqlglot.parse_one(sql, read="postgres")
            return sorted({c.name for c in tree.find_all(sqlglot_exp.Column) if c.name})
        except Exception as e:
            logger.warning(f"[SQL] sqlglot no parseó; validación fail-open: {e}")
            return []
    cols = extract_columns_used(sql)
    return [c for c in cols if c.lower() not in _SQL_KEYWORDS_LEGACY]


def _validate_columns_in_sql(sql: str) -> Tuple[bool, str]:
    """
    Cortafuegos exacto. El error es ACCIONABLE (incluye columnas válidas)
    y su marcador COLUMNAS_INVALIDAS lo hace recuperable en el router.
    """
    used_views = extract_views_used(sql)
    if not used_views:
        return True, ""

    cols = _extract_referenced_columns(sql)
    if not cols:
        return True, ""

    for view_name in used_views:
        invalid_cols = [c for c in cols if not _column_exists_in_view(view_name, c)]
        if invalid_cols:
            view_info = _biz_mem.get_view(view_name.replace("semantic.", "").strip())
            available = (
                list(view_info.metricas.keys()) + view_info.columnas_fecha
                if view_info else []
            )
            return False, (
                f"{COL_ERR_PREFIX}: La vista '{view_name}' no contiene las columnas "
                f"{invalid_cols}. Columnas VÁLIDAS: {available}. "
                f"Reescribe la query usando EXACTAMENTE esos nombres."
            )

    return True, ""


# ------------------------------------------------------------------
# Nodos del subgrafo
# ------------------------------------------------------------------
def sql_fetch_schema(state: SQLAgentState, **kwargs) -> Dict[str, Any]:
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


def sql_generate_query(state: SQLAgentState, **kwargs) -> Dict[str, Any]:
    preferred = state.get("preferred_view")
    allowed = state.get("allowed_views", [])
    instruction = state.get("supervisor_instruction")

    catalogo_detallado = _build_sql_catalog(allowed)

    supervisor_section = ""
    if instruction:
        supervisor_section = f"""
{'=' * 60}
INSTRUCCIÓN DIRECTA DEL SUPERVISOR:
{instruction}
{'=' * 60}
"""

    # C1: "" != "Ninguno" disparaba este bloque en el PRIMER intento
    error_context = state.get("error_message", "") or "Ninguno"
    if error_context != "Ninguno" and instruction:
        error_context = (
            f"ERROR PREVIO: {error_context}\n\n"
            f"CORRECCIÓN SOLICITADA POR EL SUPERVISOR:\n{instruction}\n\n"
            f"Corrige el SQL usando EXACTAMENTE las columnas VÁLIDAS del catálogo. "
            f"NO inventes columnas ni uses nombres 'parecidos'."
        )

    # B3: la corrección es accionable — el agente VE su intento anterior
    previous_block = ""
    prev_sql = (state.get("previous_sql") or "").strip()
    if prev_sql:
        previous_block = (
            "\nINTENTO ANTERIOR — NO LO REPITAS:\n"
            f"```sql\n{prev_sql}\n```\n"
            f"Resultado obtenido: {state.get('previous_row_count', 0)} filas | "
            f"Error previo: {state.get('previous_error') or 'ninguno'}\n"
            "Debes CAMBIAR el enfoque: otros filtros, otra vista, otra granularidad "
            "u otro rango de fechas.\n"
        )

    system = SystemMessage(content=f"""
Eres un Data Engineer senior experto en PostgreSQL.

CONTEXTO CRÍTICO:
- TÚ SOLO puedes acceder al esquema 'semantic.'.
- PROHIBIDO usar tablas de staging, raw, public o cualquier otro esquema.

REGLAS ABSOLUTAS:
1. Antes de escribir CUALQUIER query, consulta el CATÁLOGO ESTRUCTURADO DE VISTAS.
2. SOLO puedes usar columnas que aparezcan en 'metricas' o 'columnas_fecha' de la vista seleccionada.
3. Usa el nombre EXACTO de la columna tal como aparece en el catálogo. NO existen sinónimos ni
   columnas "equivalentes": si el nombre no está en el catálogo, la columna no existe.
4. Si tras revisar el catálogo NINGUNA vista permitida resuelve la tarea, responde ERROR_INSALVABLE.
5. Usa ÚNICAMENTE vistas que estén en `allowed_views` o en el catálogo documentado.
6. Si `preferred_view` existe y tiene todas las columnas necesarias, ÚSALA.
7. Toda referencia a tabla DEBE ser: semantic.nombre_vista.
8. Genera UNA query SQL SELECT válida para PostgreSQL, en un bloque ```sql ... ```.
9. PROHIBIDO: DELETE, DROP, INSERT, UPDATE, TRUNCATE.
10. Si hay un error previo o intento anterior, CORRÍGELO usando las columnas VÁLIDAS indicadas.
    NO repitas la misma query.
11. La query debe responder EXACTAMENTE a la tarea del payload.

REGLA DE DIMENSIONES DE TEXTO:
- Para columnas de texto (sedes, productos, categorías), NUNCA uses `=` directo.
- Usa siempre `ILIKE` o `LOWER(column) = LOWER('valor')`.
- Ejemplo: `WHERE nombre_sede ILIKE 'merced'`.

NO respondas en lenguaje natural. Solo SQL o la marca ERROR_INSALVABLE.
{supervisor_section}
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
{previous_block}
CATÁLOGO ESTRUCTURADO DE VISTAS (USA SOLO ESTAS COLUMNAS):
{json.dumps(catalogo_detallado, indent=2, ensure_ascii=False)}

VISTA PREFERIDA: {preferred or 'Ninguna'}
VISTAS PERMITIDAS (allowed_views): {allowed}

CONTEXTO SEMÁNTICO DE NEGOCIO:
{state.get('semantic_context', 'No disponible')}

SCHEMA TÉCNICO (solo vistas permitidas):
{state.get('schema_info', 'No cargado')}

ERROR PREVIO / INSTRUCCIÓN DE CORRECCIÓN:
{error_context}
"""
    human = HumanMessage(content=ctx)

    # El contexto siempre se reconstruye desde el estado; el historial
    # truncado aporta solo la respuesta previa del LLM y los errores.
    messages = state.get("messages", [])
    if len(messages) > 4:
        messages = messages[-4:]
        logger.debug("[SQL] Historial truncado a 4 mensajes")

    response = LLM.invoke([system] + messages + [human])
    content = response.content

    match = re.search(r"```sql\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
    sql_extracted = match.group(1).strip() if match else content.strip()

    # --- Seguridad DML/DDL (D1: word boundaries, recuperable con feedback) ---
    if _FORBIDDEN_RE.search(sql_extracted.upper()):
        return {
            "generated_sql": "",
            "error_message": "Seguridad: comando DML/DDL detectado. Genera SOLO SELECT.",
            "attempts": state.get("attempts", 0) + 1,
            "messages": messages + [response, AIMessage(content="[SQL] Bloqueado por seguridad")]
        }

    # --- Insalvabilidad declarada por el propio LLM (terminal) ---
    if "ERROR_INSALVABLE" in content.upper():
        return {
            "generated_sql": "",
            "error_message": "Ninguna vista permitida resuelve la consulta. Se necesita revisión humana.",
            "attempts": state.get("attempts", 0) + 1,
            "messages": messages + [response, AIMessage(content="[SQL] Insalvable por vistas faltantes")]
        }

    used_views = extract_views_used(sql_extracted)

    if used_views:
        # --- Allowlist dura: vistas no documentadas (terminal por diseño) ---
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

        # Vista documentada pero fuera del shortlist del planner: se audita, no se bloquea
        allowed_set = {v.lower().replace("semantic.", "") for v in allowed}
        for v in used_views:
            if v.lower().replace("semantic.", "") not in allowed_set:
                logger.info(
                    f"[SQL] Vista '{v}' documentada pero fuera de allowed_views. "
                    f"Shortlist: {allowed}. Queda trazada en contract.views_used."
                )
    else:
        if sql_extracted and re.search(r'\bselect\b', sql_extracted, re.IGNORECASE):
            logger.warning("[SQL] Query SELECT sin vistas semantic. detectadas.")

    # --- Cortafuegos de columnas (exacto; error accionable y recuperable) ---
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


def sql_execute_query(state: SQLAgentState, **kwargs) -> Dict[str, Any]:
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
            "query_columns": columns or [],          # P0: columnas del driver aun en error parcial
            "error_message": error,
            "messages": state.get("messages", []) + [AIMessage(content=f"[SQL] Error DB: {error}")]
        }

    return {
        "query_result": rows,
        "query_columns": columns or [],              # P0: sobreviven aunque haya 0 filas
        "error_message": "",
        "messages": state.get("messages", []) + [AIMessage(content=f"[SQL] Ejecutado. Filas: {len(rows) if rows else 0}")]
    }


# ------------------------------------------------------------------
# Clasificación de errores y enriquecimiento
# ------------------------------------------------------------------
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
    # Gate real de psycopg: 'column "x" does not exist' (antes exigía el literal
    # "UndefinedColumn" y casi nunca disparaba)
    e = (error or "").lower()
    if not ("column" in e and "does not exist" in e):
        return error

    enrichments = []
    for view_name in extract_views_used(sql):
        view_info = _biz_mem.get_view(view_name.replace("semantic.", "").strip())
        if view_info:
            available = list(view_info.metricas.keys()) + view_info.columnas_fecha
            enrichments.append(f"\nColumnas VÁLIDAS para {view_name}: {available}")

    if enrichments:
        return error + "\n" + "\n".join(enrichments)
    return error


# ------------------------------------------------------------------
# Empaquetado del contrato
# ------------------------------------------------------------------
def sql_validate_and_package(state: SQLAgentState, **kwargs) -> Dict[str, Any]:
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

    if err:
        err = _enrich_error_with_valid_columns(err, sql)

    # Política de salida por tipo de error
    if any(m in (err or "") for m in ("SECURITY:", "Ninguna vista permitida", "ERROR_INSALVABLE")):
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
        # P0: columnas reales del driver incluso cuando no hay filas
        driver_columns = state.get("query_columns") or []
        if isinstance(rows_raw, list) and len(rows_raw) > 0:
            columns = list(rows_raw[0].keys())
            rows_norm = rows_raw
            status = "success"
            reason = "Query válida y con datos"
        else:
            # P1: 0 filas ES una respuesta ("no hubo registros"), no un error.
            # can_answer=True corta el loop de re-ejecución en wrapper/supervisor.
            status = "no_data"
            columns = driver_columns
            reason = "Query ejecutada correctamente; no hay registros para los filtros indicados"
            warnings.append("La query devolvió 0 filas (respuesta válida, no error).")

    can_answer = status in ("success", "no_data") and not needs_followup

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
        views_used=extract_views_used(sql),        # P0: auditoría real de obediencia
        attempts=attempts,                          # P0: costo de la generación
        semantic_context_used=semantic_ctx[:500] + "..." if len(semantic_ctx) > 500 else semantic_ctx,
        query_confidence=0.95 if status == "success" else (0.6 if status == "no_data" else 0.0),
        needs_followup=needs_followup,
        reason_for_view_choice=reason,
        can_answer=can_answer,
        reasoning=reason,
        warnings=warnings,
    )

    return {"contract": contract}


# ------------------------------------------------------------------
# Router de reintento — política explícita
# ------------------------------------------------------------------
#  Terminal:      cap de intentos | SECURITY (allowlist) | insalvable por el LLM
#  Recuperable:   COLUMNAS_INVALIDAS (autocorrección con catálogo) |
#                 DML detectado | errores DB recuperables
# ------------------------------------------------------------------
def sql_route_retry(state: SQLAgentState) -> Literal["generate_query", "validate_package"]:
    err = state.get("error_message", "")
    attempts = state.get("attempts", 0)

    if not err:
        return "validate_package"
    if attempts >= 3:
        return "validate_package"

    # Terminales por diseño
    if "SECURITY:" in err:
        return "validate_package"
    if "Ninguna vista permitida" in err:
        return "validate_package"

    # Recuperables con feedback estructurado
    if err.startswith(f"{COL_ERR_PREFIX}:"):
        return "generate_query"
    if "Seguridad: comando DML" in err:
        return "generate_query"
    if _is_recoverable_db_error(err):
        return "generate_query"

    return "validate_package"


# ------------------------------------------------------------------
# Grafo
# ------------------------------------------------------------------
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
