# core/graph_sql_agent.py
# -------------------------------------------------
# Refactor de producción. Principios:
#
#  1. generate ≠ validate. El LLM produce; el código valida. Ambos son nodos
#     separados: la política de reintento se razona mirando el grafo.
#  2. La validación de columnas cubre SOLO referencias externas a la vista.
#     Los alias definidos por la propia query (AS, CTEs, subqueries) son
#     nombres nuevos y legítimos — nunca son "columnas inválidas".
#  3. Autocorrección determinista INEQUÍVOCA: una columna inválida se
#     reescribe solo si exactamente UNA columna del catálogo es compatible
#     por tokens. 0 o 2+ candidatos → error accionable y reintento LLM.
#     Sin sinónimos hardcodeados, sin fuzzy silencioso.
#  4. La similitud semántica la hace el LLM con DEFINICIONES en el catálogo
#     (metricas llega como {nombre: definición}, no como lista de claves).
#  5. Errores clasificados por categoría con política explícita:
#       terminal:    security_view | unsolvable | cap de intentos
#       recuperable: columns | dml | db_recoverable | empty_output
#  6. Guardarraíl de extracción: LIMIT por defecto si la query no lo trae.
#  7. Catálogo con hot-reload por firma sha256 de AGENTS.md.
#
# Compatibilidad (consumidos por orchestrator/supervisor):
#  - COL_ERR_PREFIX exportado; marcadores "SECURITY:", "Ninguna vista
#    permitida", status="no_data", campos views_used/attempts/query_columns.
# -------------------------------------------------
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Literal, NotRequired, Optional, Set, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from core.llm import LLM
from core.database import execute_sql_query
from core.contracts import SQLContract
from core.sql_utils import extract_views_used, extract_columns_used
from core.harness import BusinessMemory, is_view_allowed, DEFAULT_AGENTS_MD_PATH

# Extracción estructural (AST). Requerido en la práctica: sin él el validador
# de columnas es propenso a falsos positivos con alias.
try:
    import sqlglot
    from sqlglot import exp as sqlglot_exp
    _HAS_SQLGLOT = True
except ImportError:
    _HAS_SQLGLOT = False

logger = logging.getLogger("bi_orchestrator")

if not _HAS_SQLGLOT:
    logger.warning("[SQL] sqlglot no instalado (pip install sqlglot). Fallback léxico activo.")


# ------------------------------------------------------------------
# Configuración (todo por env; nada de negocio aquí)
# ------------------------------------------------------------------
MAX_ATTEMPTS = int(os.getenv("SQL_AGENT_MAX_ATTEMPTS", "3"))
SQL_MAX_ROWS = int(os.getenv("SQL_MAX_ROWS", "5000"))   # guardarraíl anti-extracción masiva

COL_ERR_PREFIX = "COLUMNAS_INVALIDAS"   # exportado: lo consume el orchestrator


# ------------------------------------------------------------------
# Catálogo con hot-reload (firma del archivo como clave de caché)
# ------------------------------------------------------------------
def _agents_md_sig() -> str:
    try:
        return hashlib.sha256(DEFAULT_AGENTS_MD_PATH.read_bytes()).hexdigest()[:16]
    except Exception:
        return "na"


@lru_cache(maxsize=8)
def _load_biz_mem(sig: str) -> BusinessMemory:
    logger.info(f"[SQL] Cargando BusinessMemory (sig={sig})")
    return BusinessMemory.from_file()


def get_biz_mem() -> BusinessMemory:
    """Si AGENTS.md cambia en disco, el siguiente llamado recarga el catálogo."""
    return _load_biz_mem(_agents_md_sig())


# ------------------------------------------------------------------
# Categorías de error — máquina de estados con política centralizada
# ------------------------------------------------------------------
class ErrCategory:
    SECURITY_VIEW = "security_view"     # vista fuera del catálogo → terminal
    UNSOLVABLE = "unsolvable"           # el LLM declara imposible con el catálogo → terminal
    COLUMNS = "columns"                 # columnas inválidas → autocorrección o retry
    DML = "dml"                         # comando prohibido → retry
    DB_RECOVERABLE = "db_recoverable"   # errores Postgres corregibles → retry
    DB_TERMINAL = "db_terminal"         # errores Postgres estructurales → terminal
    EMPTY_OUTPUT = "empty_output"       # structured output vacío → retry


_TERMINAL_CATEGORIES = {ErrCategory.SECURITY_VIEW, ErrCategory.UNSOLVABLE}

_FORBIDDEN_RE = re.compile(r"\b(?:DELETE|DROP|INSERT|UPDATE|TRUNCATE)\b")

_DB_DIALECT_RECOVERABLE = (
    "syntax error",
    "invalid input syntax",
    "operator does not exist",
    "ambiguous column",
    "must appear in the group by clause",
    "aggregate functions are not allowed",
    "missing from-clause entry",
    "invalid reference to from-clause entry",
)


def _classify_db_error(error: str) -> str:
    """
    Clasifica por texto de psycopg. MEJORA FUTURA: si core.database expone el
    SQLSTATE de psycopg (42703=undefined_column, 42P01=undefined_table,
    42601=syntax_error, 42803=grouping_error...), clasificar por código y
    eliminar este string-matching.
    """
    e = (error or "").lower()
    if "column" in e and "does not exist" in e:
        return ErrCategory.DB_RECOVERABLE
    if any(p in e for p in _DB_DIALECT_RECOVERABLE):
        return ErrCategory.DB_RECOVERABLE
    return ErrCategory.DB_TERMINAL


@dataclass
class ValidationOutcome:
    ok: bool
    category: Optional[str] = None
    message: str = ""
    invalid_columns: List[str] = field(default_factory=list)


# ------------------------------------------------------------------
# Estado del subgrafo
# ------------------------------------------------------------------
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

    previous_sql: NotRequired[str]
    previous_error: NotRequired[Optional[str]]
    previous_row_count: NotRequired[int]
    query_columns: NotRequired[List[str]]

    # Nuevos: resultados estructurados de la validación
    error_category: NotRequired[Optional[str]]
    invalid_columns: NotRequired[List[str]]
    validation_warnings: NotRequired[List[str]]


# ------------------------------------------------------------------
# Helpers genéricos
# ------------------------------------------------------------------
def _normalize(text: str) -> str:
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
    """
    Catálogo CON DEFINICIONES: la similitud concepto→columna la resuelve el LLM,
    pero solo si ve el significado, no el identificador desnudo.
    """
    biz_mem = get_biz_mem()
    catalog: Dict[str, Any] = {}
    for view_full_name in allowed_views:
        view_info = biz_mem.get_view(view_full_name.replace("semantic.", "").strip())
        if not view_info:
            continue
        catalog[view_full_name] = {
            "tipo": view_info.tipo,
            "descripcion": view_info.descripcion,
            "granularidad": view_info.granularidad,
            "filtro_fecha": view_info.filtro_fecha,
            "metricas": dict(view_info.metricas),      # {nombre: definición}
            "columnas_fecha": view_info.columnas_fecha,
            "notas": view_info.notas,
        }
    return catalog


def _available_columns(view_name: str) -> List[str]:
    view_info = get_biz_mem().get_view(view_name.replace("semantic.", "").strip())
    if not view_info:
        return []
    return list(view_info.metricas.keys()) + view_info.columnas_fecha


# ------------------------------------------------------------------
# Extracción de columnas EXTERNAS (Fix A: alias nunca son inválidas)
# ------------------------------------------------------------------
_SQL_KEYWORDS_LEGACY = {
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


def _parse(sql: str):
    """Árbol sqlglot + conjunto de alias definidos por la propia query."""
    tree = sqlglot.parse_one(sql, read="postgres")
    local_aliases: Set[str] = {a.alias for a in tree.find_all(sqlglot_exp.Alias)}
    for cte in tree.find_all(sqlglot_exp.CTE):
        if cte.alias:
            local_aliases.add(cte.alias)
    for sq in tree.find_all(sqlglot_exp.Subquery):
        if sq.alias:
            local_aliases.add(sq.alias)
    return tree, local_aliases


def _extract_external_columns(sql: str) -> List[str]:
    """Referencias a columnas DE LA VISTA. ORDER BY total_unidades (donde
    total_unidades = SUM(unidades) AS total_unidades) NO es una de ellas."""
    if _HAS_SQLGLOT:
        try:
            tree, local_aliases = _parse(sql)
            return sorted({
                c.name
                for c in tree.find_all(sqlglot_exp.Column)
                if c.name and c.name != "*" and c.name not in local_aliases
            })
        except Exception as e:
            logger.warning(f"[SQL] sqlglot no parseó; validación fail-open: {e}")
            return []
    cols = extract_columns_used(sql)
    return [c for c in cols if c.lower() not in _SQL_KEYWORDS_LEGACY]


# ------------------------------------------------------------------
# Autocorrección determinista inequívoca (derivada del catálogo)
# ------------------------------------------------------------------
def _resolve_by_tokens(concept: str, available: List[str]) -> Optional[str]:
    """
    'total_unidades' → 'unidades' si es la ÚNICA columna compatible.
    0 o 2+ candidatos → None (ambigüedad: decide el LLM, nunca el azar).
    """
    tokens = {t for t in _normalize(concept).split() if len(t) > 2}
    if not tokens:
        return None
    matches = []
    for col in available:
        col_tokens = set(_normalize(col).split())
        if tokens <= col_tokens or col_tokens <= tokens:
            matches.append(col)
    return matches[0] if len(matches) == 1 else None


def _rewrite_columns(sql: str, mapping: Dict[str, str]) -> str:
    """Reescritura AST de referencias externas. Los alias de salida nunca
    entran al mapping (fueron excluidos en la extracción)."""
    tree = sqlglot.parse_one(sql, read="postgres")

    def _tx(node):
        if isinstance(node, sqlglot_exp.Column) and node.name in mapping:
            node.set("this", mapping[node.name])
        return node

    return tree.transform(_tx).sql(dialect="postgres")


def _validate_columns(sql: str) -> ValidationOutcome:
    used_views = extract_views_used(sql)
    if not used_views:
        return ValidationOutcome(ok=True)

    cols = _extract_external_columns(sql)
    if not cols:
        return ValidationOutcome(ok=True)

    biz_mem = get_biz_mem()
    for view_name in used_views:
        available = {_normalize(c) for c in _available_columns(view_name)}
        invalid = [c for c in cols if _normalize(c) not in available]

        if invalid and _HAS_SQLGLOT:
            # Autocorrección inequívoca: no consume intento LLM
            real_cols = _available_columns(view_name)
            mapping = {
                c: _resolve_by_tokens(c, real_cols)
                for c in invalid
            }
            resolved = {k: v for k, v in mapping.items() if v}
            still_invalid = [c for c in invalid if c not in resolved]
            if resolved and not still_invalid:
                # Señal para que el nodo validate aplique la reescritura
                return ValidationOutcome(
                    ok=False,
                    category="autofix",          # categoría interna del nodo
                    message=json.dumps(resolved, ensure_ascii=False),
                    invalid_columns=invalid,
                )
            invalid = still_invalid or invalid

        if invalid:
            view_info = biz_mem.get_view(view_name.replace("semantic.", "").strip())
            available_list = (
                list(view_info.metricas.keys()) + view_info.columnas_fecha
                if view_info else []
            )
            return ValidationOutcome(
                ok=False,
                category=ErrCategory.COLUMNS,
                invalid_columns=invalid,
                message=(
                    f"{COL_ERR_PREFIX}: La vista '{view_name}' no contiene las columnas "
                    f"{invalid}. Columnas VÁLIDAS (con su significado): "
                    f"{json.dumps(dict(view_info.metricas), ensure_ascii=False) if view_info else available_list}. "
                    f"Columnas de fecha: {view_info.columnas_fecha if view_info else []}. "
                    f"Elige la columna cuya DEFINICIÓN coincida con lo pedido y usa su "
                    f"nombre EXACTO."
                ),
            )

    return ValidationOutcome(ok=True)


# ------------------------------------------------------------------
# Guardarraíl de extracción
# ------------------------------------------------------------------
def _enforce_row_limit(sql: str) -> Tuple[str, Optional[str]]:
    """Añade LIMIT si la query no lo trae. Protege la BD y el contexto del analyst."""
    if not _HAS_SQLGLOT:
        return sql, None
    try:
        tree, _ = _parse(sql)
        if isinstance(tree, (sqlglot_exp.Select, sqlglot_exp.Union)) and not tree.args.get("limit"):
            limited = tree.limit(SQL_MAX_ROWS)
            return limited.sql(dialect="postgres"), f"LIMIT {SQL_MAX_ROWS} aplicado por guardarraíl."
    except Exception as e:
        logger.debug(f"[SQL] enforce limit omitido: {e}")
    return sql, None


# ------------------------------------------------------------------
# Generación con salida estructurada (adiós regex de bloques ```sql)
# ------------------------------------------------------------------
class _SQLGeneration(BaseModel):
    """Salida estructurada del LLM: SQL o declaración de imposibilidad."""
    sql: Optional[str] = Field(
        default=None,
        description="UNA query SELECT PostgreSQL válida. Null SOLO si impossible=true.",
    )
    impossible: bool = Field(
        default=False,
        description="True solo si NINGUNA vista permitida resuelve la tarea con el catálogo dado.",
    )
    reason: str = Field(default="", description="Justificación breve.")


_SQL_GEN_LLM = LLM.with_structured_output(_SQLGeneration, method="function_calling")


def sql_fetch_schema(state: SQLAgentState, **kwargs) -> Dict[str, Any]:
    if state.get("schema_info") and state["schema_info"].strip():
        schema = state["schema_info"]
    else:
        schema = "Schema no disponible. Usar el catálogo semántico (con definiciones) y allowed_views."
    return {
        "schema_info": schema,
        "messages": state.get("messages", []) + [AIMessage(content="[SQL] Schema listo.")],
    }


def sql_generate_query(state: SQLAgentState, **kwargs) -> Dict[str, Any]:
    preferred = state.get("preferred_view")
    allowed = state.get("allowed_views", [])
    instruction = state.get("supervisor_instruction")
    catalogo_detallado = _build_sql_catalog(allowed)

    supervisor_section = ""
    if instruction:
        supervisor_section = (
            f"{'=' * 60}\nINSTRUCCIÓN DIRECTA DEL SUPERVISOR:\n{instruction}\n{'=' * 60}\n"
        )

    error_context = state.get("error_message", "") or "Ninguno"
    if error_context != "Ninguno":
        error_context = (
            f"ERROR PREVIO: {error_context}\n"
            f"Corrige usando EXACTAMENTE una columna VÁLIDA cuya DEFINICIÓN coincida "
            f"con lo pedido. NO inventes nombres."
        )

    previous_block = ""
    prev_sql = (state.get("previous_sql") or "").strip()
    if prev_sql:
        previous_block = (
            "\nINTENTO ANTERIOR — NO LO REPITAS:\n"
            f"```sql\n{prev_sql}\n```\n"
            f"Resultado: {state.get('previous_row_count', 0)} filas | "
            f"Error previo: {state.get('previous_error') or 'ninguno'}\n"
        )

    system = SystemMessage(content=f"""
Eres un Data Engineer senior experto en PostgreSQL.

CONTEXTO CRÍTICO:
- TÚ SOLO puedes acceder al esquema 'semantic.'.
- PROHIBIDO usar tablas de staging, raw, public o cualquier otro esquema.

REGLAS ABSOLUTAS:
1. Antes de escribir CUALQUIER query, consulta el CATÁLOGO ESTRUCTURADO DE VISTAS.
2. SOLO puedes referenciar columnas que aparezcan en 'metricas' o 'columnas_fecha'.
3. TRADUCCIÓN CONCEPTO → COLUMNA: si la tarea pide "unidades vendidas" y la vista
   expone `unidades` (definición: "total de unidades vendidas..."), USA `unidades`.
   Decide por la DEFINICIÓN; escribe el NOMBRE exacto del catálogo.
4. Los alias que defines con AS (ej. SUM(unidades) AS total_unidades) son nombres
   NUEVOS, tuyos y válidos. La regla 2 aplica a columnas DE LA VISTA, no a tus alias.
5. Si NINGUNA vista permitida resuelve la tarea, responde impossible=true con reason.
6. Si `preferred_view` existe y contiene las columnas necesarias, ÚSALA.
7. Toda referencia a tabla DEBE ser: semantic.nombre_vista.
8. PROHIBIDO: DELETE, DROP, INSERT, UPDATE, TRUNCATE.
9. Dimensiones de texto: usa ILIKE, nunca '=' directo (ej. sucursal ILIKE 'merced').
10. La query debe responder EXACTAMENTE a la tarea del payload.

PATRÓN OBLIGATORIO cuando execution_strategy="top_n_per_group":
  WITH base AS (
    SELECT <grupo>, <item>, SUM(<metrica>) AS total_<metrica>
    FROM semantic.<vista> WHERE <filtros> GROUP BY <grupo>, <item>
  )
  SELECT *, ROW_NUMBER() OVER (PARTITION BY <grupo> ORDER BY total_<metrica> DESC) AS rn
  FROM base WHERE rn <= <N>
  (una sola query; no dividas por grupo; el ORDER BY puede usar el alias)

Devuelve el objeto estructurado. No escribas prosa.
{supervisor_section}
""")

    payload = state.get("payload")
    if not payload:
        return {
            "error_message": "No se encontró payload en el estado del subgrafo SQL.",
            "error_category": ErrCategory.DB_TERMINAL,
            "attempts": state.get("attempts", 0) + 1,
        }

    ctx = f"""
TAREA DEL PAYLOAD:
{json.dumps(payload, indent=2, ensure_ascii=False)}
{previous_block}
CATÁLOGO ESTRUCTURADO DE VISTAS (columnas con su significado):
{json.dumps(catalogo_detallado, indent=2, ensure_ascii=False)}

VISTA PREFERIDA: {preferred or 'Ninguna'}
VISTAS PERMITIDAS: {allowed}

CONTEXTO SEMÁNTICO DE NEGOCIO:
{state.get('semantic_context', 'No disponible')}

SCHEMA TÉCNICO (solo vistas permitidas):
{state.get('schema_info', 'No cargado')}

ERROR PREVIO / INSTRUCCIÓN DE CORRECCIÓN:
{error_context}
"""
    human = HumanMessage(content=ctx)

    # Historial truncado: solo texto (jamás AIMessages con tool_calls pendientes).
    messages = [m for m in state.get("messages", []) if isinstance(m, (SystemMessage, HumanMessage, AIMessage)) and getattr(m, "content", "")]
    if len(messages) > 4:
        messages = messages[-4:]

    response = _SQL_GEN_LLM.invoke([system] + messages + [human])
    if isinstance(response, dict):
        response = _SQLGeneration(**response)

    attempts = state.get("attempts", 0) + 1
    note = AIMessage(content=f"[SQL] Generación intento {attempts}: impossible={response.impossible}")

    if response.impossible:
        return {
            "generated_sql": "",
            "error_message": f"Ninguna vista permitida resuelve la consulta. {response.reason}",
            "error_category": ErrCategory.UNSOLVABLE,
            "attempts": attempts,
            "messages": messages + [note],
        }

    if not (response.sql or "").strip():
        return {
            "generated_sql": "",
            "error_message": "Salida vacía del generador (sin SQL y sin declaración de imposibilidad).",
            "error_category": ErrCategory.EMPTY_OUTPUT,
            "attempts": attempts,
            "messages": messages + [note],
        }

    return {
        "generated_sql": response.sql.strip(),
        "error_message": "",
        "error_category": None,
        "attempts": attempts,
        "messages": messages + [AIMessage(content=f"Intento {attempts}:\n```sql\n{response.sql.strip()}\n```")],
    }


# ------------------------------------------------------------------
# Nodo de validación programática (determinista)
# ------------------------------------------------------------------
def sql_validate_query(state: SQLAgentState, **kwargs) -> Dict[str, Any]:
    sql = state.get("generated_sql", "")
    warnings = list(state.get("validation_warnings") or [])

    if not sql:
        return {
            "error_message": state.get("error_message") or "No SQL generado",
            "error_category": state.get("error_category") or ErrCategory.EMPTY_OUTPUT,
        }

    if _FORBIDDEN_RE.search(sql.upper()):
        return {
            "generated_sql": "",
            "error_message": "Seguridad: comando DML/DDL detectado. Genera SOLO SELECT.",
            "error_category": ErrCategory.DML,
        }

    used_views = extract_views_used(sql)
    biz_mem = get_biz_mem()

    if used_views:
        invalid_views = [v for v in used_views if not is_view_allowed(v, biz_mem)]
        if invalid_views:
            return {
                "generated_sql": "",
                "error_message": (
                    f"SECURITY: Vistas no documentadas en AGENTS.md: {invalid_views}. "
                    f"Catálogo disponible: {', '.join(biz_mem.list_views())}"
                ),
                "error_category": ErrCategory.SECURITY_VIEW,
            }

        allowed_set = {v.lower().replace("semantic.", "") for v in state.get("allowed_views", [])}
        for v in used_views:
            if v.lower().replace("semantic.", "") not in allowed_set:
                logger.info(f"[SQL] Vista '{v}' documentada pero fuera del shortlist (auditada en views_used).")

    outcome = _validate_columns(sql)

    if outcome.category == "autofix":
        mapping = json.loads(outcome.message)
        sql = _rewrite_columns(sql, mapping)
        fixes = ", ".join(f"{k}→{v}" for k, v in mapping.items())
        logger.info(f"[SQL] Autocorrección inequívoca aplicada: {fixes}")
        warnings.append(f"Autocorrección determinista (inequívoca): {fixes}")
        outcome = ValidationOutcome(ok=True)

    if not outcome.ok:
        logger.warning(f"[SQL] {outcome.message}")
        return {
            "generated_sql": "",
            "error_message": outcome.message,
            "error_category": outcome.category,
            "invalid_columns": outcome.invalid_columns,
            "validation_warnings": warnings,
        }

    sql, limit_note = _enforce_row_limit(sql)
    if limit_note:
        warnings.append(limit_note)

    return {
        "generated_sql": sql,
        "error_message": "",
        "error_category": None,
        "validation_warnings": warnings,
    }


def sql_execute_query(state: SQLAgentState, **kwargs) -> Dict[str, Any]:
    sql = state.get("generated_sql", "")
    if not sql:
        return {
            "query_result": None,
            "error_message": state.get("error_message") or "No SQL generado",
            "error_category": state.get("error_category") or ErrCategory.DB_TERMINAL,
        }

    rows, columns, error = execute_sql_query(sql)

    if error:
        return {
            "query_result": None,
            "query_columns": columns or [],
            "error_message": error,
            "error_category": _classify_db_error(error),
            "messages": state.get("messages", []) + [AIMessage(content=f"[SQL] Error DB: {error}")],
        }

    return {
        "query_result": rows,
        "query_columns": columns or [],
        "error_message": "",
        "error_category": None,
        "messages": state.get("messages", []) + [AIMessage(content=f"[SQL] Ejecutado. Filas: {len(rows) if rows else 0}")],
    }


# ------------------------------------------------------------------
# Empaquetado del contrato (política por categoría)
# ------------------------------------------------------------------
def sql_validate_and_package(state: SQLAgentState, **kwargs) -> Dict[str, Any]:
    rows_raw = state.get("query_result")
    err = state.get("error_message", "")
    sql = state.get("generated_sql", "")
    category = state.get("error_category")
    attempts = state.get("attempts", 0)
    warnings = list(state.get("validation_warnings") or [])

    columns: List[str] = []
    rows_norm: List[Dict[str, Any]] = []
    status = "success"
    reason = ""
    needs_followup = False

    if err:
        err = _enrich_error_with_valid_columns(err, sql)

    if category in _TERMINAL_CATEGORIES:
        status = "error"
        needs_followup = True
        reason = f"Violación de política o insalvable: {err}"
        warnings.append(reason)
    elif attempts >= MAX_ATTEMPTS and err:
        status = "error"
        needs_followup = True
        reason = f"Máximo de reintentos alcanzado ({MAX_ATTEMPTS}). Último error: {err}"
        warnings.append(reason)
    elif err:
        status = "error"
        needs_followup = category != ErrCategory.DB_RECOVERABLE
        reason = f"Error SQL/DB [{category}]: {err}"
    else:
        driver_columns = state.get("query_columns") or []
        if isinstance(rows_raw, list) and len(rows_raw) > 0:
            columns = list(rows_raw[0].keys())
            rows_norm = rows_raw
            status = "success"
            reason = "Query válida y con datos"
        else:
            status = "no_data"
            columns = driver_columns
            reason = "Query ejecutada correctamente; no hay registros para los filtros indicados"
            warnings.append("La query devolvió 0 filas (respuesta válida, no error).")

    can_answer = status in ("success", "no_data") and not needs_followup
    semantic_ctx = state.get("semantic_context", "")

    contract = SQLContract(
        status=status,
        generated_sql=sql,
        columns=columns,
        rows=rows_norm,
        row_count=len(rows_norm),
        error_message=err or None,
        schema_used=["semantic"],
        allowed_views=state.get("allowed_views", []),
        preferred_view=state.get("preferred_view"),
        views_used=extract_views_used(sql),
        attempts=attempts,
        semantic_context_used=semantic_ctx[:500] + "..." if len(semantic_ctx) > 500 else semantic_ctx,
        query_confidence=0.95 if status == "success" else (0.6 if status == "no_data" else 0.0),
        needs_followup=needs_followup,
        reason_for_view_choice=reason,
        can_answer=can_answer,
        reasoning=reason,
        warnings=warnings,
    )
    return {"contract": contract}


def _enrich_error_with_valid_columns(error: str, sql: str) -> str:
    e = (error or "").lower()
    if not ("column" in e and "does not exist" in e):
        return error
    enrichments = []
    for view_name in extract_views_used(sql):
        available = _available_columns(view_name)
        if available:
            enrichments.append(f"\nColumnas VÁLIDAS para {view_name}: {available}")
    return error + "\n" + "\n".join(enrichments) if enrichments else error


# ------------------------------------------------------------------
# Routers — la política se lee directamente del grafo
# ------------------------------------------------------------------
def _route_after_validate(state: SQLAgentState) -> Literal["generate_query", "execute_query", "validate_package"]:
    err = state.get("error_message", "")
    category = state.get("error_category")
    attempts = state.get("attempts", 0)

    if not err:
        return "execute_query"                       # SQL validado → ejecutar
    if category in _TERMINAL_CATEGORIES:
        return "validate_package"                    # terminal por diseño
    if attempts >= MAX_ATTEMPTS:
        return "validate_package"
    return "generate_query"                          # columns/dml/empty → regenerar


def _route_after_execute(state: SQLAgentState) -> Literal["generate_query", "validate_package"]:
    err = state.get("error_message", "")
    category = state.get("error_category")
    attempts = state.get("attempts", 0)

    if not err:
        return "validate_package"
    if category in _TERMINAL_CATEGORIES or category == ErrCategory.DB_TERMINAL:
        return "validate_package"
    if attempts >= MAX_ATTEMPTS:
        return "validate_package"
    if category == ErrCategory.DB_RECOVERABLE:
        return "generate_query"
    return "validate_package"


# ------------------------------------------------------------------
# Grafo: START → fetch → generate → validate → execute → package
# ------------------------------------------------------------------
sql_builder = StateGraph(SQLAgentState)
sql_builder.add_node("fetch_schema", sql_fetch_schema)
sql_builder.add_node("generate_query", sql_generate_query)
sql_builder.add_node("validate_sql", sql_validate_query)
sql_builder.add_node("execute_query", sql_execute_query)
sql_builder.add_node("validate_package", sql_validate_and_package)

sql_builder.add_edge(START, "fetch_schema")
sql_builder.add_edge("fetch_schema", "generate_query")
sql_builder.add_edge("generate_query", "validate_sql")
sql_builder.add_conditional_edges(
    "validate_sql",
    _route_after_validate,
    {"generate_query": "generate_query", "execute_query": "execute_query", "validate_package": "validate_package"},
)
sql_builder.add_conditional_edges(
    "execute_query",
    _route_after_execute,
    {"generate_query": "generate_query", "validate_package": "validate_package"},
)
sql_builder.add_edge("validate_package", END)

SQL_SUBGRAPH = sql_builder.compile()
