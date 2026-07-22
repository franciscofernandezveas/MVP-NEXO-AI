import os
import logging
import time as time_module  # Renombrar para evitar conflicto
from functools import lru_cache
from decimal import Decimal
from datetime import date, datetime, time
from uuid import UUID
from typing import Any, List

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool
from langchain_community.utilities.sql_database import SQLDatabase

logger = logging.getLogger("bi_orchestrator")


def serialize_value(value: Any) -> Any:
    """Convierte tipos de BD no serializables a nativos Python."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, bytes):
        try:
            return value.decode('utf-8')
        except UnicodeDecodeError:
            return repr(value)
    return value


def serialize_rows(rows: list[dict]) -> list[dict]:
    """Aplica serialize_value a todas las celdas."""
    return [
        {key: serialize_value(val) for key, val in row.items()}
        for row in rows
    ]


# ---------------------------------------------------------------------------
# URI y Engine
# ---------------------------------------------------------------------------
def _get_db_uri() -> str:
    uri = os.getenv("SUPABASE_DB_URI")
    if not uri or "host" in uri.lower():
        raise RuntimeError(f"SUPABASE_DB_URI no configurada correctamente: {uri}")
    return uri


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    raw_uri = _get_db_uri()
    connection_uri = raw_uri
    if "postgresql://" in raw_uri:
        if "?" not in raw_uri:
            connection_uri += "?client_encoding=utf8&connect_timeout=10"
        elif "client_encoding" not in raw_uri:
            connection_uri += "&client_encoding=utf8&connect_timeout=10"
    
    logger.debug("Configurando pool de conexiones a base de datos...")
    engine = create_engine(
        connection_uri,
        poolclass=QueuePool,
        pool_size=10,          # Conexiones siempre listas
        max_overflow=20,       # Conexiones extra bajo demanda
        pool_timeout=5,        # Fallar rápido si no hay conexión
        pool_pre_ping=True,    # Verificar conexión antes de usar
        pool_recycle=3600,     # Reciclar conexiones cada 1h
    )
    return engine


def warmup_db():
    """Forzar conexión inicial para evitar cold start en primer query."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ DB pool warm-up completado.")
    except Exception as e:
        logger.error(f"❌ DB Warm-up falló: {e}")


# ---------------------------------------------------------------------------
# SQLDatabase restringido al esquema SEMANTIC
# ---------------------------------------------------------------------------
SEMANTIC_SCHEMA = "semantic"


@lru_cache(maxsize=1)
def get_database() -> SQLDatabase:
    """
    Instancia SQLDatabase que SOLO conoce tablas del esquema semantic.
    Esto previene que el toolkit de LangChain vea tablas de staging.
    """
    engine = get_engine()
    # Obtenemos SOLO las tablas del esquema semantic para pasarlas a include_tables
    semantic_tables = get_semantic_table_names()  # devuelve ["semantic.tabla1", ...]
    
    db = SQLDatabase(
        engine,
        sample_rows_in_table_info=2,  # Reducimos para no saturar prompt
        include_tables=semantic_tables,
        ignore_tables=[],
    )
    logger.debug(f"✅ SQLDatabase restringido a {len(semantic_tables)} objetos en '{SEMANTIC_SCHEMA}'")
    return db


# ---------------------------------------------------------------------------
# Inspector nativo: tablas y vistas del esquema semantic
# ---------------------------------------------------------------------------
def get_semantic_table_names() -> list[str]:
    """Devuelve lista de tablas/vistas CUALIFICADAS: ['semantic.xxx', ...]"""
    engine = get_engine()
    inspector = inspect(engine)
    
    tables = inspector.get_table_names(schema=SEMANTIC_SCHEMA)
    views = inspector.get_view_names(schema=SEMANTIC_SCHEMA)
    
    # Devolvemos nombres calificados para forzar al LLM a usar semantic.
    qualified = [f"{SEMANTIC_SCHEMA}.{t}" for t in tables + views]
    return qualified


# Caché con TTL para el esquema (5 minutos)
_schema_cache = {"value": None, "timestamp": 0}
_SCHEMA_TTL = 60 * 5  # 5 minutos

def get_semantic_schema_info_cached(max_objects: int = 30) -> str:
    """
    Genera un DDL-like del esquema semantic usando caché con TTL.
    Evita consultas repetidas a information_schema.
    """
    now = time_module.time()  # Usar time_module en lugar de time
    if _schema_cache["value"] is None or now - _schema_cache["timestamp"] > _SCHEMA_TTL:
        logger.debug("Recargando y cacheando schema semantic...")
        _schema_cache["value"] = get_semantic_schema_info(max_objects)
        _schema_cache["timestamp"] = now
        logger.debug("Schema semantic cacheado.")
    return _schema_cache["value"]


def get_semantic_schema_info(max_objects: int = 30) -> str:
    """
    Genera un DDL-like del esquema semantic usando SQLAlchemy Inspector.
    Solo tablas y vistas del esquema semantic.
    """
    engine = get_engine()
    inspector = inspect(engine)
    
    tables = inspector.get_table_names(schema=SEMANTIC_SCHEMA)
    views = inspector.get_view_names(schema=SEMANTIC_SCHEMA)
    all_objects = tables + views
    
    output_lines = []
    for obj_name in all_objects[:max_objects]:
        obj_type = "VIEW" if obj_name in views else "TABLE"
        
        cols = inspector.get_columns(obj_name, schema=SEMANTIC_SCHEMA)
        col_defs = []
        for c in cols:
            # c es dict con 'name', 'type', 'nullable', 'default', etc.
            col_type = str(c["type"])
            nullable = "" if c.get("nullable", True) else " NOT NULL"
            col_defs.append(f"  {c['name']} {col_type}{nullable}")
        
        ddl = f"-- {obj_type}: {SEMANTIC_SCHEMA}.{obj_name}\nCREATE {obj_type} {SEMANTIC_SCHEMA}.{obj_name} (\n" + ",\n".join(col_defs) + "\n);"
        output_lines.append(ddl)
    
    return "\n\n".join(output_lines)


# ---------------------------------------------------------------------------
# NUEVO: Schema filtrado por vistas permitidas (shortlist del harness/planner)
# ---------------------------------------------------------------------------
def get_semantic_schema_for_views(allowed_views: List[str]) -> str:
    """
    Genera un DDL-like SOLO para las vistas listadas en allowed_views.
    Esto reduce el ruido en el prompt del SQL subgraph al mínimo imprescindible.
    
    Args:
        allowed_views: Lista de vistas calificadas, ej. ['semantic.sales_review_day', ...]
    
    Returns:
        String con DDL de solo esas vistas.
    """
    if not allowed_views:
        return "-- No hay vistas permitidas para describir"
    
    # Limpiar prefijo semantic. si viene incluido
    view_names = []
    for v in allowed_views:
        if isinstance(v, str) and v.startswith(f"{SEMANTIC_SCHEMA}."):
            view_names.append(v.replace(f"{SEMANTIC_SCHEMA}.", ""))
        elif isinstance(v, str):
            view_names.append(v)
    
    engine = get_engine()
    inspector = inspect(engine)
    views = inspector.get_view_names(schema=SEMANTIC_SCHEMA)
    
    output_lines = []
    for v in view_names:
        # Seguridad: validar que existe en semantic antes de describir
        if v not in views:
            output_lines.append(f"-- ▼ Vista: {SEMANTIC_SCHEMA}.{v} (NO ENCONTRADA EN SCHEMA)")
            continue
        
        try:
            cols = inspector.get_columns(v, schema=SEMANTIC_SCHEMA)
            if not cols:
                output_lines.append(f"-- ▼ Vista: {SEMANTIC_SCHEMA}.{v} (sin columnas detectadas)")
                continue
            
            col_defs = []
            for c in cols:
                col_type = str(c["type"])
                nullable = "" if c.get("nullable", True) else " NOT NULL"
                col_defs.append(f"  {c['name']} {col_type}{nullable}")
            
            ddl = f"-- ▼ Vista: {SEMANTIC_SCHEMA}.{v}\nCREATE VIEW {SEMANTIC_SCHEMA}.{v} (\n" + ",\n".join(col_defs) + "\n);"
            output_lines.append(ddl)
        except Exception as e:
            output_lines.append(f"-- Error leyendo {SEMANTIC_SCHEMA}.{v}: {e}")
    
    return "\n\n".join(output_lines)


# ---------------------------------------------------------------------------
# Ejecución SQL segura con serialización
# ---------------------------------------------------------------------------
def execute_sql_query(sql: str) -> tuple[list[dict], list[str], str]:
    engine = get_engine()
    rows: list[dict] = []
    columns: list[str] = []
    error = ""
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            raw_rows = [dict(row) for row in result.mappings()]
            rows = serialize_rows(raw_rows)
            if rows:
                columns = list(rows[0].keys())
    except Exception as e:
        error = str(e)
        logger.warning(f"[SQL] Error en ejecución: {e}")
    
    return rows, columns, error


# Legacy wrappers (ya no usados por el SQL Agent, pero los dejamos compatibles)
def get_table_names() -> list[str]:
    return get_semantic_table_names()


def get_schema_info(tables_limit: int = 20) -> str:
    return get_semantic_schema_info(max_objects=tables_limit)
