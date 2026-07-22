"""
core/sql_utils.py
Utilidades para parseo y saneamiento de SQL.
"""

import re
import logging
from typing import Set

logger = logging.getLogger(__name__)


def clean_sql_for_extraction(sql: str) -> str:
    """
    Limpia el SQL antes de extraer vistas o columnas:
    - Quita bloques de comentarios /* ... */.
    - Quita comentarios de línea --.
    - Quita literales de string entre comillas simples (reemplaza por ?).
    - Reduce espacios.
    """
    if not sql:
        return ""

    # Quitar comentarios de bloque
    text = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    # Quitar comentarios de línea
    text = re.sub(r"--[^\n]*", " ", text)
    # Quitar literales de string entre comillas simples (incluyendo escapes '')
    text = re.sub(r"'(?:[^']|'')*'", " '?str' ", text)
    # Colapsar espacios
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_views_used(sql: str) -> Set[str]:
    """
    Extrae vistas del esquema 'semantic' referenciadas en el SQL.
    """
    cleaned = clean_sql_for_extraction(sql)
    if not cleaned:
        return set()

    pattern = r"(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)(?:\s+(?:AS\s+)?[a-zA-Z_][a-zA-Z0-9_]*)?(?:\s*,)?"
    matches = re.findall(pattern, cleaned, re.IGNORECASE)

    views: Set[str] = set()
    for match in matches:
        match = match.strip().lower()
        if not match:
            continue

        if "." in match:
            schema, view = match.split(".", 1)
            if schema == "semantic":
                views.add(match)
            else:
                logger.warning(f"[extract_views_used] Esquema no permitido detectado: {match}")
        else:
            logger.warning(
                f"[extract_views_used] Referencia sin esquema detectada: {match}. "
                "El SQL agent debe usar siempre 'semantic.'."
            )

    return views


def normalize_view_name(view_name: str) -> str:
    if not view_name:
        return ""
    clean = view_name.strip().lower()
    if not clean.startswith("semantic."):
        clean = f"semantic.{clean}"
    return clean


def _remove_quoted_strings(sql: str) -> str:
    """Elimina literales de string del SQL para no confundirlos con columnas."""
    return re.sub(r"'(?:[^']|'')*'", " ", sql)


def _remove_functions_and_aliases(sql: str) -> str:
    """
    Heurístico para quitar funciones SQL y alias comunes que no son columnas.
    No es perfecto, pero reduce ruido.
    """
    # Quitar funciones comunes: SUM(col), COUNT(*), etc.
    text = re.sub(r"\b(SUM|COUNT|AVG|MIN|MAX|COALESCE|NULLIF|ROUND|DATE_TRUNC|LOWER|UPPER|TRIM)\s*\(", " ", sql, flags=re.IGNORECASE)
    return text


def extract_columns_used(sql: str) -> Set[str]:
    """
    Extrae columnas referenciadas en posiciones sintácticas de columnas.
    Ignora literales de string, esquemas, alias y funciones.
    Útil para diagnóstico y validación ligera. No es 100% preciso.
    """
    if not sql:
        return set()

    # Paso 1: limpiar comentarios y strings
    text = clean_sql_for_extraction(sql)
    # Paso 2: quitar strings nuevamente por seguridad
    text = _remove_quoted_strings(text)
    # Paso 3: quitar funciones
    text = _remove_functions_and_aliases(text)

    # Extraer identificadores después de palabras clave que introducen columnas
    columns: Set[str] = set()

    # SELECT col1, col2, ...
    for match in re.finditer(r"\bSELECT\b(.*?)\bFROM\b", text, re.IGNORECASE | re.DOTALL):
        select_part = match.group(1)
        # Quitar alias AS alias
        select_part = re.sub(r"\bAS\b\s+[a-zA-Z_][a-zA-Z0-9_]*", " ", select_part, flags=re.IGNORECASE)
        # Extraer identificadores que no sean funciones ya removidas
        identifiers = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", select_part)
        for ident in identifiers:
            if ident.lower() not in RESERVED_WORDS:
                columns.add(ident)

    # WHERE, GROUP BY, ORDER BY, HAVING, ON
    for clause_keyword in ["WHERE", "GROUP BY", "ORDER BY", "HAVING", "ON"]:
        pattern = rf"\b{clause_keyword}\b(.*?)(?:\b(SELECT|FROM|WHERE|GROUP BY|ORDER BY|HAVING|LIMIT|OFFSET|UNION|EXCEPT|INTERSECT)\b|$)"
        for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
            clause_part = match.group(1)
            identifiers = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", clause_part)
            for ident in identifiers:
                if ident.lower() not in RESERVED_WORDS:
                    columns.add(ident)

    return columns


# Palabras reservadas de PostgreSQL que nunca son columnas
RESERVED_WORDS = {
    "select", "from", "where", "group", "by", "order", "having",
    "and", "or", "not", "as", "join", "inner", "left", "right",
    "full", "outer", "on", "limit", "offset", "distinct", "all",
    "union", "except", "intersect", "case", "when", "then", "else",
    "end", "null", "is", "in", "between", "like", "ilike", "true",
    "false", "sum", "count", "avg", "min", "max", "date_trunc",
    "current_date", "interval", "cast", "coalesce", "nullif",
    "lower", "upper", "trim", "round", "semantic", "asc", "desc",
}


if __name__ == "__main__":
    queries = [
        "SELECT * FROM semantic.sales_review_day",
        "SELECT producto, SUM(ventas) AS total FROM semantic.sales_producto_daily WHERE sucursal ILIKE 'merced' AND producto ILIKE 'capuccino' GROUP BY producto ORDER BY total DESC",
        "SELECT a.x FROM semantic.kpi_fidelizacion_detalle AS f JOIN semantic.sales_producto_daily AS p ON f.fecha = p.fecha",
        "SELECT * FROM public.tabla_mala",
    ]

    for q in queries:
        print(f"\nSQL: {q.strip()[:100]}...")
        print(f"Vistas: {extract_views_used(q)}")
        print(f"Columnas: {extract_columns_used(q)}")
