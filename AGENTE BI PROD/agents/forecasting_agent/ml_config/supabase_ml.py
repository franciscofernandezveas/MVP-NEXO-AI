import os
from datetime import date
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor
from supabase import create_client, Client


# ------------------------------------------------------------------
# Cargar .env desde la raíz del proyecto
# ------------------------------------------------------------------
_ENV_PATH = Path(__file__).resolve().parent.parent.parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise EnvironmentError(
        "Debes configurar SUPABASE_URL y SUPABASE_SERVICE_KEY en el archivo .env"
    )

if not DATABASE_URL:
    raise EnvironmentError(
        "Debes configurar DATABASE_URL en el archivo .env"
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

BATCH_SIZE = 1000  # Lotes de 1000 registros para escrituras


# ------------------------------------------------------------------
# Helpers internos
# ------------------------------------------------------------------
def _clean_records(df: pd.DataFrame) -> List[dict]:
    """Convierte NaN/NaT/inf a None para serializar correctamente en Supabase."""
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str).replace(["NaT", "nan", "None"], None)
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].replace([np.nan, np.inf, -np.inf], None)
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].replace({np.nan: None})
        elif df[col].dtype == object:
            df[col] = df[col].replace(["nan", "None", ""], None)

    return df.to_dict("records")


def _serialize_for_json(obj: Any) -> Any:
    """Convierte numpy/pandas a tipos nativos de Python serializables en JSON."""
    if isinstance(obj, dict):
        return {str(k): _serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_for_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


def _get_db_connection():
    """Conexión directa a PostgreSQL usando DATABASE_URL."""
    return psycopg2.connect(DATABASE_URL)


def _execute_query(query: str, params: tuple = ()) -> pd.DataFrame:
    """Ejecuta una consulta SQL y devuelve un DataFrame."""
    conn = _get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    df = pd.DataFrame(rows)

    if not df.empty:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce").fillna(0)
        if "precio_promedio" in df.columns:
            df["precio_promedio"] = pd.to_numeric(df["precio_promedio"], errors="coerce")

    return df


def _upsert_batch(table: str, records: List[dict], on_conflict: str) -> None:
    """Sube un lote mediante Supabase REST API."""
    supabase.table(table).upsert(
        records,
        on_conflict=on_conflict
    ).execute()


def _insert_batch(table: str, records: List[dict]) -> None:
    """Inserta un lote mediante Supabase REST API."""
    supabase.table(table).insert(records).execute()


def _upsert_in_batches(
    df: pd.DataFrame,
    table: str,
    on_conflict: str,
    batch_size: int = BATCH_SIZE
) -> None:
    """Divide el DataFrame en lotes y los sube progresivamente."""
    records = _clean_records(df)
    total = len(records)

    for i in range(0, total, batch_size):
        batch = records[i : i + batch_size]
        _upsert_batch(table, batch, on_conflict)
        print(f"   ⬆️  Subidas {min(i + batch_size, total):,} de {total:,} filas...")


def _insert_in_batches(
    records: List[dict],
    table: str,
    batch_size: int = BATCH_SIZE
) -> None:
    """Inserta registros en lotes."""
    total = len(records)
    for i in range(0, total, batch_size):
        batch = records[i : i + batch_size]
        _insert_batch(table, batch)
        print(f"   ⬆️  Insertadas {min(i + batch_size, total):,} de {total:,} filas...")


# ------------------------------------------------------------------
# sales_clean: datos históricos
# ------------------------------------------------------------------
def load_sales_clean(
    producto: Optional[str] = None,
    sede: Optional[str] = None,
    fecha_min: Optional[date] = None,
    fecha_max: Optional[date] = None
) -> pd.DataFrame:
    """
    Carga datos históricos de sales_clean usando PostgreSQL directo.
    Sin límite de 1000 filas.
    """
    query = "SELECT * FROM sales_clean WHERE 1=1"
    params = []

    if sede:
        query += " AND sede = %s"
        params.append(sede)
    if producto:
        query += " AND producto = %s"
        params.append(producto)
    if fecha_min:
        query += " AND fecha >= %s"
        params.append(fecha_min.isoformat())
    if fecha_max:
        query += " AND fecha <= %s"
        params.append(fecha_max.isoformat())

    query += " ORDER BY fecha"

    df = _execute_query(query, tuple(params))
    print(f"   📥 Cargados {len(df):,} registros de sales_clean")

    return df


def upsert_sales_clean(df: pd.DataFrame) -> None:
    """
    Inserta o actualiza sales_clean por lotes.
    Columnas requeridas: fecha, sede, producto, cantidad
    """
    required = {"fecha", "sede", "producto", "cantidad"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    out = df.copy()
    out["sede"] = out["sede"].astype(str).str.strip()
    out["producto"] = out["producto"].astype(str).str.strip().str.lower()

    # Limpiar valores inválidos
    valores_invalidos = ["", "nan", "none", "null", "no_especificado", "na"]
    out = out[
        out["producto"].notna() &
        ~out["producto"].isin(valores_invalidos) &
        out["sede"].notna() &
        ~out["sede"].isin(valores_invalidos) &
        (out["sede"] != "Sede No Identificada")
    ].copy()

    # Asegurar precio_promedio opcional
    if "precio_promedio" not in out.columns:
        out["precio_promedio"] = np.nan

    out = out[["fecha", "sede", "producto", "cantidad", "precio_promedio"]].copy()

    out["fecha"] = pd.to_datetime(out["fecha"], errors="coerce").dt.date.astype(str)
    out["cantidad"] = pd.to_numeric(out["cantidad"], errors="coerce").fillna(0)
    out["precio_promedio"] = pd.to_numeric(out["precio_promedio"], errors="coerce")

    out = out.dropna(subset=["fecha"]).copy()

    # Eliminar duplicados dentro del mismo lote
    duplicados_antes = out.duplicated(subset=["fecha", "sede", "producto"], keep=False).sum()

    if duplicados_antes > 0:
        print(f"   ⚠️  Detectados {duplicados_antes:,} registros duplicados. Agregando...")
        out = out.groupby(["fecha", "sede", "producto"], as_index=False).agg({
            "cantidad": "sum",
            "precio_promedio": "mean"
        })

    if out.empty:
        raise ValueError("No quedaron registros válidos para subir a sales_clean.")

    print(f"   📦 Total a subir: {len(out):,} filas en lotes de {BATCH_SIZE}")
    _upsert_in_batches(out, "sales_clean", "fecha,sede,producto")
    print(f"   ✅ sales_clean actualizada.")


def delete_sales_clean(
    producto: Optional[str] = None,
    sede: Optional[str] = None
) -> None:
    query = supabase.table("sales_clean").delete()

    if sede:
        query = query.eq("sede", sede)
    if producto:
        query = query.eq("producto", producto)

    query.execute()


def count_sales_clean() -> int:
    response = supabase.table("sales_clean").select("*", count="exact").limit(1).execute()
    return response.count or 0


# ------------------------------------------------------------------
# demand_forecasts: predicciones generadas
# ------------------------------------------------------------------
def save_forecasts(records: List[dict]) -> None:
    """Guarda predicciones en demand_forecasts por lotes."""
    print(f"   📦 Guardando {len(records):,} predicciones...")
    clean_records = [_serialize_for_json(r) for r in records]
    _insert_in_batches(clean_records, "demand_forecasts")


def get_latest_forecasts(
    producto: str,
    sede: str,
    tipo: Optional[str] = None,
    limit: int = 30
) -> pd.DataFrame:
    """
    Recupera predicciones recientes de demand_forecasts usando PostgreSQL directo.
    """
    query = "SELECT * FROM demand_forecasts WHERE producto = %s AND sede = %s"
    params = [producto, sede]

    if tipo:
        query += " AND tipo = %s"
        params.append(tipo)

    query += " ORDER BY fecha DESC LIMIT %s"
    params.append(limit)

    return _execute_query(query, tuple(params))


def delete_forecasts(
    producto: Optional[str] = None,
    sede: Optional[str] = None,
    modelo_version: Optional[str] = None
) -> None:
    query = supabase.table("demand_forecasts").delete()

    if producto:
        query = query.eq("producto", producto)
    if sede:
        query = query.eq("sede", sede)
    if modelo_version:
        query = query.eq("modelo_version", modelo_version)

    query.execute()


# ------------------------------------------------------------------
# model_artifacts: metadatos de modelos entrenados
# ------------------------------------------------------------------
def save_model_artifact(metadata: dict) -> None:
    """Guarda metadatos del modelo entrenado, sanitizando tipos numpy/pandas."""
    clean_metadata = _serialize_for_json(metadata)
    supabase.table("model_artifacts").upsert(
        clean_metadata,
        on_conflict="modelo_version"
    ).execute()


def get_latest_artifact(producto: str, sede: str) -> Optional[dict]:
    response = (
        supabase.table("model_artifacts")
        .select("*")
        .eq("producto", producto)
        .eq("sede", sede)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    data = response.data or []
    return data[0] if data else None


def list_artifacts(producto: Optional[str] = None, sede: Optional[str] = None) -> pd.DataFrame:
    query = supabase.table("model_artifacts").select("*")

    if producto:
        query = query.eq("producto", producto)
    if sede:
        query = query.eq("sede", sede)

    response = query.order("created_at", desc=True).execute()
    return pd.DataFrame(response.data or [])


# ------------------------------------------------------------------
# Utilidades de diagnóstico
# ------------------------------------------------------------------
def verificar_tablas() -> dict:
    """
    Verifica que las tablas existan y tengan registros usando PostgreSQL directo.
    """
    result = {}

    try:
        df = _execute_query("SELECT COUNT(*) AS n FROM sales_clean")
        result["sales_clean"] = int(df["n"].iloc[0])
    except Exception as e:
        result["sales_clean"] = f"error: {e}"

    try:
        df = _execute_query("SELECT COUNT(*) AS n FROM demand_forecasts")
        result["demand_forecasts"] = int(df["n"].iloc[0])
    except Exception as e:
        result["demand_forecasts"] = f"error: {e}"

    try:
        df = _execute_query("SELECT COUNT(*) AS n FROM model_artifacts")
        result["model_artifacts"] = int(df["n"].iloc[0])
    except Exception as e:
        result["model_artifacts"] = f"error: {e}"

    return result


def resumen_sales_clean() -> pd.DataFrame:
    """
    Devuelve un resumen de registros por sede y producto.
    """
    query = """
        SELECT sede, producto, COUNT(DISTINCT fecha) AS dias, SUM(cantidad) AS total_unidades
        FROM sales_clean
        GROUP BY sede, producto
        ORDER BY total_unidades DESC
    """
    return _execute_query(query)
