# --------------------------------------------------------------
# etl_csv_to_supabase.py
# Limpia informe_ventas.csv y lo sube a la tabla sales_clean.
# NO genera features. Solo carga datos históricos normalizados.
# --------------------------------------------------------------
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import unicodedata


from ml_config.supabase_ml import upsert_sales_clean


def sin_acentos(texto: str) -> str:
    if pd.isna(texto):
        return ""
    return (
        unicodedata.normalize("NFKD", str(texto))
        .encode("ASCII", "ignore")
        .decode("utf-8")
    )


def normalizar_nombres(df: pd.DataFrame) -> pd.DataFrame:
    import unicodedata
    nuevo = {}
    for col in df.columns:
        col_ascii = sin_acentos(col).strip().replace(" ", "_").replace("-", "_")
        nuevo[col] = col_ascii
    return df.rename(columns=nuevo)


def limpiar_informe_ventas(csv_path: str):
    """
    Lee informe_ventas.csv, limpia y devuelve DataFrame agregado
    listo para subir a sales_clean.
    columns: fecha, sede, producto, cantidad, precio_promedio
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path.resolve()}")

    log = lambda msg: print(f"[ETL] {msg}")

    log(f"Cargando {path}")
    df = pd.read_csv(path, encoding="utf-8")
    log(f"Filas iniciales: {len(df):,}")

    # Validar columnas mínimas
    if "Fecha" not in df.columns:
        raise KeyError("No se encontró la columna 'Fecha'")
    if "Descripción" not in df.columns:
        raise KeyError("No se encontró la columna 'Descripción'")

    # Fecha
    df["fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["fecha"]).copy()

    # Producto: normalizar, quitar acentos, minúsculas
    df["producto"] = (
        df["Descripción"]
        .astype(str)
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.strip()
        .str.lower()
    )

    # Filtrar productos no reales
    mask_no_reales = df["producto"].str.contains(
        r"\b(propina|importe.*personalizado|tip)\b",
        case=False,
        regex=True,
        na=False,
    )
    df = df[~mask_no_reales].copy()

    # Filtrar productos vacíos o inválidos
    invalidos = ["", "nan", "none", "null", "no_especificado"]
    df = df[~df["producto"].isin(invalidos)].copy()

    # Sede
    if "Sede_Normalizada" in df.columns:
        df["sede"] = df["Sede_Normalizada"].astype(str).str.strip()
    elif "Cuenta" in df.columns:
        cuenta = df["Cuenta"].astype(str).str.lower()
        conditions = [
            cuenta.str.contains("plaza.bolsillo|plaza bolsillo", na=False),
            cuenta.str.contains("merced", na=False),
            cuenta.str.contains("tajamar", na=False),
            cuenta.str.contains("persa.*victor.*manuel|victor.*manuel", na=False),
        ]
        choices = ["Plaza Bolsillo", "Merced", "Tajamar", "Persa Victor Manuel"]
        df["sede"] = np.select(conditions, choices, default="Sede No Identificada")
    else:
        df["sede"] = "Sede No Identificada"

    # Filtrar sedes no identificadas
    df = df[df["sede"] != "Sede No Identificada"].copy()

    # Cantidad
    df["cantidad"] = pd.to_numeric(df["Cantidad"], errors="coerce").fillna(0)

    # Precio
    if "Precio_Neto" in df.columns:
        df["precio_promedio"] = pd.to_numeric(df["Precio_Neto"], errors="coerce")
    else:
        df["precio_promedio"] = np.nan

    # Outliers
    q99 = df["cantidad"].quantile(0.99)
    outliers = (df["cantidad"] > q99).sum()
    df.loc[df["cantidad"] > q99, "cantidad"] = q99

    # Agregar por fecha/sede/producto
    df_agg = df.groupby(["fecha", "sede", "producto"], as_index=False).agg({
        "cantidad": "sum",
        "precio_promedio": "mean"
    })

    stats = {
        "filas_iniciales": len(df),
        "filas_finales": len(df_agg),
        "outliers_capeados": int(outliers),
        "fecha_min": str(df_agg["fecha"].min().date()),
        "fecha_max": str(df_agg["fecha"].max().date()),
        "productos": int(df_agg["producto"].nunique()),
        "sedes": int(df_agg["sede"].nunique()),
    }

    return df_agg, stats



def main(csv_path: str = "informe_ventas.csv"):
    df_agg, stats = limpiar_informe_ventas(csv_path)

    print("\n--- Estadísticas ETL ---")
    for k, v in stats.items():
        print(f"{k}: {v}")

    print("\nSubiendo a Supabase sales_clean...")
    upsert_sales_clean(df_agg)
    print(f"✅ Subidas {stats['filas_finales']:,} filas a sales_clean.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL: CSV de ventas a Supabase")
    parser.add_argument(
        "--csv",
        default="informe_ventas.csv",
        help="Ruta al archivo informe_ventas.csv"
    )
    args = parser.parse_args()

    main(args.csv)
