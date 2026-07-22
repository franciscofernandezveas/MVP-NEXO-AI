# --------------------------------------------------------------
# fusion.py
# ETL completo (ventas + transacciones) → sales_clean en Supabase
# --------------------------------------------------------------
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from ml_config.supabase_ml import upsert_sales_clean
from etl_portacafe import limpiar_informe_ventas, limpiar_transacciones, unificar_datasets_ventas


def adaptar_a_sales_clean(df_ml: pd.DataFrame) -> pd.DataFrame:
    """
    Adapta el dataset ML al esquema sales_clean:
    fecha, sede, producto, cantidad
    """
    columnas_necesarias = {"anio", "mes", "dia"}
    faltan = columnas_necesarias - set(df_ml.columns)
    if faltan:
        raise ValueError(f"El dataset ML no tiene columnas para reconstruir fecha: {faltan}")

    df = df_ml.copy()

    df["fecha"] = pd.to_datetime(
        df[["anio", "mes", "dia"]].rename(
            columns={"anio": "year", "mes": "month", "dia": "day"}
        )
    )

    rename_map = {}
    if "Sede_Normalizada" in df.columns:
        rename_map["Sede_Normalizada"] = "sede"
    if "Descripcion_normalizada" in df.columns:
        rename_map["Descripcion_normalizada"] = "producto"
    if "demanda_total" in df.columns:
        rename_map["demanda_total"] = "cantidad"

    df = df.rename(columns=rename_map)

    required = {"fecha", "sede", "producto", "cantidad"}
    faltan = required - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas obligatorias para sales_clean: {faltan}")

    df["producto"] = df["producto"].astype(str).str.strip().str.lower()
    df["sede"] = df["sede"].astype(str).str.strip()
    df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce").fillna(0)

    df = df[df["sede"] != "Sede No Identificada"].copy()
    df = df[df["producto"].notna() & (df["producto"] != "")].copy()

    df = df.groupby(["fecha", "sede", "producto"], as_index=False).agg({
        "cantidad": "sum"
    })

    return df[["fecha", "sede", "producto", "cantidad"]]


def main():
    parser = argparse.ArgumentParser(
        description="Fusion: ETL completo (ventas + transacciones) → Supabase sales_clean"
    )
    parser.add_argument("--ventas", required=True, help="Ruta a informe_ventas.csv")
    parser.add_argument("--transacciones", required=True, help="Ruta a transacciones.csv")
    args = parser.parse_args()

    print("=" * 60)
    print("FUSIÓN ETL → SUPABASE")
    print("=" * 60)

    print("\n[1/4] Limpiando informe_ventas.csv...")
    df_ventas, _ = limpiar_informe_ventas(args.ventas)
    print(f"   ✅ Ventas limpias: {len(df_ventas):,} filas")

    print("\n[2/4] Limpiando transacciones.csv...")
    df_trans, _ = limpiar_transacciones(args.transacciones)
    print(f"   ✅ Transacciones limpias: {len(df_trans):,} filas")

    print("\n[3/4] Fusionando datasets...")
    df_ml = unificar_datasets_ventas(df_ventas, df_trans)
    print(f"   ✅ Dataset ML generado: {len(df_ml):,} filas")

    print("\n[4/4] Adaptando dataset ML a formato sales_clean...")
    df_sales = adaptar_a_sales_clean(df_ml)

    print(f"   📊 Filas finales: {len(df_sales):,}")
    print(f"   📅 Fechas: {df_sales['fecha'].min().date()} → {df_sales['fecha'].max().date()}")
    print(f"   🏪 Sedes: {df_sales['sede'].nunique()}")
    print(f"   ☕ Productos: {df_sales['producto'].nunique()}")

    print("\n   ⬆️  Subiendo a Supabase tabla sales_clean...")
    upsert_sales_clean(df_sales)

    print("\n" + "=" * 60)
    print("✅ FUSIÓN COMPLETADA")
    print("=" * 60)
    print("Tabla actualizada: sales_clean")


if __name__ == "__main__":
    main()
