from ml_config.supabase_ml import load_sales_clean

import pandas as pd

# Cargar todo sales_clean
df = load_sales_clean()

print("=" * 60)
print(f"Total filas en sales_clean: {len(df):,}")
print(f"Fechas: {df['fecha'].min().date()} → {df['fecha'].max().date()}")
print(f"Días únicos: {df['fecha'].nunique()}")
print(f"Sedes: {df['sede'].nunique()}")
print(f"Productos únicos: {df['producto'].nunique()}")

print("\n--- Sedes ---")
print(df["sede"].value_counts())

print("\n--- Top 20 productos (total unidades) ---")
prod_counts = df.groupby("producto")["cantidad"].sum().sort_values(ascending=False).head(20)
print(prod_counts)

print("\n--- Combinaciones producto/sede con más registros ---")
combo_counts = df.groupby(["sede", "producto"]).agg(
    dias=("fecha", "nunique"),
    total_unidades=("cantidad", "sum")
).sort_values("dias", ascending=False).head(20)
print(combo_counts)

print("\n--- Buscando 'capuccino' ---")
capu = df[df["producto"].str.contains("capu", na=False)]
print(capu.groupby("producto")["cantidad"].sum().sort_values(ascending=False))

print("\n--- Buscando 'Plaza Bolsillo' ---")
pb = df[df["sede"].str.contains("Bolsillo", na=False, case=False)]
print(pb["sede"].value_counts())
