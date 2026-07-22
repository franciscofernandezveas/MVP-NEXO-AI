# --------------------------------------------------------------
# etl_portacafe.py
# ETL completo: informe_ventas.csv + transacciones.csv → dataset ML
# --------------------------------------------------------------
import logging
import os
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def sin_acentos(texto: str) -> str:
    if pd.isna(texto):
        return ""
    return (
        unicodedata.normalize("NFKD", str(texto))
        .encode("ASCII", "ignore")
        .decode("utf-8")
    )


def normalizar_nombres(df: pd.DataFrame) -> pd.DataFrame:
    nuevo = {}
    for col in df.columns:
        col_ascii = sin_acentos(col).strip().replace(" ", "_").replace("-", "_")
        nuevo[col] = col_ascii
    return df.rename(columns=nuevo)


def parse_fecha(fecha_str):
    if pd.isna(fecha_str):
        return pd.NaT

    formatos = [
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
    ]

    fecha_str = str(fecha_str).strip()

    for formato in formatos:
        try:
            return pd.to_datetime(fecha_str, format=formato, dayfirst=True)
        except Exception:
            continue

    try:
        return pd.to_datetime(fecha_str, dayfirst=True)
    except Exception:
        return pd.NaT


def limpiar_informe_ventas(input_file, output_file=None):
    logger.info("=== LIMPIANDO INFORME_VENTAS ===")
    stats = {}

    try:
        df = pd.read_csv(input_file, encoding="utf-8")
        stats["filas_iniciales"] = len(df)
        stats["columnas_iniciales"] = len(df.columns)
        logger.info(f"Archivo cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    except Exception as e:
        logger.error(f"Error cargando archivo: {e}")
        return None, {}

    # Normalizar nombres de columnas
    df = normalizar_nombres(df)

    if "Descripcion" not in df.columns:
        raise KeyError("No se encontró 'Descripcion'")

    # Eliminar productos no reales
    filas_antes = len(df)
    df["Descripcion_normalizada_filtro"] = (
        df["Descripcion"].astype(str).str.strip().str.lower()
    )
    mask_no_reales = df["Descripcion_normalizada_filtro"].str.contains(
        r"\b(propina|importe.*personalizado|tip)\b",
        case=False,
        regex=True,
        na=False,
    )
    df = df[~mask_no_reales].copy()
    df = df.drop("Descripcion_normalizada_filtro", axis=1)
    stats["filas_no_reales_eliminadas"] = filas_antes - len(df)

    # Normalizar textos
    columnas_texto = df.select_dtypes(include=["object"]).columns.tolist()
    columnas_excluir = ["Cantidad", "Precio_sin_descuento", "Descuento", "Precio_Bruto", "Precio_Neto", "IVA"]

    for columna in columnas_texto:
        if columna in df.columns and columna not in columnas_excluir:
            df[columna] = df[columna].astype(str).str.strip()
            df[columna] = df[columna].replace(
                ["none", "null", "nan", ""], "no_especificado"
            )

    # Eliminar duplicados
    filas_antes = len(df)
    df = df.drop_duplicates()
    stats["duplicados_eliminados"] = filas_antes - len(df)

    # Fechas
    if "Fecha" in df.columns:
        df["Fecha"] = df["Fecha"].apply(parse_fecha)
        stats["fechas_invalidas_inicial"] = df["Fecha"].isnull().sum()
        df = df.dropna(subset=["Fecha"]).copy()

    # Numéricos
    campos_numericos = ["Cantidad", "Precio_sin_descuento", "Descuento", "Precio_Bruto", "Precio_Neto", "IVA"]
    for campo in campos_numericos:
        if campo in df.columns:
            df[campo] = df[campo].astype(str).str.replace(r"[^\d,.]", "", regex=True)
            df[campo] = df[campo].str.replace(",", ".", regex=False)
            df[campo] = pd.to_numeric(df[campo], errors="coerce").fillna(0)

    # Sedes
    if "Cuenta" in df.columns:
        cuenta_lower = df["Cuenta"].astype(str).str.lower()
        conditions = [
            cuenta_lower.str.contains("plaza.bolsillo|plaza bolsillo", na=False),
            cuenta_lower.str.contains("merced", na=False),
            cuenta_lower.str.contains("tajamar", na=False),
            cuenta_lower.str.contains("persa.*victor.*manuel|victor.*manuel", na=False),
        ]
        choices = ["Plaza Bolsillo", "Merced", "Tajamar", "Persa Victor Manuel"]
        df["Sede_Normalizada"] = np.select(conditions, choices, default="Sede No Identificada")

        mask_na = df["Cuenta"].isna() | df["Cuenta"].astype(str).str.lower().isin(
            ["no_especificado", "nan", "none", ""]
        )
        df.loc[mask_na, "Sede_Normalizada"] = "Sede No Identificada"

    # Productos
    df["Descripcion_normalizada"] = (
        df["Descripcion"].astype(str).str.strip().str.lower()
    )

    # Precios negativos
    if "Precio_Bruto" in df.columns:
        df = df[df["Precio_Bruto"] >= 0].copy()

    # Outliers en precio
    if "Precio_Bruto" in df.columns and len(df) > 10:
        Q1 = df["Precio_Bruto"].quantile(0.25)
        Q3 = df["Precio_Bruto"].quantile(0.75)
        IQR = Q3 - Q1
        limite_superior = Q3 + 3 * IQR
        outliers_mask = df["Precio_Bruto"] > limite_superior
        df.loc[outliers_mask, "Precio_Bruto"] = limite_superior

    stats["filas_finales"] = len(df)
    stats["porcentaje_retencion"] = (
        len(df) / stats["filas_iniciales"] * 100 if stats["filas_iniciales"] > 0 else 0
    )

    if output_file:
        df.to_csv(output_file, index=False, encoding="utf-8")

    logger.info("=== LIMPIEZA INFORME_VENTAS COMPLETADA ===")
    return df, stats


def limpiar_transacciones(input_file, output_file=None):
    logger.info("=== LIMPIANDO TRANSACCIONES ===")
    stats = {}

    try:
        df = pd.read_csv(input_file, encoding="utf-8")
        stats["filas_iniciales"] = len(df)
    except Exception as e:
        logger.error(f"Error cargando archivo: {e}")
        return None, {}

    # Normalizar textos
    columnas_texto = df.select_dtypes(include=["object"]).columns.tolist()
    for columna in columnas_texto:
        if columna in df.columns:
            df[columna] = df[columna].astype(str).str.strip().str.lower()
            df[columna] = df[columna].replace(
                ["none", "null", "nan", "", "nan"], "no_especificado"
            )

    # Normalizar nombres de columnas
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.replace("-", "_")
    )

    # Fechas
    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
        stats["fechas_invalidas"] = df["Fecha"].isnull().sum()
        df = df.dropna(subset=["Fecha"]).copy()

    # Numéricos
    campos_numericos = ["Total", "Depositos", "Comision", "Subtotal", "Impuesto", "Propina"]
    for campo in campos_numericos:
        campo_normalizado = campo.replace(" ", "_")
        if campo_normalizado in df.columns:
            df[campo_normalizado] = pd.to_numeric(df[campo_normalizado], errors="coerce").fillna(0)

    # Duplicados por ID de transacción
    if "ID_de_transaccion" in df.columns and "Estado" in df.columns:
        filas_antes = len(df)
        duplicados_ids = df[df.duplicated("ID_de_transaccion", keep=False)]
        ids_duplicados = duplicados_ids["ID_de_transaccion"].unique()
        stats["ids_duplicados_encontrados"] = len(ids_duplicados)

        if len(ids_duplicados) > 0:

            def priorizar_estado(grupo):
                prioridad = {
                    "exitosa": 3,
                    "pagado": 2,
                    "fallida": 1,
                    "cancelada": 1,
                    "agendado": 1,
                    "no_especificado": 0,
                }
                grupo["prioridad"] = grupo["Estado"].map(prioridad).fillna(0)
                max_prioridad = grupo["prioridad"].max()
                return grupo[grupo["prioridad"] == max_prioridad].iloc[0]

            df_unicas = df.groupby("ID_de_transaccion").apply(priorizar_estado).reset_index(drop=True)
            df_unicas = df_unicas.drop("prioridad", axis=1) if "prioridad" in df_unicas.columns else df_unicas
            stats["duplicados_eliminados"] = filas_antes - len(df_unicas)
            df = df_unicas

    # Filtrar estados válidos
    if "Estado" in df.columns:
        filas_antes = len(df)
        estados_validos = ["exitosa", "pagado"]
        df = df[df["Estado"].isin(estados_validos)].copy()
        stats["filas_filtradas_por_estado"] = filas_antes - len(df)

    # Transacciones válidas
    if "Total" in df.columns:
        df = df[df["Total"] > 0].copy()

    stats["filas_finales"] = len(df)
    stats["porcentaje_retencion"] = (
        len(df) / stats["filas_iniciales"] * 100 if stats["filas_iniciales"] > 0 else 0
    )

    if output_file:
        df.to_csv(output_file, index=False, encoding="utf-8")

    logger.info("=== LIMPIEZA TRANSACCIONES COMPLETADA ===")
    return df, stats


def unificar_datasets_ventas(df_ventas, df_transacciones, output_file=None):
    logger.info("=== FUSIONANDO DATASETS ===")

    if df_ventas is None or df_transacciones is None:
        raise ValueError("Se requieren ambos DataFrames para fusionar")

    # Identificar columnas de ID
    ventas_id_col = (
        "ID_de_transaccion" if "ID_de_transaccion" in df_ventas.columns
        else "ID_de_transacción" if "ID_de_transacción" in df_ventas.columns
        else None
    )
    trans_id_col = (
        "ID_de_transaccion" if "ID_de_transaccion" in df_transacciones.columns
        else "ID_de_transacción" if "ID_de_transacción" in df_transacciones.columns
        else None
    )

    if not ventas_id_col or not trans_id_col:
        raise ValueError(
            "No se encontraron columnas de ID_de_transaccion en ventas o transacciones"
        )

    logger.info(f"Columna ID ventas: {ventas_id_col}")
    logger.info(f"Columna ID transacciones: {trans_id_col}")

    # Preparar datasets
    cols_trans_relevantes = [trans_id_col]
    for col in ["Fecha", "Total", "Subtotal", "Impuesto", "Propina", "Metodo_de_pago", "Estado"]:
        if col in df_transacciones.columns:
            cols_trans_relevantes.append(col)

    cols_trans_relevantes = [c for c in cols_trans_relevantes if c in df_transacciones.columns]
    df_trans_prep = df_transacciones[cols_trans_relevantes].copy()

    df_trans_prep = df_trans_prep.rename(columns={trans_id_col: "ID_transaccion"})
    df_ventas_prep = df_ventas.copy()
    df_ventas_prep = df_ventas_prep.rename(columns={ventas_id_col: "ID_transaccion"})

    # Merge
    df_unificado = pd.merge(
        df_ventas_prep,
        df_trans_prep,
        on="ID_transaccion",
        how="left",
        suffixes=("_venta", "_trans"),
    )

    logger.info(f"Dataset unificado: {df_unificado.shape[0]} registros")

    # Fecha
    if "Fecha" in df_unificado.columns:
        df_unificado["Fecha"] = pd.to_datetime(df_unificado["Fecha"], errors="coerce")
    elif "Fecha_venta" in df_unificado.columns:
        df_unificado["Fecha"] = pd.to_datetime(df_unificado["Fecha_venta"], errors="coerce")
    elif "Fecha_trans" in df_unificado.columns:
        df_unificado["Fecha"] = pd.to_datetime(df_unificado["Fecha_trans"], errors="coerce")

    # Componentes temporales
    df_unificado["anio"] = df_unificado["Fecha"].dt.year
    df_unificado["mes"] = df_unificado["Fecha"].dt.month
    df_unificado["dia"] = df_unificado["Fecha"].dt.day
    df_unificado["dia_semana"] = df_unificado["Fecha"].dt.dayofweek
    df_unificado["semana_anio"] = df_unificado["Fecha"].dt.isocalendar().week

    # Normalizar textos
    if "Categoria" in df_unificado.columns:
        df_unificado["Categoria_normalizada"] = df_unificado["Categoria"].astype(str).str.strip().str.lower()
    else:
        df_unificado["Categoria_normalizada"] = "no_especificado"

    if "Descripcion" in df_unificado.columns:
        df_unificado["Descripcion_normalizada"] = df_unificado["Descripcion"].astype(str).str.strip().str.lower()
    elif "Descripcion_normalizada" in df_unificado.columns:
        df_unificado["Descripcion_normalizada"] = df_unificado["Descripcion_normalizada"].astype(str).str.strip().str.lower()
    else:
        df_unificado["Descripcion_normalizada"] = "no_especificado"

    # Asegurar tipos numéricos
    campos_numericos = ["Cantidad", "Precio_sin_descuento", "Descuento", "Precio_Bruto", "Precio_Neto", "IVA", "Total"]
    for campo in campos_numericos:
        if campo in df_unificado.columns:
            df_unificado[campo] = pd.to_numeric(df_unificado[campo], errors="coerce").fillna(0)

    # Dataset agregado para ML
    cols_agrupacion = ["Descripcion_normalizada", "Categoria_normalizada"]
    if "Sede_Normalizada" in df_unificado.columns:
        cols_agrupacion.append("Sede_Normalizada")
    cols_agrupacion.extend(["anio", "mes", "dia", "dia_semana"])

    cols_agrupacion = [c for c in cols_agrupacion if c in df_unificado.columns]

    if "Cantidad" not in df_unificado.columns:
        raise ValueError("No se encontró la columna Cantidad")

    agg_dict = {"Cantidad": "sum"}
    if "ID_transaccion" in df_unificado.columns:
        agg_dict["ID_transaccion"] = "nunique"

    df_ml = df_unificado.groupby(cols_agrupacion).agg(agg_dict).reset_index()

    rename_dict = {"Cantidad": "demanda_total"}
    if "ID_transaccion" in df_unificado.columns:
        rename_dict["ID_transaccion"] = "num_transacciones"

    df_ml = df_ml.rename(columns=rename_dict)

    # Features adicionales
    if "Descripcion_normalizada" in df_ml.columns and "Sede_Normalizada" in df_ml.columns:
        try:
            demanda_producto_sede = (
                df_ml.groupby(["Descripcion_normalizada", "Sede_Normalizada"])["demanda_total"]
                .mean()
                .reset_index()
                .rename(columns={"demanda_total": "demanda_promedio_producto"})
            )
            df_ml = pd.merge(
                df_ml, demanda_producto_sede,
                on=["Descripcion_normalizada", "Sede_Normalizada"],
                how="left",
            )
        except Exception as e:
            logger.warning(f"Error calculando demanda promedio: {e}")

    # Validación final
    if "demanda_total" in df_ml.columns and "Sede_Normalizada" in df_ml.columns:
        df_ml = df_ml[
            (df_ml["demanda_total"] >= 0) &
            (df_ml["Sede_Normalizada"] != "Sede No Identificada")
        ]

    if output_file:
        df_ml.to_csv(output_file, index=False, encoding="utf-8")

    logger.info(f"Dataset ML final: {df_ml.shape[0]} registros, {df_ml.shape[1]} columnas")
    return df_ml
