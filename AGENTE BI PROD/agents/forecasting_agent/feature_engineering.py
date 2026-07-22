import logging
import unicodedata
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .ml_config.supabase_ml import load_sales_clean




logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

MAX_LAGS = 28

FERIADOS_FIJOS_CHILE = [
    "01-01", "01-05", "21-05", "18-09", "19-09",
    "12-10", "31-10", "01-11", "08-12", "25-12",
]


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


def generar_feriados_chile(fecha_min: date, fecha_max: date) -> pd.DatetimeIndex:
    anios = range(fecha_min.year, fecha_max.year + 1)
    feriados = []
    for anio in anios:
        for dia_mes in FERIADOS_FIJOS_CHILE:
            m, d = map(int, dia_mes.split("-"))
            try:
                feriados.append(date(anio, m, d))
            except ValueError:
                continue
    return pd.DatetimeIndex(feriados)


# -----------------------------------------------------------------
# 3️⃣  Carga y preparación desde CSV
# -----------------------------------------------------------------
def cargar_y_preparar(ruta_ventas: Path) -> pd.DataFrame:
    log.info(f"Cargando archivo: {ruta_ventas}")
    df = pd.read_csv(ruta_ventas, encoding="utf-8")
    log.info(f"Shape original: {df.shape}")
    df = normalizar_nombres(df)

    if "Descripcion" in df.columns:
        df["Descripcion_normalizada"] = (
            df["Descripcion"].astype(str).apply(sin_acentos).str.lower().str.strip()
        )
    else:
        raise KeyError("No se encontró 'Descripcion'")

    if "Sede_Normalizada" not in df.columns:
        if "Cuenta" in df.columns:
            cuenta = df["Cuenta"].astype(str).str.lower()
            conditions = [
                cuenta.str.contains("plaza.bolsillo|plaza bolsillo", na=False),
                cuenta.str.contains("merced", na=False),
                cuenta.str.contains("tajamar", na=False),
                cuenta.str.contains("persa.*victor.*manuel|victor.*manuel", na=False),
            ]
            choices = ["Plaza Bolsillo", "Merced", "Tajamar", "Persa Victor Manuel"]
            df["Sede_Normalizada"] = np.select(conditions, choices, default="Sede No Identificada")
        else:
            df["Sede_Normalizada"] = "Sede No Identificada"

    if "Fecha" not in df.columns:
        raise KeyError("No se encontró 'Fecha'")

    df["fecha"] = pd.to_datetime(df["Fecha"], errors="coerce").dt.date
    df = df.dropna(subset=["fecha"]).copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def agregar_y_reindexar_completo(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Agregando y reindexando series completas...")

    df["Cantidad"] = pd.to_numeric(df["Cantidad"], errors="coerce").fillna(0)

    group_cols = ["fecha", "Descripcion_normalizada", "Sede_Normalizada"]
    df_agg = df.groupby(group_cols).agg({"Cantidad": "sum"}).reset_index()
    df_agg = df_agg.rename(columns={"Cantidad": "demanda_total"})

    fecha_min = df_agg["fecha"].min()
    fecha_max = df_agg["fecha"].max()
    fechas_completas = pd.date_range(fecha_min, fecha_max, freq="D")

    grupos = df_agg[["Sede_Normalizada", "Descripcion_normalizada"]].drop_duplicates()
    grupos["_key"] = 1
    fechas_df = pd.DataFrame({"fecha": fechas_completas})
    fechas_df["_key"] = 1

    df_completo = pd.merge(grupos, fechas_df, on="_key").drop("_key", axis=1)
    df_completo = df_completo.merge(df_agg, on=group_cols, how="left")
    df_completo["demanda_total"] = df_completo["demanda_total"].fillna(0)

    log.info(f"Dataset reindexado: {len(df_completo)} filas")
    return df_completo


# -----------------------------------------------------------------
# 5️⃣  Features calendario
# -----------------------------------------------------------------
def features_calendario_expertas(df: pd.DataFrame) -> pd.DataFrame:
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["dia_semana"] = df["fecha"].dt.dayofweek
    df["es_fin_de_semana"] = df["dia_semana"].isin([5, 6]).astype(int)
    df["mes"] = df["fecha"].dt.month
    df["dia_del_mes"] = df["fecha"].dt.day
    df["semana_del_mes"] = ((df["dia_del_mes"] - 1) // 7 + 1).astype(int)
    df["trimestre"] = df["fecha"].dt.quarter
    df["dia_del_ano"] = df["fecha"].dt.dayofyear
    df["semana_del_ano"] = df["fecha"].dt.isocalendar().week.astype(int)

    df["sin_dia_semana"] = np.sin(2 * np.pi * df["dia_semana"] / 7)
    df["cos_dia_semana"] = np.cos(2 * np.pi * df["dia_semana"] / 7)
    df["sin_mes"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["cos_mes"] = np.cos(2 * np.pi * df["mes"] / 12)
    df["sin_dia_ano"] = np.sin(2 * np.pi * df["dia_del_ano"] / 365.25)
    df["cos_dia_ano"] = np.cos(2 * np.pi * df["dia_del_ano"] / 365.25)

    fecha_inicio = df["fecha"].min()
    df["dias_desde_inicio"] = (df["fecha"] - fecha_inicio).dt.days

    feriados = generar_feriados_chile(df["fecha"].min().date(), df["fecha"].max().date())
    df["es_feriado"] = df["fecha"].isin(feriados).astype(int)

    def calc_dias_feriados(row_fecha):
        f = pd.Timestamp(row_fecha)
        proximos = feriados[feriados >= f]
        anteriores = feriados[feriados <= f]
        d_prox = (proximos.min() - f).days if len(proximos) > 0 else 999
        d_ant = (f - anteriores.max()).days if len(anteriores) > 0 else 999
        return pd.Series([d_prox, d_ant])

    feriados_cols = df["fecha"].apply(calc_dias_feriados)
    df["dias_a_feriado"] = feriados_cols[0].astype(int)
    df["dias_desde_feriado"] = feriados_cols[1].astype(int)
    return df


# -----------------------------------------------------------------
# 6️⃣  Lags, Rollings y EWMA
# -----------------------------------------------------------------
def features_lags_rollings(df: pd.DataFrame, max_lag: int = MAX_LAGS) -> pd.DataFrame:
    df = df.sort_values(["Sede_Normalizada", "Descripcion_normalizada", "fecha"]).reset_index(drop=True)
    gcols = ["Sede_Normalizada", "Descripcion_normalizada"]

    for lag in [1, 7, 14, 21, 28]:
        if lag <= max_lag:
            df[f"lag_{lag}"] = df.groupby(gcols)["demanda_total"].shift(lag)

    df["rolling_mean_7"] = df.groupby(gcols)["demanda_total"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    df["rolling_std_7"] = df.groupby(gcols)["demanda_total"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).std())
    df["rolling_mean_14"] = df.groupby(gcols)["demanda_total"].transform(
        lambda x: x.shift(1).rolling(14, min_periods=1).mean())
    df["rolling_max_7"] = df.groupby(gcols)["demanda_total"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).max())

    df["ewma_7"] = df.groupby(gcols)["demanda_total"].transform(
        lambda x: x.shift(1).ewm(span=7, adjust=False).mean())
    df["ewma_14"] = df.groupby(gcols)["demanda_total"].transform(
        lambda x: x.shift(1).ewm(span=14, adjust=False).mean())

    df["rolling_4same_day"] = df.groupby(gcols + ["dia_semana"])["demanda_total"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).mean())
    return df



# -----------------------------------------------------------------
# 7️⃣  Features de comportamiento (SIN leakage)
# -----------------------------------------------------------------
def features_comportamiento_demanda(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Generando features de comportamiento...")
    gcols = ["Sede_Normalizada", "Descripcion_normalizada"]

    def calc_dias_sin_venta(arr):
        res = np.zeros(len(arr), dtype=int)
        for i in range(len(arr)):
            if arr[i] > 0:
                res[i] = 0
            else:
                j = i - 1
                count = 0
                while j >= 0 and arr[j] == 0:
                    count += 1
                    j -= 1
                res[i] = count + 1
        return res

    df["dias_desde_ultima_venta"] = df.groupby(gcols)["demanda_total"].transform(
        lambda x: calc_dias_sin_venta(x.values)
    )

    df["delta_lag1"] = df["lag_1"] - df["lag_2"] if "lag_2" in df.columns else 0

    log.info("Features de comportamiento completados (sin leakage)")
    return df


# -----------------------------------------------------------------
# 8️⃣  Preparación final
# -----------------------------------------------------------------
def preparar_dataset_final(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Preparando dataset final...")

    # Encontrar el lag más alto disponible
    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    if not lag_cols:
        raise ValueError("No se generaron columnas de lag. Datos insuficientes.")

    lag_cols_sorted = sorted(lag_cols, key=lambda x: int(x.split("_")[1]), reverse=True)
    max_lag_col = lag_cols_sorted[0]

    df = df.dropna(subset=[max_lag_col]).copy()

    le_sede = LabelEncoder()
    le_prod = LabelEncoder()
    df["sede_encoded"] = le_sede.fit_transform(df["Sede_Normalizada"])
    df["producto_encoded"] = le_prod.fit_transform(df["Descripcion_normalizada"])

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ["fecha"]:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            if df[col].isna().any():
                media_global = df[col].mean()
                df[col] = df[col].fillna(media_global if pd.notna(media_global) else 0)

    df = df.sort_values(["Sede_Normalizada", "Descripcion_normalizada", "fecha"]).reset_index(drop=True)

    log.info(f"Dataset final: {len(df)} filas, {len(df.columns)} columnas")
    return df


# -----------------------------------------------------------------
# 9️⃣  Build desde Supabase (nuevo)
# -----------------------------------------------------------------
def build_ml_dataset_from_db(
    producto: str = None,
    sede: str = None,
) -> pd.DataFrame:
    # CORREGIDO: ya importado arriba, no es necesario volver a importar
    # from ml_config.supabase_ml import load_sales_clean

    df = load_sales_clean(producto=producto, sede=sede)

    if df.empty:
        raise ValueError("No hay datos históricos en sales_clean para producto/sede")

    df = df.rename(columns={
        "sede": "Sede_Normalizada",
        "producto": "Descripcion_normalizada",
        "cantidad": "Cantidad",
    })

    df = normalizar_nombres(df)
    df_completo = agregar_y_reindexar_completo(df)
    df_completo = features_calendario_expertas(df_completo)
    df_completo = features_lags_rollings(df_completo, max_lag=MAX_LAGS)
    df_completo = features_comportamiento_demanda(df_completo)
    df_final = preparar_dataset_final(df_completo)

    return df_final


# -----------------------------------------------------------------
# 9️⃣  MAIN desde CSV (opcional)
# -----------------------------------------------------------------
if __name__ == "__main__":
    ruta_ventas = Path("informe_ventas_limpio.csv")

    df_raw = cargar_y_preparar(ruta_ventas)
    df_completo = agregar_y_reindexar_completo(df_raw)
    df_completo = features_calendario_expertas(df_completo)
    df_completo = features_lags_rollings(df_completo, max_lag=MAX_LAGS)
    df_completo = features_comportamiento_demanda(df_completo)
    df_final = preparar_dataset_final(df_completo)

    salida = Path("dataset_ml_experto_corregido.csv")
    df_final.to_csv(salida, index=False, encoding="utf-8")
    log.info(f"Archivo guardado: {salida.resolve()}")
