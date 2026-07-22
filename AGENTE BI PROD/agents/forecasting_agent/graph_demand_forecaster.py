# --------------------------------------------------------------
# graph_demand_forecaster.py
# Agente de predicción de demanda con XGBoost.
# Grafo LangGraph optimizado:
#   - Carga única de datos históricos
#   - Control de frescura del modelo
#   - Tipos nativos de Python para Supabase
#   - Lags dinámicos según histórico disponible
#   - Estado serializable: sin DataFrames ni modelos XGBoost
# --------------------------------------------------------------
from typing import Any, Dict, TypedDict, Optional, List, Tuple
from datetime import timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from langchain_core.messages import AIMessage

from agents.forecasting_agent.ml_config.supabase_ml import (
    load_sales_clean,
    save_forecasts,
    get_latest_artifact,
    save_model_artifact,
)
import agents.forecasting_agent.feature_engineering as fe


# ------------------------------------------------------------------
# Cachés de módulo para objetos no serializables (DataFrame / modelo)
# ------------------------------------------------------------------
_historical_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
_artifact_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}


def _cache_key(producto: str, sede: str) -> Tuple[str, str]:
    return (normalizar_texto(producto), sede.strip().lower())


# ------------------------------------------------------------------
# Configuración global
# ------------------------------------------------------------------
ARTIFACT_MAX_AGE_DAYS = 7
LOCAL_ARTIFACTS_DIR = Path("artifacts")
LOCAL_ARTIFACTS_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------------
# Estado del grafo: SOLO datos serializables
# ------------------------------------------------------------------
class DemandForecasterState(TypedDict):
    question: str
    producto: str
    sede: str
    modo: str                  # "un_dia" | "n_dias"
    fecha_inicio: Optional[str]
    n_dias: int
    retrain_reason: Optional[str]
    forecasts: List[Dict[str, Any]]
    messages: List[Any]


# ------------------------------------------------------------------
# Funciones auxiliares
# ------------------------------------------------------------------
def normalizar_texto(valor) -> str:
    return str(valor).strip().lower()


def smape(y_true, y_pred):
    return 100 / len(y_true) * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )


def post_procesar(predicciones):
    return np.maximum(np.round(predicciones), 0).astype(int)


def preparar_features(df: pd.DataFrame):
    leakage_cols = ["venta_bruta", "demanda_acumulada_mes", "ratio_vs_same_day"]
    exclude = (
        ["fecha", "Descripcion_normalizada", "Sede_Normalizada", "demanda_total"]
        + leakage_cols
    )
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].select_dtypes(include=[np.number])
    non_num = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_num:
        X = X.drop(columns=non_num)
    y = df["demanda_total"]
    return X, y, list(X.columns)


def entrenar_modelo(X_train, y_train, X_test, y_test):
    X_train = X_train.copy()
    X_test = X_test.copy()
    for d in [X_train, X_test]:
        d.replace([np.inf, -np.inf], 0, inplace=True)
        d.fillna(0, inplace=True)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=4,
        min_child_weight=30,
        subsample=0.6,
        colsample_bytree=0.6,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        tree_method="hist",
        eval_metric="rmse",
    )
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=50,
            verbose=False,
        )
    except TypeError:
        model.set_params(n_estimators=500, learning_rate=0.05)
        model.fit(X_train, y_train, verbose=False)
    return model


def _artifact_local_path(producto: str, sede: str) -> Path:
    safe_prod = normalizar_texto(producto).replace(" ", "_").replace("/", "_")
    safe_sede = normalizar_texto(sede).replace(" ", "_").replace("/", "_")
    return LOCAL_ARTIFACTS_DIR / f"model_{safe_prod}_{safe_sede}.pkl"


def _model_is_fresh(artifact: Dict[str, Any]) -> bool:
    """Verifica si el modelo fue entrenado con datos recientes."""
    fecha_max_str = artifact.get("fecha_max")
    if not fecha_max_str:
        return False

    try:
        fecha_max = pd.Timestamp(fecha_max_str).normalize()
        hoy = pd.Timestamp.now().normalize()
        dias = (hoy - fecha_max).days
        return dias <= ARTIFACT_MAX_AGE_DAYS
    except Exception:
        return False


# ------------------------------------------------------------------
# Nodo 1: Cargar datos históricos UNA SOLA VEZ
# ------------------------------------------------------------------
def load_historical_node(state: DemandForecasterState) -> Dict[str, Any]:
    producto = normalizar_texto(state["producto"])
    sede = state["sede"].strip()

    print(f"   📥 Cargando histórico de {producto} @ {sede} desde Supabase...")

    # Carga eficiente: solo producto/sede necesarios
    df_full = fe.build_ml_dataset_from_db(producto=producto, sede=sede)

    # Guardar en caché de módulo, NO en el estado del grafo
    key = _cache_key(producto, sede)
    _historical_cache[key] = df_full

    print(f"   ✅ Histórico cargado: {len(df_full):,} filas")

    return {
        "messages": state.get("messages", []) + [
            AIMessage(content=f"[DemandForecaster] Histórico cargado: {len(df_full)} registros")
        ],
    }


# ------------------------------------------------------------------
# Nodo 2: Entrenar o reutilizar modelo
# ------------------------------------------------------------------
def train_or_load_artifact(state: DemandForecasterState) -> Dict[str, Any]:
    producto = normalizar_texto(state["producto"])
    sede = state["sede"].strip()
    local_path = _artifact_local_path(producto, sede)
    key = _cache_key(producto, sede)

    df_full = _historical_cache.get(key)
    if df_full is None:
        raise ValueError("No hay historical_df en el cache. Ejecuta load_historical_node primero.")

    # Filtrar dataset para el producto/sede objetivo
    df_obj = df_full[
        (df_full["Descripcion_normalizada"].str.lower() == producto) &
        (df_full["Sede_Normalizada"].str.lower() == normalizar_texto(sede))
    ].copy()

    if df_obj.empty:
        productos_similares = df_full[
            df_full["Descripcion_normalizada"].str.contains(producto[:4], case=False, na=False)
        ]["Descripcion_normalizada"].unique().tolist()[:5]

        sedes_disponibles = df_full[
            df_full["Descripcion_normalizada"].str.lower() == producto
        ]["Sede_Normalizada"].unique().tolist()

        msg = f"No hay datos históricos para {producto} @ {sede}."
        if productos_similares:
            msg += f" Productos similares: {productos_similares}."
        if sedes_disponibles:
            msg += f" Sedes disponibles para {producto}: {sedes_disponibles}."
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # 1. Intentar reutilizar artefacto si existe y es fresco
    # ------------------------------------------------------------------
    artifact_db = get_latest_artifact(producto, sede)
    if artifact_db and local_path.exists():
        try:
            artifact = joblib.load(local_path)

            if _model_is_fresh(artifact):
                print(f"   ♻️  Reutilizando modelo fresco: {artifact['modelo_version']}")
                _artifact_cache[key] = artifact
                return {
                    "retrain_reason": "Modelo reutilizado (fresco)",
                    "messages": state.get("messages", []),
                }

            print(f"   ⚠️  Modelo anterior caducado ({artifact['fecha_max']}). Reentrenando...")

        except Exception as e:
            print(f"   ⚠️  No se pudo cargar modelo local: {e}")

    # ------------------------------------------------------------------
    # 2. Entrenar modelo de cero
    # ------------------------------------------------------------------
    print(f"   🏋️ Entrenando modelo para {producto} @ {sede}...")
    print(f"   📊 Registros históricos: {len(df_obj):,}")
    print(f"   📅 Rango: {df_obj['fecha'].min().date()} → {df_obj['fecha'].max().date()}")

    X, y, feature_names = preparar_features(df_obj)

    # Asegurar que feature_names sean strings nativos
    feature_names = [str(f) for f in feature_names]

    # Split temporal
    orden = df_obj.sort_values(["Sede_Normalizada", "Descripcion_normalizada", "fecha"]).index
    X = X.loc[orden].reset_index(drop=True)
    y = y.loc[orden].reset_index(drop=True)
    df_obj = df_obj.loc[orden].reset_index(drop=True)

    dias_historicos = len(df_obj)
    test_days = min(28, max(7, dias_historicos // 5))
    cutoff = df_obj["fecha"].max() - pd.Timedelta(days=test_days)

    train_mask = df_obj["fecha"] <= cutoff
    test_mask = df_obj["fecha"] > cutoff

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(
            f"No hay suficientes datos para dividir entrenamiento/test. "
            f"Histórico: {dias_historicos} días, necesitas al menos {test_days + 5} días."
        )

    model = entrenar_modelo(
        X[train_mask], y[train_mask],
        X[test_mask], y[test_mask]
    )

    y_pred = post_procesar(model.predict(X[test_mask]))

    df_test = df_obj[test_mask].copy()
    df_test["prediccion"] = y_pred
    df_test["subestimacion"] = np.maximum(df_test["demanda_total"] - df_test["prediccion"], 0)
    safety = float(np.percentile(df_test["subestimacion"].values, 80)) if len(df_test) > 0 else 0.0

    # Métricas como floats nativos de Python
    metrics = {
        "R2": float(r2_score(y[test_mask], y_pred)),
        "MAE": float(mean_absolute_error(y[test_mask], y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y[test_mask], y_pred))),
        "SMAPE": float(smape(y[test_mask], y_pred)),
    }

    # Label encoders como dict de strings -> ints nativos
    le_prod = {
        str(k): int(v)
        for k, v in zip(df_full["Descripcion_normalizada"].unique(), df_full["producto_encoded"].unique())
    }
    le_sede = {
        str(k): int(v)
        for k, v in zip(df_full["Sede_Normalizada"].unique(), df_full["sede_encoded"].unique())
    }

    modelo_version = f"{producto}_{sede}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

    artifact = {
        "model": model,
        "features": feature_names,
        "producto": producto,
        "sede": sede,
        "safety_stock": safety,
        "metrics": metrics,
        "le_prod": le_prod,
        "le_sede": le_sede,
        "test_days": test_days,
        "modelo_version": modelo_version,
        "fecha_max": str(df_obj["fecha"].max().date()),
    }

    # Guardar en caché de módulo (no serializable)
    _artifact_cache[key] = artifact

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------
    joblib.dump(artifact, local_path)
    print(f"   💾 Modelo guardado localmente: {local_path}")

    save_model_artifact({
        "producto": producto,
        "sede": sede,
        "modelo_version": modelo_version,
        "entrenado_hasta": artifact["fecha_max"],
        "features": feature_names,
        "le_prod": le_prod,
        "le_sede": le_sede,
        "safety_stock": float(safety),
        "metricas": metrics,
        "test_days": int(test_days),
        "ruta_artifact": str(local_path),
    })

    print(f"   ✅ Modelo entrenado. R²={metrics['R2']:.3f}, MAE={metrics['MAE']:.2f}, safety_stock={safety:.1f}")

    return {
        "retrain_reason": f"Entrenado con {dias_historicos} días históricos",
        "messages": state.get("messages", []),
    }


# ------------------------------------------------------------------
# Nodo 3: Generar predicciones
# ------------------------------------------------------------------
def predict_node(state: DemandForecasterState) -> Dict[str, Any]:
    producto = normalizar_texto(state["producto"])
    sede = state["sede"].strip()
    n_dias = state.get("n_dias", 7)
    key = _cache_key(producto, sede)

    artifact = _artifact_cache.get(key)
    if not artifact:
        raise ValueError("No hay artefacto cargado. Ejecuta train_or_load_artifact primero.")

    df_full = _historical_cache.get(key)
    if df_full is None:
        raise ValueError("No hay historical_df en el cache.")

    features = artifact["features"]
    le_prod = artifact["le_prod"]
    le_sede = artifact["le_sede"]
    safety = artifact["safety_stock"]

    hist = df_full[
        (df_full["Descripcion_normalizada"].str.lower() == producto) &
        (df_full["Sede_Normalizada"].str.lower() == normalizar_texto(sede))
    ][["fecha", "Sede_Normalizada", "Descripcion_normalizada", "demanda_total"]].copy()

    if hist.empty:
        raise ValueError("No hay histórico para pronosticar")

    # Fecha de inicio: día después del último dato histórico si no se especifica
    fecha_inicio_str = state.get("fecha_inicio")
    if not fecha_inicio_str:
        fecha_inicio_str = (pd.Timestamp(hist["fecha"].max()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    fecha_inicio = pd.Timestamp(fecha_inicio_str)

    # Lags dinámicos según histórico disponible
    dias_historicos = len(hist)
    max_lag = min(28, max(dias_historicos - 2, 1))
    print(f"   📈 Histórico: {dias_historicos} días, usando max_lag={max_lag}")

    registros = []
    for i in range(1, n_dias + 1):
        siguiente_fecha = fecha_inicio + timedelta(days=i - 1)

        nueva_fila = pd.DataFrame([{
            "fecha": siguiente_fecha,
            "Sede_Normalizada": sede,
            "Descripcion_normalizada": producto,
            "demanda_total": 0,
        }])

        combinado = pd.concat([hist, nueva_fila], ignore_index=True)
        combinado = fe.features_calendario_expertas(combinado)
        combinado = fe.features_lags_rollings(combinado, max_lag=max_lag)
        combinado = fe.features_comportamiento_demanda(combinado)
        combinado["producto_encoded"] = combinado["Descripcion_normalizada"].map(le_prod)
        combinado["sede_encoded"] = combinado["Sede_Normalizada"].map(le_sede)

        fila = combinado.iloc[[-1]].copy()
        X = fila[features].fillna(0).replace([np.inf, -np.inf], 0)
        pred = max(round(float(artifact["model"].predict(X)[0])), 0)

        fila["demanda_total"] = pred
        hist = pd.concat([
            hist,
            fila[["fecha", "Sede_Normalizada", "Descripcion_normalizada", "demanda_total"]]
        ], ignore_index=True)

        registros.append({
            "fecha": siguiente_fecha.date().isoformat(),
            "sede": sede,
            "producto": producto,
            "prediccion": int(pred),
            "prediccion_con_buffer": int(pred + np.ceil(safety)),
            "tipo": "futura",
            "modelo_version": artifact["modelo_version"],
            "dias_pronosticados": n_dias,
            "safety_stock": float(safety),
            "metricas": {
                "R2": float(artifact["metrics"]["R2"]),
                "MAE": float(artifact["metrics"]["MAE"]),
                "RMSE": float(artifact["metrics"]["RMSE"]),
                "SMAPE": float(artifact["metrics"]["SMAPE"]),
            },
        })

    # Guardar en Supabase
    save_forecasts(registros)

    return {
        "forecasts": registros,
        "messages": state.get("messages", []) + [
            AIMessage(content=f"[DemandForecaster] {len(registros)} días pronosticados para {producto} @ {sede}")
        ],
    }


# ------------------------------------------------------------------
# Construcción del grafo LangGraph
# ------------------------------------------------------------------
from langgraph.graph import StateGraph, START, END

builder = StateGraph(DemandForecasterState)

builder.add_node("load_historical", load_historical_node)
builder.add_node("train_or_load", train_or_load_artifact)
builder.add_node("predict", predict_node)

builder.add_edge(START, "load_historical")
builder.add_edge("load_historical", "train_or_load")
builder.add_edge("train_or_load", "predict")
builder.add_edge("predict", END)

DEMAND_FORECASTER_GRAPH = builder.compile()


# ------------------------------------------------------------------
# Helper para invocación directa (compatible con versiones anteriores)
# ------------------------------------------------------------------
def forecast_node(state: DemandForecasterState) -> Dict[str, Any]:
    """Invoca todo el grafo de una sola vez."""
    return DEMAND_FORECASTER_GRAPH.invoke(state)


# ------------------------------------------------------------------
# Helper para integración con orquestador
# ------------------------------------------------------------------
def run_forecast(producto: str, sede: str, n_dias: int = 7, fecha_inicio: Optional[str] = None) -> Dict[str, Any]:
    """
    Función de alto nivel para ejecutar una predicción.
    Puede ser usada como tool o nodo del orquestador.
    """
    state = DemandForecasterState(
        question=f"Predice demanda de {producto} en {sede}",
        producto=producto,
        sede=sede,
        modo="n_dias",
        fecha_inicio=fecha_inicio,
        n_dias=n_dias,
        retrain_reason=None,
        forecasts=[],
        messages=[],
    )
    return DEMAND_FORECASTER_GRAPH.invoke(state)
