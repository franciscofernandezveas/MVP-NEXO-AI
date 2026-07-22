# --------------------------------------------------------------
# test_agent.py
# Prueba el demand_forecaster usando el grafo completo.
# --------------------------------------------------------------
from graph_demand_forecaster import DemandForecasterState, DEMAND_FORECASTER_GRAPH


# Configurar caso de prueba
state = DemandForecasterState(
    question="Predice la demanda de americano en Plaza Bolsillo para los próximos 7 días",
    producto="americano",
    sede="Plaza Bolsillo",
    modo="n_dias",
    fecha_inicio="2026-07-07",  # día después del último dato histórico
    n_dias=7,
    artifact=None,
    historical_df=None,
    retrain_reason=None,
    forecasts=[],
    messages=[],
)

# Ejecutar el grafo
print("=" * 60)
print("EJECUTANDO GRAFO DEMAND_FORECASTER")
print("=" * 60)

result = DEMAND_FORECASTER_GRAPH.invoke(state)

print("\n" + "=" * 60)
print("RESULTADO")
print("=" * 60)

artifact = result["artifact"]

print(f"Producto: {artifact['producto']}")
print(f"Sede: {artifact['sede']}")
print(f"Modelo: {artifact['modelo_version']}")
print(f"Motivo: {result.get('retrain_reason', 'No especificado')}")
print(f"Entrenado hasta: {artifact['fecha_max']}")

print(f"\nMétricas del modelo:")
for k, v in artifact["metrics"].items():
    print(f"  {k}: {v:.3f}")

print(f"\nSafety stock p80: {artifact['safety_stock']:.1f} unidades")

print("\nPronóstico:")
print("-" * 40)
print(f"{'Fecha':<12} {'Predicción':>12} {'Con buffer':>12}")
print("-" * 40)

for r in result["forecasts"]:
    print(f"{r['fecha']:<12} {r['prediccion']:>12} {r['prediccion_con_buffer']:>12}")

print("-" * 40)
print(f"\n✅ Prueba completada. Predicciones guardadas en demand_forecasts.")
print(f"   Total de días pronosticados: {len(result['forecasts'])}")
