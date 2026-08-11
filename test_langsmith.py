import os
from dotenv import load_dotenv
load_dotenv()

# Mostrar variables
print("=" * 60)
print("DIAGNÓSTICO DE LANGSMITH")
print("=" * 60)
for k in ["LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_ENDPOINT", 
          "LANGSMITH_PROJECT", "LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY",
          "LANGCHAIN_ENDPOINT", "LANGCHAIN_PROJECT"]:
    v = os.getenv(k, "NOT SET")
    if k.endswith("API_KEY") and v and v != "NOT SET":
        v = v[:10] + "..." + v[-4:]  # enmascarar
    print(f"  {k:25s} = {v}")

# Test directo al endpoint
print("\n" + "=" * 60)
print("TEST DE CONEXIÓN")
print("=" * 60)

api_key = os.getenv("LANGSMITH_API_KEY")
endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT", "default")

if not api_key:
    print("❌ LANGSMITH_API_KEY no está definida")
    exit(1)

try:
    from langsmith import Client
    client = Client(api_key=api_key, api_url=endpoint)
    
    # Listar proyectos del workspace
    print(f"\n📋 Proyectos en el workspace:")
    projects = list(client.list_projects(limit=10))
    for p in projects:
        print(f"   - {p.name} (id={p.id})")
    
    # Verificar que el proyecto existe
    if not any(p.name == project for p in projects):
        print(f"\n⚠️ El proyecto '{project}' NO existe en este workspace")
        print(f"   Proyectos disponibles: {[p.name for p in projects]}")
    else:
        print(f"\n✅ Proyecto '{project}' existe en el workspace")
    
    # Intentar crear un run de prueba
    print(f"\n🧪 Creando run de prueba...")
    from langsmith import traceable
    
    @traceable(run_type="chain", name="test_connection")
    def test_fn(x):
        return x * 2
    
    result = test_fn(21)
    print(f"   Resultado: {result}")
    print(f"✅ Run creado. Revisá https://smith.langchain.com → proyecto '{project}'")
    
except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}: {e}")
    if "403" in str(e):
        print("\n🔍 DIAGNÓSTICO 403:")
        print("   - El API key no es válido o fue revocado")
        print("   - El API key no pertenece a este workspace")
        print("   - El workspace está suspendido o excedió el plan")
