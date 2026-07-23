import json
import requests

BASE_URL = "https://mvp-nexo-ai-production.up.railway.app"

# 1. Obtener session_id demo
r = requests.post(f"{BASE_URL}/onboarding/demo/start")
print("demo/start status:", r.status_code)
print("demo/start response:", r.json())
session_id = r.json()["session_id"]

# 2. Enviar pregunta al chat SSE
question = "¿Cuáles fueron las ventas del mes de junio de 2026 en Merced"
print(f"\n➡️  Pregunta: {question}")
print("=" * 70)

response = requests.post(
    f"{BASE_URL}/onboarding/sessions/{session_id}/chat",
    json={"question": question},
    stream=True,
)

print(f"chat status: {response.status_code}")
print("-" * 70)

content_buffer = ""
for line in response.iter_lines():
    if not line:
        continue

    text = line.decode("utf-8")
    if not text.startswith("data: "):
        continue

    data = text[6:]
    if data == "[DONE]":
        continue

    try:
        payload = json.loads(data)
        event_type = payload.get("type")

        if event_type == "start":
            print("🟢 Stream iniciado")

        elif event_type == "progress":
            print(f"📍 {payload.get('node')}: {payload.get('message')}")

        elif event_type == "chart":
            print(f"📊 Chart: {payload.get('url')}")

        elif event_type == "chunk":
            chunk = payload.get("content", "")
            content_buffer += chunk
            print(chunk, end="", flush=True)

        elif event_type == "end":
            print("\n🏁 Stream finalizado")

        elif event_type == "error":
            print(f"\n❌ Error: {payload.get('content')}")

    except json.JSONDecodeError:
        print(f"raw: {data}")

print("\n\n" + "=" * 70)
print("RESPUESTA COMPLETA:")
print(content_buffer)
