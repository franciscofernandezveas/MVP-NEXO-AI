import json
import requests


def test_demo():
    # 1. Obtener session_id demo
    r = requests.post("http://localhost:8000/onboarding/demo/start")
    print("demo/start status:", r.status_code)
    print("demo/start response:", r.json())
    session_id = r.json()["session_id"]

    # 2. Probar chat SSE
    url = f"http://localhost:8000/onboarding/sessions/{session_id}/chat"
    question = "¿Cuáles fueron las ventas del mes?"

    print(f"\n➡️  Enviando pregunta: {question}")
    print("=" * 70)

    response = requests.post(url, json={"question": question}, stream=True)

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
                content_buffer += payload.get("content", "")
                print(f"📝 chunk: {payload.get('content', '')[:80]}")

            elif event_type == "end":
                print("🏁 Stream finalizado")
                print("=" * 70)
                print("\nRESPUESTA COMPLETA:")
                print(content_buffer)

            elif event_type == "error":
                print(f"❌ Error: {payload.get('content')}")

        except json.JSONDecodeError:
            print(f"raw: {data}")

    print("\nBuffer final:\n", content_buffer)


if __name__ == "__main__":
    test_demo()
