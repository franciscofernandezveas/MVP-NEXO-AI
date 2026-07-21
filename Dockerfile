FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Asegurar __init__.py en AGENTE BI PROD y subpaquetes
RUN touch "AGENTE BI PROD/__init__.py" && \
    touch "AGENTE BI PROD/core/__init__.py" && \
    touch "AGENTE BI PROD/agents/__init__.py" && \
    for d in "AGENTE BI PROD"/agents/*/; do touch "$d/__init__.py"; done

# Patch: en langgraph >= 1.2.0, InMemorySaver se llama MemorySaver
RUN python - <<'PY'
import os

path = "/app/AGENTE BI PROD/core/orchestrator.py"
if os.path.exists(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    old = "from langgraph.checkpoint.memory import InMemorySaver"
    new = "from langgraph.checkpoint.memory import MemorySaver as InMemorySaver"

    if old in content:
        content = content.replace(old, new)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("✅ core/orchestrator.py patched: MemorySaver as InMemorySaver")
    else:
        print("⚠️ No se encontro import de InMemorySaver")
else:
    print(f"❌ No existe {path}")
PY

ENV CHROMA_DIR=/data/chroma_db
ENV AGENTS_DIR=/data/agents
ENV FILES_DIR=/data/files
ENV VIZ_DIR=/data/visualizations

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
