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

# Patch para compatibilidad de InMemorySaver
RUN python - <<'PY'
import os

path = "/app/AGENTE BI PROD/core/orchestrator.py"
if os.path.exists(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    old = "from langgraph.checkpoint.memory import InMemorySaver"
    new = '''try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:
    from langgraph.checkpoint.base import BaseCheckpointSaver

    class InMemorySaver(BaseCheckpointSaver):
        """Fallback simple de InMemorySaver para compatibilidad entre versiones."""
        def __init__(self):
            self.storage = {}

        def get_tuple(self, config):
            return self.storage.get(config["configurable"]["thread_id"])

        def put(self, config, checkpoint, metadata, new_versions):
            self.storage[config["configurable"]["thread_id"]] = (checkpoint, metadata)
            return {
                "configurable": {
                    "thread_id": config["configurable"]["thread_id"],
                    "checkpoint_ns": "",
                }
            }

        def list(self, config, *, filter=None, before=None, limit=None):
            return []
'''

    if old in content:
        content = content.replace(old, new)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("✅ core/orchestrator.py patched para InMemorySaver")
    else:
        print("⚠️ No se encontró import de InMemorySaver")
else:
    print(f"❌ No existe {path}")
PY

ENV CHROMA_DIR=/data/chroma_db
ENV AGENTS_DIR=/data/agents
ENV FILES_DIR=/data/files
ENV VIZ_DIR=/data/visualizations

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
