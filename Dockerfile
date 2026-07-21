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

RUN touch "AGENTE BI PROD/__init__.py" && \
    touch "AGENTE BI PROD/core/__init__.py" && \
    touch "AGENTE BI PROD/agents/__init__.py" && \
    for d in "AGENTE BI PROD"/agents/*/; do touch "$d/__init__.py"; done

ENV CHROMA_DIR=/data/chroma_db
ENV AGENTS_DIR=/data/agents
ENV FILES_DIR=/data/files
ENV VIZ_DIR=/data/visualizations

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]