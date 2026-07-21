FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    sed \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hacer ejecutable el entrypoint
RUN chmod +x /app/entrypoint.sh

ENV CHROMA_DIR=/data/chroma_db
ENV AGENTS_DIR=/data/agents
ENV FILES_DIR=/data/files
ENV VIZ_DIR=/data/visualizations

EXPOSE 8000

CMD ["/app/entrypoint.sh"]