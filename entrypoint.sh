#!/bin/bash
set -e

echo "=== Aplicando patches para compatibilidad ==="

# Patch 1: InMemorySaver -> MemorySaver en langgraph >= 1.2.0
ORCH_PATH="/app/AGENTE BI PROD/core/orchestrator.py"
if [ -f "$ORCH_PATH" ]; then
    if grep -q "from langgraph.checkpoint.memory import InMemorySaver" "$ORCH_PATH"; then
        sed -i 's/from langgraph.checkpoint.memory import InMemorySaver/from langgraph.checkpoint.memory import MemorySaver as InMemorySaver/' "$ORCH_PATH"
        echo "✅ Patch aplicado: MemorySaver as InMemorySaver"
    else
        echo "ℹ️  No se encontró import de InMemorySaver (ya puede estar parchado)"
    fi
else
    echo "❌ No existe $ORCH_PATH"
fi

# Patch 2: lazy supabase client (si aún no es lazy)
SUPABASE_PATH="/app/AGENTE BI PROD/core/supabase_client.py"
if [ -f "$SUPABASE_PATH" ]; then
    if grep -q "^supabase = get_supabase_client()" "$SUPABASE_PATH"; then
        sed -i 's/^supabase = get_supabase_client()/# supabase = get_supabase_client()  # lazy/' "$SUPABASE_PATH"
        echo "✅ Patch aplicado: supabase client lazy"
    fi
fi

# Patch 3: Asegurar __init__.py
touch "/app/AGENTE BI PROD/__init__.py"
touch "/app/AGENTE BI PROD/core/__init__.py"
touch "/app/AGENTE BI PROD/agents/__init__.py"
for dir in /app/AGENTE\ BI\ PROD/agents/*/; do
    touch "$dir/__init__.py"
done

echo "=== Iniciando servidor ==="
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"