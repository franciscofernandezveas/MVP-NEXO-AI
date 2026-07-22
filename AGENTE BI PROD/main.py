#!/usr/bin/env python3

import sys
from pathlib import Path
from decimal import Decimal
from datetime import date, datetime, time
from uuid import UUID
import json

# 1. PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. CARGAR .ENV Y NORMALIZAR VARIABLES
from core.environment import setup_environment
api_key = setup_environment()

if not api_key:
    print("❌ ERROR: OPENAI_API_KEY no está definida. Verifica tu archivo .env")
    sys.exit(1)

import os
db_uri = os.getenv("SUPABASE_DB_URI", "")
if not db_uri:
    print("❌ ERROR: No se pudo construir SUPABASE_DB_URI.")
    sys.exit(1)

print(f"✅ Configuración OK. DB_HOST detectado: {db_uri.split('@')[-1].split('/')[0]}")

# 3. WARM-UP DE BASE DE DATOS (ANTES de importar el orquestador)
print("🔌 Iniciando warm-up de base de datos...")
from core.database import warmup_db
warmup_db()
print("✅ Warm-up de base de datos completado")

# 4. AHORA SÍ importar el orquestador
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from core.orchestrator import BI_ORCHESTRATOR
from core.config import logger


# =============================================================================
# SERIALIZADOR JSON DEFENSIVO (para Decimals, datetimes, UUIDs)
# =============================================================================
def _json_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


# =============================================================================
# VISUALIZACIÓN
# =============================================================================
def print_header(text: str):
    print(f"\n{'='*70}")
    print(f" {text}")
    print(f"{'='*70}")


def print_event(agent, iteration: int, event: dict):
    """Renderiza el estado de cada nodo durante el streaming."""
    if agent == "planner":
        plan = event.get("plan")
        if plan:
            print(f"\n📋 [Ciclo {iteration}] PLANNER")
            print(f"   ├─ Intención : {plan.intent}")
            print(f"   ├─ Tipo      : {plan.question_type}")
            print(f"   ├─ Métricas  : {plan.metrics}")
            print(f"   ├─ Dimension : {plan.dimensions}")

            if hasattr(plan, 'tasks') and plan.tasks:
                tasks_summary = " | ".join([f"T{t.task_id}:{t.task[:40]}" for t in plan.tasks])
                print(f"   ├─ Tasks     : {len(plan.tasks)} ({tasks_summary})")
            else:
                print(f"   ├─ Tasks     : Ninguna")

            print(f"   └─ Confianza : {plan.confidence:.2f}")

    elif agent == "researcher":
        # NUEVO: Impresión del nodo researcher
        findings = event.get("research_findings", "")
        sql_results = event.get("sql_results", [])
        print(f"\n🔬 [Ciclo {iteration}] RESEARCHER")
        print(f"   ├─ Queries ejecutadas : {len(sql_results)}")
        print(f"   ├─ Findings preview   : {findings[:180]}{'...' if len(findings) > 180 else ''}")
        print(f"   └─ Informe generado   : {'Sí' if findings else 'No'}")

    elif agent == "forecaster":
        # NUEVO: Impresión del nodo de predicción de demanda
        forecasts = event.get("forecast_results", [])
        forecast_error = event.get("forecast_error")
        print(f"\n🏭 [Ciclo {iteration}] FORECASTER")
        if forecast_error:
            print(f"   └─ Error: {forecast_error}")
        else:
            print(f"   └─ Pronóstico generado: {len(forecasts)} días")

    elif agent == "sql_agent":
        results = event.get("sql_results", [])
        print(f"\n🔍 [Ciclo {iteration}] SQL AGENT ({len(results)} tarea(s))")
        for contract in results:
            sql_clean = (contract.generated_sql or "N/A").replace("\n", " ").strip()
            print(f"   ├─ Tarea {contract.task_id}")
            print(f"   │  ├─ Status: {contract.status}")
            print(f"   │  ├─ SQL   : {sql_clean}")
            print(f"   │  ├─ Filas : {contract.row_count}")
            print(f"   │  └─ CanAns: {contract.can_answer}")
            if contract.rows:
                for i, row in enumerate(contract.rows[:3]):
                    print(f"   │     📊 Fila {i+1}: {json.dumps(row, ensure_ascii=False, default=_json_default)}")
                if len(contract.rows) > 3:
                    print(f"   │     ... y {len(contract.rows) - 3} filas más")
            else:
                print(f"   │     ⚠️ Sin filas")
        print(f"   └─ Todas las tareas evaluadas.")

    elif agent == "viz_agent":
        viz = event.get("viz_result")
        print(f"\n📈 [Ciclo {iteration}] VIZ AGENT")
        if viz:
            print(f"   └─ Chart: {getattr(viz, 'chart_type', 'N/A')} | "
                  f"Título: {getattr(viz, 'title', 'N/A')}")

    elif agent == "render_plotly":
        print(f"\n🎨 [Ciclo {iteration}] RENDER PLOTLY → figura renderizada")

    elif agent == "viz_approval":
        print(f"\n✅ [Ciclo {iteration}] VIZ APPROVAL → esperando confirmación")

    elif agent == "analyst":
        answer = event.get("final_answer", "")
        print(f"\n📊 [Ciclo {iteration}] ANALYST")
        preview = answer[:150] + "..." if len(answer) > 150 else answer
        print(f"   └─ Respuesta : {preview}")

    elif agent == "supervisor":
        print(f"\n🧠 [Ciclo {iteration}] SUPERVISOR → reevaluando...")


# =============================================================================
# EJECUCIÓN DEL ORQUESTADOR
# =============================================================================
def run_bi_query(question: str, thread_id: str = "cli-session-001") -> str | None:
    initial_state = {
        "question": question,
        "messages": [HumanMessage(content=question)],
        # NUEVOS CAMPOS: inicializamos todo el estado para evitar missing keys
        "plan": None,
        "sql_results": [],
        "viz_result": None,
        "viz_approved": None,
        "viz_rendered": False,
        "final_answer": None,
        "iteration_count": 0,
        "last_agent": None,
        "harness_context": None,
        "semantic_context": "",
        "allowed_views": [],
        "preferred_view": None,
        "schema_info": "",
        "research_findings": None,
        # NUEVOS CAMPOS: forecasting
        "forecast_request": None,
        "forecast_results": None,
        "forecast_error": None,
    }

    config = RunnableConfig(
        configurable={"thread_id": thread_id},
        recursion_limit=100  # <-- NUEVO: evita límite de recursión
    )

    print_header(f"🚀 EJECUTANDO: {question[:60]}{'...' if len(question) > 60 else ''}")

    final_state = None
    events_seen = 0

    try:
        for event in BI_ORCHESTRATOR.stream(initial_state, config, stream_mode="values"):
            events_seen += 1
            agent = event.get("last_agent")
            iteration = event.get("iteration_count", 0)
            logger.debug(f"Evento {events_seen} | Agente: {agent} | Iter: {iteration}")
            print_event(agent, iteration, event)
            final_state = event

    except Exception as e:
        logger.error(f"Error durante la ejecución del grafo: {e}", exc_info=True)
        print(f"\n❌ Error en la ejecución: {e}")
        return None

    if final_state is None:
        print("\n❌ No se obtuvo estado final del orquestador.")
        return None

    final_answer = final_state.get("final_answer")
    total_iterations = final_state.get("iteration_count", 0)

    print_header("RESPUESTA FINAL")
    if final_answer:
        print(final_answer)
    else:
        print("No se pudo generar una respuesta.")

    print(f"\n📈 Estadísticas: {total_iterations} iteraciones | {events_seen} eventos")
    return final_answer


# =============================================================================
# MODO INTERACTIVO
# =============================================================================
def interactive_mode():
    print_header("MULTI-AGENT BI ORCHESTRATOR")
    print(" Escribe tu pregunta de negocio o 'salir' para terminar.\n")
    session_counter = 1

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Saliendo...")
            break

        if question.lower() in ("salir", "exit", "quit", "q"):
            print("👋 Saliendo del sistema...")
            break

        if not question:
            continue

        run_bi_query(question, thread_id=f"interactive-{session_counter}")
        session_counter += 1
        print("\n" + "-" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Agent BI Orchestrator")
    parser.add_argument("question", nargs="?", help="Pregunta de negocio")
    parser.add_argument("--thread-id", default="cli-session-001")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.getLogger("bi_orchestrator").setLevel(logging.DEBUG)

    if args.question:
        run_bi_query(args.question, thread_id=args.thread_id)
    else:
        interactive_mode()
