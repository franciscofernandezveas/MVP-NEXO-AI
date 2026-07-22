import json
from typing import Any, Dict, Literal
from langchain_core.messages import AIMessage
from langgraph.types import Command
from core.llm import LLM
from core.contracts import SupervisorDecision
from core.config import logger

from langsmith import traceable

MAX_ITERATIONS = 15


@traceable(name="Supervisor: Route Next Agent")
def supervisor_node(state: Dict[str, Any]) -> Command[
    Literal[
        "planner", "sql_agent", "analyst", "viz_agent",
        "render_plotly", "viz_approval", "researcher",
        "forecaster",
        "__end__"
    ]
]:
    current_iter = state.get("iteration_count", 0)
    next_iter = current_iter + 1

    # 1. Límite de iteraciones
    if current_iter >= MAX_ITERATIONS:
        logger.warning(f"[Supervisor] Límite de iteraciones ({MAX_ITERATIONS}) alcanzado.")
        return Command(
            goto="__end__",
            update={
                "final_answer": state.get("final_answer") or "No pude completar la consulta tras varios intentos.",
                "last_agent": "supervisor",
                "iteration_count": next_iter,
                "messages": [AIMessage(content=f"[Supervisor] Iteración límite ({MAX_ITERATIONS}) alcanzado. Cerrando.")]
            }
        )

    # Lectura de estado
    plan = state.get("plan")
    sql_results = state.get("sql_results")
    viz_result = state.get("viz_result")
    viz_approved = state.get("viz_approved")
    viz_rendered = state.get("viz_rendered", False)
    final_answer = state.get("final_answer")
    last_agent = state.get("last_agent")
    forecast_results = state.get("forecast_results")
    forecast_error = state.get("forecast_error")

    # === ANTI-LOOP: Validación de viz_result ===
    viz_is_valid = False
    if viz_result:
        has_chart_type = getattr(viz_result, "chart_type", None) is not None
        is_success = getattr(viz_result, "status", None) == "success"
        viz_is_valid = has_chart_type and is_success

    if viz_result and not viz_is_valid and not viz_rendered:
        logger.info("[Supervisor] Viz result inválido (sin chart_type o con error). Saltando renderización.")
        return Command(
            goto="analyst",
            update={
                "viz_rendered": True,
                "last_agent": "supervisor",
                "iteration_count": next_iter,
                "messages": [AIMessage(content="[Supervisor] Spec de visualización inválida, se omite render y se continúa.")]
            }
        )

    if last_agent == "render_plotly" and not viz_is_valid:
        logger.info("[Supervisor] Prevención de loop post-render fallido. Forzando avance.")
        return Command(
            goto="analyst",
            update={
                "viz_rendered": True,
                "last_agent": "supervisor",
                "iteration_count": next_iter,
                "messages": [AIMessage(content="[Supervisor] Render previo no aplicable, continuando.")]
            }
        )

    # === RUTAS DE FLUJO ===

    # 1. Sin plan → Planner
    if not plan:
        logger.info("[Supervisor] Ruteando a Planner - no hay plan")
        return Command(
            goto="planner",
            update={
                "last_agent": "supervisor",
                "iteration_count": next_iter
            }
        )

    # ------------------------------------------------------------------
    # 2. Ruta de predicción de demanda (PRIORIDAD MÁXIMA)
    # ------------------------------------------------------------------
    if plan and getattr(plan, "question_type", None) == "demand_forecast":
        # Aún no se ejecuta forecast → ir al forecaster
        if not forecast_results and not forecast_error:
            logger.info("[Supervisor] Ruteando a Forecaster - predicción de demanda solicitada")
            return Command(
                goto="forecaster",
                update={
                    "last_agent": "supervisor",
                    "iteration_count": next_iter,
                }
            )

        # Forecast ejecutado con error → al analyst para redactar respuesta de error
        if forecast_error and not final_answer:
            logger.info("[Supervisor] Forecast falló, ruteando a Analyst para explicar el error")
            return Command(
                goto="analyst",
                update={
                    "last_agent": "supervisor",
                    "iteration_count": next_iter,
                }
            )

        # Forecast ejecutado con resultados → al analyst para redactar la respuesta final
        if forecast_results and not final_answer:
            logger.info("[Supervisor] Forecast completado, ruteando a Analyst para redactar respuesta")
            return Command(
                goto="analyst",
                update={
                    "last_agent": "supervisor",
                    "iteration_count": next_iter,
                }
            )

        # Si ya hay respuesta final → terminar
        logger.info("[Supervisor] Forecast y respuesta final listos, finalizando")
        return Command(
            goto="__end__",
            update={
                "last_agent": "supervisor",
                "iteration_count": next_iter,
            }
        )

    # ------------------------------------------------------------------
    # 3. Ruta de informe profundo
    # ------------------------------------------------------------------
    if (
        plan
        and getattr(plan, "question_type", None) == "deep_research"
        and state.get("research_findings") is None
    ):
        logger.info("[Supervisor] Ruteando a Researcher - informe profundo solicitado")
        return Command(
            goto="researcher",
            update={
                "last_agent": "supervisor",
                "iteration_count": next_iter,
            }
        )

    # 4. Con plan pero sin resultados SQL → SQL Agent
    if not sql_results:
        logger.info("[Supervisor] Ruteando a SQL Agent - no hay resultados")
        return Command(
            goto="sql_agent",
            update={
                "last_agent": "supervisor",
                "iteration_count": next_iter
            }
        )

    # 5. Plan pide visualización y aún no hemos corrido el viz_agent
    if getattr(plan, "visualization_candidate", False) and viz_result is None:
        logger.info("[Supervisor] Ruteando a Viz Agent - visualización solicitada (sin spec previa)")
        return Command(
            goto="viz_agent",
            update={
                "last_agent": "supervisor",
                "iteration_count": next_iter
            }
        )

    # 6. Hay spec VÁLIDA y aún no renderizada → Render Plotly
    if viz_is_valid and not viz_rendered:
        logger.info("[Supervisor] Ruteando a Render Plotly")
        return Command(
            goto="render_plotly",
            update={
                "last_agent": "supervisor",
                "iteration_count": next_iter
            }
        )

    # 7. Sin respuesta final → Analyst
    if not final_answer:
        logger.info("[Supervisor] Ruteando a Analyst - no hay respuesta final")
        return Command(
            goto="analyst",
            update={
                "last_agent": "supervisor",
                "iteration_count": next_iter
            }
        )

    # 8. Viz ya resuelto y hay datos → Viz Approval (si aplica)
    if viz_is_valid and viz_rendered and viz_approved is None:
        suitable_results = [
            r for r in sql_results
            if getattr(r, "can_answer", False) and len(getattr(r, "rows", [])) > 0
        ]
        if suitable_results:
            logger.info("[Supervisor] Ruteando a Viz Approval")
            return Command(
                goto="viz_approval",
                update={
                    "last_agent": "supervisor",
                    "iteration_count": next_iter
                }
            )

    # 9. Todo listo → Fin
    logger.info("[Supervisor] Todo completado, finalizando")
    return Command(
        goto="__end__",
        update={
            "last_agent": "supervisor",
            "iteration_count": next_iter
        }
    )
