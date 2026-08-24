# core/graph_planner.py
# -------------------------------------------------
# Cambios aplicados:
#  2.1 Contexto conversacional filtrado (sin logs internos de agentes)
#  2.2 Señal estructurada `is_replan`/`replan_reason` (sin "olfateo de strings")
#  2.3 Extracción de forecast contra catálogo (sedes/productos desde BusinessMemory)
#  2.4 Camino de replan consciente de forecast_error
#  2.5 Reglas del prompt generadas desde el catálogo (sin vistas/SQL hardcodeado)
#  2.6 needs_followup respeta al LLM + gate de confidence + missing_information
#  2.7 IDs deterministas versionados (v{plan_version}-t{i}) con remap de depends_on
#  2.8 ALINEACIÓN: _column_exists EXACTO-only contra el catálogo (BusinessMemory =
#      única verdad). SEMANTIC_MAP ELIMINADO — era anti-patrón.
#  2.9 Menores: hints consistentes vía _select_view_for_task en el fallback,
#      safeguard conserva tasks, PLANNER_LLM hoisted, _get() robusto
#
#  3.x SIMETRÍA CON EL SQL AGENT REFACTORIZADO:
#  3.1 Catálogo con DEFINICIONES: _build_view_catalog envía {nombre: significado},
#      no solo claves — la resolución concepto→columna la hace el LLM con contexto.
#  3.2 get_biz_mem() compartido (hot-reload por sha256 de AGENTS.md): se elimina
#      _biz_mem de módulo. Antes: editar AGENTS.md dejaba al planner sirviendo
#      vistas viejas mientras el sql_agent validaba contra las nuevas.
#  3.3 _align_task_columns: normaliza metrics/dimensions del payload cuando el
#      match por tokens es INEQUÍVOCO (mismo _resolve_by_tokens que el sql_agent);
#      lo ambiguo sigue viajando como warning hacia el LLM del sql_agent.
#  3.4 Prompt: regla 8 (elegir por DEFINICIÓN, escribir NOMBRE exacto) +
#      regla 16 (top N por grupo = execution_strategy="top_n_per_group",
#      NUNCA dividir por valor de dimensión — lo ejecuta el sql_agent con
#      ROW_NUMBER() OVER (PARTITION BY ...)).
#  3.5 Replan con memoria: consume state["replan_errors"] (evidencia que el
#      orchestrator preserva antes de limpiar) y la marca como consumida;
#      plan_version deriva del conteo de replans del ledger (antes: siempre v1).
#
# REQUIERE (cambios externos):
#  - harness.py:      get_biz_mem() (snippet al pie de esta migración)
#  - orchestrator.py: OrchestratorState.replan_errors + stall branch poblándolo
#  - sql_agent:       refactor con catálogo con definiciones (archivo alineado)
# -------------------------------------------------
from typing import Any, Dict, List, Optional, Set
import json
import logging
import re

from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.llm import LLM
from core.contracts import PlannerContract, SQLPayload
from core.harness import get_biz_mem          # (3.2) antes: _biz_mem de módulo

from core.rag import obtener_candidatas_detalles


logger = logging.getLogger(__name__)

# (2.9) Structured output hoisted: no se reconstruye en cada llamada al nodo
PLANNER_LLM = LLM.with_structured_output(PlannerContract, method="function_calling")


# ------------------------------------------------------------------
# Helpers básicos
# ------------------------------------------------------------------
def _normalize(text: str) -> str:
    """Higiene de strings (case/acentos/underscores). NO es un mapa semántico."""
    return (
        (text or "").lower().strip().replace("_", " ")
        .replace("á", "a").replace("é", "e").replace("í", "i")
        .replace("ó", "o").replace("ú", "u")
    )


def _get(obj: Any, attr: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


# ------------------------------------------------------------------
# (2.1) Contexto conversacional FILTRADO — nunca logs internos
# ------------------------------------------------------------------
def _build_conversational_context(
    messages: List[Any],
    current_question: str,
    max_turns: int = 4,
) -> str:
    if not messages:
        return current_question

    turnos: List[tuple] = []
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = getattr(msg, "content", "") or ""
            if content:
                turnos.append(("Usuario", content[:300]))
        elif isinstance(msg, AIMessage):
            content = getattr(msg, "content", "") or ""
            # convención del sistema: todos los logs internos empiezan con "[Agente]"
            if content and not content.startswith("["):
                turnos.append(("Asistente", content[:300]))

        if len(turnos) >= max_turns * 2:
            break

    if len(turnos) <= 1:
        return current_question

    turnos.reverse()
    return (
        "Contexto reciente de la conversación:\n"
        + "\n".join(f"{role}: {content}" for role, content in turnos)
        + f"\n\nPregunta actual del usuario: {current_question}"
    )


# ------------------------------------------------------------------
# (2.9/3.5) _build_replan_context: ahora con evidencia técnica del fallo
# ------------------------------------------------------------------
def _build_replan_context(state: Dict[str, Any]) -> str:
    """
    Contexto estructurado con hechos y progreso previo para que el planner
    genere un plan alternativo cuando hay estancamiento.
    """
    task_ledger = state.get("task_ledger")
    progress_ledger = state.get("progress_ledger")
    sql_results = state.get("sql_results", []) or []

    parts: List[str] = []

    # (2.2) la razón de replan llega estructurada desde el supervisor
    replan_reason = state.get("replan_reason")
    if replan_reason:
        parts.append(f"Motivo de replanificación: {replan_reason}")

    # (3.5) evidencia técnica preservada por el orchestrator ANTES de limpiar
    # sql_results. Sin esto el replan era amnésico y repetía el mismo plan.
    replan_errors = state.get("replan_errors") or []
    if replan_errors:
        parts.append(
            "ERRORES TÉCNICOS DEL PLAN ANTERIOR (prohibido repetirlos):\n"
            + "\n".join(replan_errors)
        )

    original_question = _get(task_ledger, "original_question") or state.get("question", "")
    parts.append(f"Pregunta original: {original_question}")

    completed_steps = _get(progress_ledger, "completed_steps", []) or []
    if completed_steps:
        parts.append(f"Pasos completados previamente: {completed_steps}")

    stall_count = _get(progress_ledger, "stall_count", 0)
    if stall_count:
        parts.append(f"Iteraciones sin progreso detectadas: {stall_count}")

    if sql_results:
        summaries = []
        for r in sql_results:
            summaries.append(
                f"- Tarea {_get(r, 'task_id')}: "
                f"status={_get(r, 'status')}, "
                f"filas={_get(r, 'row_count')}, "
                f"can_answer={_get(r, 'can_answer')}, "
                f"error={_get(r, 'error_message', 'ninguno')}"
            )
        parts.append("Resultados SQL previos:\n" + "\n".join(summaries))

    if state.get("forecast_error"):
        parts.append(f"Error previo en forecast: {state['forecast_error']}")

    if state.get("research_findings"):
        parts.append("Ya existen hallazgos de research previos; considéralos en el nuevo plan.")

    facts = _get(task_ledger, "facts_verified", []) or []
    if facts:
        facts_text = "\n".join(f"- {_get(f, 'content', '')}" for f in facts)
        parts.append(f"Hechos verificados hasta ahora:\n{facts_text}")

    return "\n\n".join(parts)


# ------------------------------------------------------------------
# Detección y extracción de demand forecast
# ------------------------------------------------------------------
FORECAST_KEYWORDS = [
    "pronosticar", "predicción", "predice", "forecast",
    "pronostico", "pronóstico", "demanda futura", "demanda proyectada",
    "cuánto se venderá", "cuanto se vendera", "cuánto venderemos",
    "cuanto venderemos", "proyección de demanda", "proyeccion de demanda",
    "estimar ventas", "pronosticar ventas",
]

# DEPRECATED como fuente primaria (2.3): solo se usan si BusinessMemory
# aún no expone list_sedes()/list_productos(). Única fuente real = catálogo.
_SEDES_FALLBACK = ["Plaza Bolsillo", "Merced", "Tajamar", "Persa Victor Manuel"]
_PRODUCTOS_FALLBACK = [
    "americano", "capuccino", "latte", "espresso", "mokaccino",
    "cortado", "flat white", "iced latte", "chai latte", "chocolate caliente",
]

_PHRASE_DAYS = {
    "una semana": 7, "dos semanas": 14, "tres semanas": 21,
    "cuatro semanas": 28, "un mes": 30, "semana": 7,
}

_N_DIAS_MIN, _N_DIAS_MAX = 1, 30  # consistente con ForecastRequest(n_dias ge/le)


class _ForecastParamsInternal(BaseModel):
    producto: Optional[str] = None
    sede: Optional[str] = None
    n_dias: int = 7
    fecha_inicio: Optional[str] = None


def _is_demand_forecast_question(question: str) -> bool:
    q = (question or "").lower()
    return any(k in q for k in FORECAST_KEYWORDS)


def _get_sedes_validas() -> List[str]:
    """(2.3/3.2) Fuente única: catálogo (hot-reload). Fallback a constante legacy."""
    for method in ("list_sedes", "get_sedes", "sedes"):
        fn = getattr(get_biz_mem(), method, None)
        try:
            sedes = fn() if callable(fn) else None
            if sedes:
                return sorted(sedes)
        except Exception as e:
            logger.warning(f"[Planner] {method}() falló: {e}")
    return list(_SEDES_FALLBACK)


def _get_productos_validos() -> List[str]:
    """(2.3/3.2) Fuente única: catálogo (hot-reload). Fallback a constante legacy."""
    for method in ("list_productos", "list_productos_frecuentes", "get_productos"):
        fn = getattr(get_biz_mem(), method, None)
        try:
            productos = fn() if callable(fn) else None
            if productos:
                return sorted(productos)
        except Exception as e:
            logger.warning(f"[Planner] {method}() falló: {e}")
    return list(_PRODUCTOS_FALLBACK)


def _match_to_catalog(value: Optional[str], valid_options: List[str]) -> Optional[str]:
    if not value:
        return None
    v = _normalize(value)
    for opt in valid_options:
        if _normalize(opt) == v:
            return opt
    for opt in valid_options:
        o = _normalize(opt)
        if v in o or o in v:
            return opt
    return None


def _extract_n_dias(text_normalized: str) -> int:
    for phrase, days in _PHRASE_DAYS.items():
        if phrase in text_normalized:
            return days
    m = re.search(r"(\d+)\s*semanas?", text_normalized)
    if m:
        return int(m.group(1)) * 7
    m = re.search(r"(\d+)\s*d[ií]as?", text_normalized)
    if m:
        return int(m.group(1))
    if re.search(r"(\d+)\s*mes(?:es)?", text_normalized):
        return _N_DIAS_MAX
    return 7


def _clamp_n_dias(n: Any) -> int:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return 7
    clamped = max(_N_DIAS_MIN, min(_N_DIAS_MAX, n))
    if clamped != n:
        logger.warning(f"[Planner] n_dias={n} fuera de rango, clampeado a {clamped}")
    return clamped


def _extract_forecast_params(question: str) -> Dict[str, Any]:
    sedes = _get_sedes_validas()
    productos = _get_productos_validos()

    parser = LLM.with_structured_output(_ForecastParamsInternal)
    try:
        result = parser.invoke(
            "Extrae los parámetros para predecir demanda de la siguiente pregunta.\n"
            f"Sedes válidas (usa el nombre EXACTO de esta lista): {sedes}\n"
            f"Productos frecuentes (orientativo): {productos}\n"
            "Si la sede mencionada no está en la lista, elige la más parecida. "
            "Si no hay fecha de inicio, devuelve null.\n\n"
            f"Pregunta: {question}"
        )
        if isinstance(result, dict):
            result = _ForecastParamsInternal(**result)
        params = result.model_dump()
    except Exception as e:
        logger.warning(f"[Planner] Falló extracción estructurada de forecast: {e}")
        params = {}

    params["sede"] = _match_to_catalog(params.get("sede"), sedes)
    params["producto"] = _match_to_catalog(params.get("producto"), productos) or params.get("producto")
    params["n_dias"] = _clamp_n_dias(params.get("n_dias", 7))
    return params


def _fallback_extract_forecast_params(question: str) -> Dict[str, Any]:
    qn = _normalize(question)
    sedes = _get_sedes_validas()
    productos = _get_productos_validos()

    sede_detectada = next((s for s in sedes if _normalize(s) in qn), None)
    producto_detectado = next((p for p in productos if _normalize(p) in qn), None)

    return {
        "producto": producto_detectado,
        "sede": sede_detectada,
        "n_dias": _clamp_n_dias(_extract_n_dias(qn)),
        "fecha_inicio": None,
    }


# ------------------------------------------------------------------
# (2.8) Validación léxica de columnas: EXACTA contra el catálogo.
# SEMANTIC_MAP eliminado (anti-patrón): cortafuegos, no resolver.
# ------------------------------------------------------------------
def _column_exists(view_info, column_name: str) -> bool:
    if not view_info:
        return False

    available = {_normalize(k) for k in view_info.metricas.keys()}
    available |= {_normalize(c) for c in view_info.columnas_fecha}
    return _normalize(column_name) in available


def _build_view_catalog(allowed_views: List[str]) -> Dict[str, Any]:
    """
    (3.1) Catálogo CON DEFINICIONES — simétrico a _build_sql_catalog del
    sql_agent refactorizado. La similitud concepto→columna la resuelve el LLM,
    pero solo si ve el significado de cada métrica, no el identificador desnudo.
    """
    catalog = {}
    for view_full_name in allowed_views:
        view_name = view_full_name.replace("semantic.", "").strip()
        view_info = get_biz_mem().get_view(view_name)
        if not view_info:
            continue
        catalog[view_name] = {
            "tipo": view_info.tipo,
            "descripcion": view_info.descripcion,
            "granularidad": view_info.granularidad,
            "filtro_fecha": view_info.filtro_fecha,
            "metricas": dict(view_info.metricas),      # {nombre: definición}
            "columnas_fecha": view_info.columnas_fecha,
            "notas": view_info.notas,
        }
    return catalog


# ------------------------------------------------------------------
# (2.5) Reglas del prompt generadas DESDE el catálogo, no hardcodeadas
# ------------------------------------------------------------------
def _build_dynamic_rules(view_catalog: Dict[str, Any]) -> str:
    rules: List[str] = []

    hourly = [
        n for n, v in view_catalog.items()
        if "hora" in _normalize(v.get("granularidad") or "")
    ]
    historical = [
        n for n, v in view_catalog.items()
        if _normalize(v.get("tipo") or "") == "historical"
    ]
    latest_like = [
        n for n, v in view_catalog.items()
        if n.endswith(("_latest", "_day")) or _normalize(v.get("tipo") or "") in ("latest", "current")
    ]

    if hourly:
        rules.append(
            f"- Preguntas por hora / franja horaria / 'horas pico' → usar: {hourly}. "
            f"En `filters_description` expresa la intención temporal en lenguaje natural "
            f"('fines de semana', 'martes', 'horas pico', 'ayer'); el SQL Agent la traduce."
        )
        rules.append(
            f"- 'Demanda por hora' HISTÓRICA es `question_type='aggregation'` o `'multi_query'` "
            f"con {hourly}, NUNCA 'demand_forecast'."
        )
    if historical:
        rules.append(
            f"- Rangos de fechas explícitos ('junio 2026', 'últimos 90 días', 'ayer', "
            f"'fines de semana') → vistas históricas: {historical}."
        )
        if latest_like:
            rules.append(
                f"- NUNCA usar {latest_like} para periodos históricos; son snapshots actuales."
            )

    if not rules:
        rules.append("- Sin reglas especiales para las vistas disponibles; usa el catálogo.")
    return "\n".join(rules)


# ------------------------------------------------------------------
# (2.8) Severidades separadas: errores duros de vista vs warnings de columna
# ------------------------------------------------------------------
def _validate_task_integrity(task, allowed_views: List[str]) -> List[str]:
    errors: List[str] = []
    preferred = task.preferred_view
    if not preferred:
        return ["La tarea no tiene preferred_view asignada."]

    if preferred not in allowed_views:
        errors.append(f"La vista '{preferred}' no está en allowed_views.")
        return errors

    view_name = preferred.replace("semantic.", "").strip()
    if not get_biz_mem().get_view(view_name):
        errors.append(f"La vista '{preferred}' no está documentada en AGENTS.md.")

    return errors


def _collect_column_warnings(task) -> List[str]:
    """
    Chequeo exacto NO bloqueante post-alineación (3.3): lo que queda aquí es
    genuinamente ambiguo o inexistente; viaja como assumption con los nombres
    válidos y el SQL Agent decide con las definiciones del catálogo.
    """
    warnings: List[str] = []
    view_info = get_biz_mem().get_view((task.preferred_view or "").replace("semantic.", "").strip())
    if not view_info:
        return warnings

    available = list(view_info.metricas.keys()) + view_info.columnas_fecha
    for col in [*(getattr(task, "metrics", []) or []), *(getattr(task, "dimensions", []) or [])]:
        if col and str(col).strip() and not _column_exists(view_info, col):
            warnings.append(
                f"'{col}' no es columna exacta de {view_info.nombre}. Usa una de: {available}"
            )
    return warnings


# ------------------------------------------------------------------
# (3.3) Alineación inequívoca de columnas del payload
# (mismo algoritmo que la autocorrección del sql_agent; derivado del catálogo)
# ------------------------------------------------------------------
def _resolve_by_tokens(concept: str, available: List[str]) -> Optional[str]:
    """
    Resolución determinista INEQUÍVOCA: 'unidades vendidas' → 'unidades_vendidas'
    solo si es la ÚNICA columna compatible. 0 o 2+ candidatos → None: la
    ambigüedad la resuelve el LLM con las definiciones, nunca una heurística.
    """
    tokens = {t for t in _normalize(concept).split() if len(t) > 2}
    if not tokens:
        return None
    matches = [
        col for col in available
        if tokens <= set(_normalize(col).split()) or set(_normalize(col).split()) <= tokens
    ]
    return matches[0] if len(matches) == 1 else None


def _align_task_columns(task) -> None:
    """Normaliza metrics/dimensions a nombres reales cuando el match es
    inequívoco. Cada resolución queda trazada en assumptions."""
    view_info = get_biz_mem().get_view((task.preferred_view or "").replace("semantic.", "").strip())
    if not view_info:
        return
    available = list(view_info.metricas.keys()) + view_info.columnas_fecha
    for attr in ("metrics", "dimensions"):
        aligned = []
        for col in getattr(task, attr, None) or []:
            if col and str(col).strip() and not _column_exists(view_info, col):
                resolved = _resolve_by_tokens(col, available)
                if resolved:
                    task.assumptions = [
                        *(getattr(task, "assumptions", []) or []),
                        f"'{col}' se resolvió a la columna '{resolved}' (match inequívoco).",
                    ]
                    col = resolved
            aligned.append(col)
        setattr(task, attr, aligned)


def _find_compatible_view(task, allowed_views: List[str]) -> Optional[str]:
    required = [*(getattr(task, "metrics", []) or []), *(getattr(task, "dimensions", []) or [])]
    required = [c for c in required if c and str(c).strip()]
    if not required:
        return None

    for view_full_name in (getattr(task, "candidate_views", []) or allowed_views):
        if view_full_name not in allowed_views:
            continue
        view_info = get_biz_mem().get_view(view_full_name.replace("semantic.", "").strip())
        if not view_info:
            continue
        missing = [c for c in required if not _column_exists(view_info, c)]
        if not missing:
            return view_full_name

    return None


def _ensure_semantic_prefix(view_name: Optional[str]) -> Optional[str]:
    if not view_name:
        return None
    return view_name if view_name.startswith("semantic.") else f"semantic.{view_name}"


# ------------------------------------------------------------------
# Selección de vista unificada (Retriever + Fallback léxico)
# ------------------------------------------------------------------
def _select_view_for_task(
    task: SQLPayload,
    allowed_views: List[str],
    original_query: str = "",
) -> tuple[Optional[str], List[str], List[str]]:
    """
    Migrado a core.rag (umbral coseno 0.20; el ranking ordena y la mejor
    candidata autorizada es [0]). Fallback unificado fuera del try/except.
    (3.3): tras validar, _align_task_columns normaliza lo inequívoco.
    """
    errors: List[str] = []

    original_candidates = [
        _ensure_semantic_prefix(cv)
        for cv in (getattr(task, "candidate_views", []) or [])
        if cv and _ensure_semantic_prefix(cv) in allowed_views
    ]

    preferred: Optional[str] = None
    retriever_candidates: List[str] = []

    try:
        detailed = obtener_candidatas_detalles(
            query=task.task,
            k=5,
            allowed_views=allowed_views,
            min_score_threshold=0.20,
        )
        retriever_candidates = [
            _ensure_semantic_prefix(c["view_name"])
            for c in detailed
            if _ensure_semantic_prefix(c["view_name"]) in allowed_views
        ]
        if retriever_candidates:
            preferred = retriever_candidates[0]
    except Exception as e:
        logger.warning(f"[Planner] retriever falló para tarea {task.task_id}: {e}")

    if not preferred:
        preferred = _find_compatible_view(task, allowed_views) or (
            original_candidates[0] if original_candidates else None
        )
        if not preferred:
            errors.append(
                f"Tarea {task.task_id}: no se pudo seleccionar vista "
                f"(retriever y fallback léxico fallaron)."
            )

    merged_candidates: List[str] = []
    seen: Set[str] = set()
    for cv in retriever_candidates + original_candidates:
        if cv and cv not in seen and cv in allowed_views:
            merged_candidates.append(cv)
            seen.add(cv)
    if preferred and preferred not in seen:
        merged_candidates.insert(0, preferred)

    if preferred:
        task.preferred_view = preferred
        task.candidate_views = merged_candidates
        lexical_errors = _validate_task_integrity(task, allowed_views)  # solo errores DUROS
        if lexical_errors:
            fallback = _find_compatible_view(task, allowed_views)
            if fallback:
                logger.info(f"[Planner] Fallback léxico a {fallback}")
                preferred = fallback
                task.preferred_view = preferred
                if preferred not in merged_candidates:
                    merged_candidates.insert(0, preferred)
                task.candidate_views = merged_candidates
            else:
                errors.extend([f"Tarea {task.task_id}: {err}" for err in lexical_errors])
        else:
            # (3.3) primero lo inequívoco (determinista, con traza); luego (2.8)
            # solo lo genuinamente ambiguo viaja como warning al SQL Agent.
            _align_task_columns(task)
            col_warnings = _collect_column_warnings(task)
            if col_warnings:
                logger.info(f"[Planner] Columnas ambiguas/inexistentes en tarea {task.task_id}: {col_warnings}")
                task.assumptions = [
                    *(getattr(task, "assumptions", []) or []),
                    "Verificación léxica: " + " | ".join(col_warnings),
                ]

    return preferred, merged_candidates, errors


# ------------------------------------------------------------------
# (2.7) Versionado de plan + normalización determinista de IDs
# (3.5): la versión deriva del conteo de replans del ledger — antes, al
# limpiarse plan/sql_results en cada replan, la versión quedaba siempre en 1.
# ------------------------------------------------------------------
def _next_plan_version(state: Dict[str, Any]) -> int:
    versions: List[int] = []
    prev_plan = state.get("plan")
    v = _get(prev_plan, "plan_version", None) if prev_plan else None
    if isinstance(v, int):
        versions.append(v)
    for r in (state.get("sql_results") or []):
        m = re.match(r"v(\d+)-", str(_get(r, "task_id", "") or ""))
        if m:
            versions.append(int(m.group(1)))
    # (3.5) el ledger recuerda los replans aunque plan/results se hayan limpiado
    progress = state.get("progress_ledger") or {}
    replans = list(_get(progress, "completed_steps", []) or []).count("replan")
    if replans:
        versions.append(replans)
    return (max(versions) + 1) if versions else 1


def _normalize_task_ids(plan: PlannerContract, plan_version: int) -> None:
    try:
        plan.plan_version = plan_version
    except Exception:
        logger.warning(
            "[Planner] PlannerContract no tiene campo plan_version — "
            "los IDs se versionan igualmente, pero el contador no persistirá."
        )

    old_to_new: Dict[str, str] = {}
    seen: Set[str] = set()
    for i, t in enumerate(plan.tasks, 1):
        old = str(getattr(t, "task_id", "") or "").strip() or str(i)
        new = f"v{plan_version}-t{i}"
        if old not in seen:
            old_to_new[old] = new
            seen.add(old)
        t.task_id = new

    for t in plan.tasks:
        dropped = [d for d in (t.depends_on or []) if d not in old_to_new]
        if dropped:
            logger.warning(f"[Planner] depends_on rotos descartados en {t.task_id}: {dropped}")
        t.depends_on = [old_to_new[d] for d in (t.depends_on or []) if d in old_to_new]


# ------------------------------------------------------------------
# Helper de salida: consumo de señales en TODAS las rutas
# ------------------------------------------------------------------
def _planner_result(
    plan: PlannerContract,
    log_msg: str,
    forecast_request: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "plan": plan,
        "last_agent": "planner",
        "messages": [AIMessage(content=log_msg)],
        "next_agent_instruction": None,  # instrucción ya consumida
        "is_replan": False,              # (2.2) señal ya consumida
        "replan_errors": None,           # (3.5) evidencia ya consumida
    }
    if forecast_request is not None or _get(plan, "question_type") == "demand_forecast":
        out["forecast_request"] = forecast_request
    return out


# ------------------------------------------------------------------
# Nodo principal
# ------------------------------------------------------------------
def planner_node(state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    harness = state.get("harness_context", {}) or {}
    allowed_views = harness.get("allowed_views", []) or []
    ambiguity_notes = harness.get("ambiguity_notes", []) or []
    question = state["question"]
    messages = state.get("messages", []) or []

    instruction = state.get("next_agent_instruction") or state.get("supervisor_instruction")

    is_replan = bool(state.get("is_replan", False))

    contextual_question = _build_conversational_context(messages, question)

    logger.info(f"[Planner] Pregunta contextual: {contextual_question[:200]}...")
    logger.info(f"[Planner] allowed_views={allowed_views}")
    logger.info(f"[Planner] ambiguity_notes={ambiguity_notes}")
    logger.info(f"[Planner] preferred_view={harness.get('preferred_view')} | is_replan={is_replan}")
    if instruction:
        logger.info(f"[Planner] Instrucción del supervisor: {instruction[:200]}...")

    # ----------------------------------------------------------------
    # Guard: sin vistas no hay plan posible
    # ----------------------------------------------------------------
    if not allowed_views:
        logger.error("[Planner] ERROR CRÍTICO: allowed_views está vacío.")
        return _planner_result(
            PlannerContract(
                intent="unknown", goal="", question_type="unknown",
                metrics=[], dimensions=[], filters="", tasks=[], confidence=0.0,
                needs_followup=True,
                followup_reason=(
                    "No hay vistas de datos disponibles para responder la pregunta. "
                    "Revisa AGENTS.md y el catálogo semántico."
                ),
            ),
            "[Planner] Error: no hay vistas de datos disponibles.",
        )

    # ================================================================
    # PRIORIDAD MÁXIMA: demand forecast
    # ================================================================
    if _is_demand_forecast_question(contextual_question):
        logger.info("[Planner] Detectada pregunta de demand forecast")

        if is_replan and state.get("forecast_error"):
            err = state["forecast_error"]
            logger.warning(f"[Planner] Replan tras error de forecast: {err}")
            return _planner_result(
                PlannerContract(
                    intent="demand_forecast", goal="",
                    question_type="demand_forecast",
                    metrics=[], dimensions=[], filters="", tasks=[],
                    confidence=0.3, visualization_candidate=False,
                    needs_followup=True,
                    followup_reason=(
                        f"El pronóstico falló previamente: {err}. "
                        f"Verifica que el producto y la sede existan o indica otros."
                    ),
                ),
                f"[Planner] Replan de forecast: se solicita clarificación (error previo: {err}).",
                forecast_request=None,
            )

        params = _extract_forecast_params(contextual_question)
        if not params.get("producto") or not params.get("sede"):
            params = _fallback_extract_forecast_params(contextual_question)

        if not params.get("producto") or not params.get("sede"):
            return _planner_result(
                PlannerContract(
                    intent="demand_forecast", goal="",
                    question_type="demand_forecast",
                    metrics=[], dimensions=[], filters="", tasks=[],
                    confidence=0.5, visualization_candidate=False,
                    needs_followup=True,
                    missing_information=["producto", "sede"],
                    followup_reason=(
                        "Necesito que especifiques el producto y la sede "
                        "para generar el pronóstico de demanda."
                    ),
                ),
                "[Planner] Predicción de demanda detectada. Necesito producto y sede.",
                forecast_request=None,
            )

        n_dias = _clamp_n_dias(params.get("n_dias", 7))
        plan = PlannerContract(
            intent="demand_forecast",
            goal=(
                f"Predecir la demanda diaria de {params['producto']} en "
                f"{params['sede']} para los próximos {n_dias} días"
            ),
            question_type="demand_forecast",
            metrics=["prediccion", "prediccion_con_buffer", "safety_stock"],
            dimensions=["fecha", "producto", "sede"],
            filters=f"producto={params['producto']}, sede={params['sede']}",
            time_window=f"next_{n_dias}_days",
            tasks=[], confidence=0.9,
            visualization_candidate=False, needs_followup=False,
        )
        return _planner_result(
            plan,
            f"[Planner] Demand forecast: {params['producto']} @ {params['sede']} | {n_dias} días",
            forecast_request={
                "producto": params["producto"],
                "sede": params["sede"],
                "n_dias": n_dias,
                "fecha_inicio": params.get("fecha_inicio"),
            },
        )

    # ================================================================
    # Corte temprano si hay ambigüedad de vista
    # ================================================================
    preferred_view = harness.get("preferred_view")
    if ambiguity_notes and not preferred_view:
        plan = PlannerContract(
            intent="unknown", goal="", question_type="unknown",
            metrics=[], dimensions=[], filters="", tasks=[], confidence=0.0,
            needs_followup=True,
            followup_reason=f"Ambigüedad en selección de vista: {'; '.join(ambiguity_notes)}",
        )
        return _planner_result(plan, f"[Planner] Ambigüedad detectada: {plan.followup_reason}")

    # ================================================================
    # Planificación normal de consultas SQL
    # ================================================================
    view_catalog = _build_view_catalog(allowed_views)   # (3.1) con definiciones
    dynamic_rules = _build_dynamic_rules(view_catalog)

    replan_context = ""
    if is_replan:
        replan_context = _build_replan_context(state)   # (3.5) incluye replan_errors
        logger.info(f"[Planner] Contexto de replanificación:\n{replan_context[:500]}...")

    replan_block = (
        f"\n{'=' * 60}\nCONTEXTO DE REPLANIFICACIÓN (el plan anterior NO funcionó):\n"
        f"{replan_context}\n{'=' * 60}\n"
        if is_replan else ""
    )

    system_prompt = f"""
Eres un Planner BI avanzado. Transformas preguntas de negocio en planes operacionales estructurados.

CATÁLOGO DE VISTAS PERMITIDAS (columnas con su significado):
{json.dumps(view_catalog, indent=2, ensure_ascii=False)}

REGLAS DERIVADAS DEL CATÁLOGO DISPONIBLE:
{dynamic_rules}

{"=" * 60}
INSTRUCCIÓN DEL SUPERVISOR:
{instruction or "Ninguna instrucción adicional. Planifica la pregunta del usuario de forma directa."}
{"=" * 60}
{replan_block}

REGLA DE PREDICCIÓN (FORECAST) - CRÍTICO:
- SOLO usa `question_type="demand_forecast"` si el usuario pide explícitamente PREDECIR EL FUTURO
  (ej. "pronosticar", "predecir ventas", "cómo van a ir las ventas la próxima semana", "demanda futura").
- NUNCA uses `demand_forecast` para consultas históricas. "Horas de mayor demanda",
  "Demanda por hora", "Volumen por hora" son consultas HISTÓRICAS (aggregation o multi_query).

REGLAS CRÍTICAS DE SELECCIÓN DE VISTA:
1. ANTES de asignar una vista a una tarea, verifica en el catálogo que contenga EXPLÍCITAMENTE
   las columnas que la tarea requiere (por su DEFINICIÓN, no solo por el nombre).
2. NUNCA asignes una vista si la columna requerida no aparece en sus métricas/columnas.
3. Desglose por PRODUCTO → solo vistas cuya definición exponga producto.
4. Desglose por SEDE/LOCAL → solo vistas cuya definición exponga sucursal/sede.
5. Desglose por CATEGORÍA → solo vistas cuya definición exponga categoría.
6. SEGURIDAD: usa ÚNICAMENTE vistas del catálogo. NO inventes vistas.
7. Respeta las REGLAS DERIVADAS DEL CATÁLOGO de arriba (temporalidad y granularidad).
8. En `metrics` y `dimensions`: elige la columna por su DEFINICIÓN en el catálogo y
   escribe su NOMBRE EXACTO. No existen columnas equivalentes ni sinónimos.

REGLAS DE MULTI-QUERY / DESCOMPOSICIÓN:
9. Si la pregunta compara métricas de distintas fuentes, cruza granularidades o requiere múltiples
   vistas, usa `question_type="multi_query"` con varios `SQLPayload` independientes.
10. Ejemplo: "ventas de café en Merced y además ticket promedio en Tajamar en junio" →
    tarea 1 (ventas café Merced) y tarea 2 (ticket promedio Tajamar junio).
11. Si una subtarea depende del resultado de otra, anota `depends_on` con los `task_id` previos.
12. NO DESCOMPONGAS si es la misma intención, granularidad y temporalidad.

CLARIFICACIÓN (needs_followup):
13. Si la pregunta es ambigua o le falta información insalvable (ej. sede o periodo no determinable),
    marca `needs_followup=true`, lista lo que falta en `missing_information` y explica en
    `followup_reason` qué debe aclarar el usuario.
14. NO marques followup si el plan es ejecutable con el catálogo disponible.

REGLAS DE NEGOCIO:
- "Se han vendido" / "ventas" / "unidades vendidas" → vistas de VENTAS NORMALES.
- "Canjes", "fidelización", "puntos" → vistas de FIDELIZACIÓN.
- "Cortesías", "gratis", "regalos" → vistas de CORTESÍA.

REPLANIFICACIÓN:
15. Si arriba hay CONTEXTO DE REPLANIFICACIÓN, genera un plan ALTERNO: cambia de vista, simplifica
    la pregunta o divide en subtareas más pequeñas. NO repitas el plan anterior NI sus errores
    técnicos (los "ERRORES TÉCNICOS DEL PLAN ANTERIOR" son prohibidos).

TOP N POR GRUPO:
16. "Top N por cada X" (ej. "5 productos más vendidos EN CADA sucursal") es UNA sola tarea
    con execution_strategy="top_n_per_group". NUNCA la dividas por sucursal, categoría u
    otro valor de la dimensión: el SQL Agent lo resuelve con ROW_NUMBER() OVER (PARTITION BY ...)
    en una query. Divide solo si cambia la métrica, la vista o el periodo.

REGLAS ADICIONALES:
- "informe completo", "reporte detallado", "análisis profundo", "deep dive" → question_type="deep_research".

OUTPUT: JSON con schema PlannerContract.
"""

    human = HumanMessage(content=f"Pregunta del usuario: {contextual_question}")
    plan_raw = PLANNER_LLM.invoke([SystemMessage(content=system_prompt), human])

    plan = PlannerContract(**plan_raw) if isinstance(plan_raw, dict) else plan_raw

    # ============================================================
    # (2.9) SAFEGUARD: forecast alucinado en query histórica
    # ============================================================
    if plan.question_type == "demand_forecast" and not _is_demand_forecast_question(contextual_question):
        logger.warning(
            "[Planner] LLM clasificó demand_forecast sin keywords. "
            "Corrigiendo a multi_query/aggregation y conservando las tasks generadas."
        )
        plan.question_type = "multi_query" if len(plan.tasks) > 1 else "aggregation"

    if plan.question_type == "demand_forecast":
        plan.tasks = []
        plan.visualization_candidate = False

    # ============================================================
    # (2.9) FALLBACK unificado: tarea genérica por el MISMO pipeline
    # ============================================================
    if not plan.tasks and plan.question_type not in ("demand_forecast", "unknown"):
        logger.warning("[Planner] LLM no generó tareas. Creando tarea fallback.")
        plan.tasks = [
            SQLPayload(
                task_id="1",
                task=f"Responder a la pregunta: {contextual_question}",
                metrics=[],
                dimensions=[],
                filters_description="",
                execution_strategy="single_view",
                candidate_views=list(allowed_views),
                preferred_view=None,
            )
        ]

    # ============================================================
    # (2.7)/(3.5) Versionado + IDs deterministas ANTES de validar
    # ============================================================
    plan_version = _next_plan_version(state)
    _normalize_task_ids(plan, plan_version)

    # ============================================================
    # VALIDACIÓN Y ENRIQUECIMIENTO SEMÁNTICO POST-LLM
    # ============================================================
    validation_errors: List[str] = []
    for task in plan.tasks:
        preferred, candidates, errs = _select_view_for_task(
            task, allowed_views, original_query=contextual_question
        )
        if errs:
            validation_errors.extend(errs)
        if preferred:
            task.preferred_view = preferred
            task.candidate_views = candidates
        else:
            validation_errors.append(f"Tarea {task.task_id}: no se pudo asignar ninguna vista.")

    # ============================================================
    # (2.6) FOLLOWUP: respetar al LLM + gate de confidence
    # ============================================================
    llm_followup = bool(plan.needs_followup)

    if validation_errors:
        plan.needs_followup = True
        plan.followup_reason = " | ".join(validation_errors)
        logger.warning(f"[Planner] Errores de validación: {validation_errors}")
    elif llm_followup and plan.missing_information:
        plan.needs_followup = True
        plan.followup_reason = plan.followup_reason or (
            "Información faltante: " + ", ".join(plan.missing_information)
        )
        logger.info(f"[Planner] Followup del LLM respetado: {plan.followup_reason}")
    elif plan.confidence < 0.5:
        plan.needs_followup = True
        plan.followup_reason = plan.followup_reason or (
            "Baja confianza en la interpretación de la pregunta; "
            "¿puedes reformularla con más detalle (sede, producto, periodo)?"
        )
        logger.info(f"[Planner] Followup por baja confianza ({plan.confidence})")
    else:
        plan.needs_followup = False
        plan.followup_reason = ""

    task_summary = " | ".join(f"{t.task_id}:{t.task[:60]}" for t in plan.tasks)
    prefs = [t.preferred_view for t in plan.tasks]

    return _planner_result(
        plan,
        f"[Planner] v{plan_version} | Intención: {plan.intent} | Tasks: {len(plan.tasks)} "
        f"({task_summary}) | Confianza: {plan.confidence} | Preferred: {prefs} | "
        f"Followup: {plan.needs_followup}",
    )
