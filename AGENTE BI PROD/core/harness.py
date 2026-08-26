# core/harness.py
# -------------------------------------------------
# Harness alineado al RAG (recuperación única).
#
# Cambios frente a la versión anterior:
#  - El RAG ya parseó, clasificó y enriqueció AGENTS.md. El harness usa
#    directamente el contexto recuperado (c["context"]) en lugar de reconstruir
#    un bloque paralelo desde BusinessMemory.
#  - Se inyecta conocimiento de negocio relevante (reglas, taxonomía,
#    definiciones de métricas) vía buscar_conocimiento_negocio().
#  - La caché del contexto compuesto depende de la firma del manifiesto del RAG,
#    no solo del sha de AGENTS.md. Si el indexer reindexa (cambio de embeddings,
#    chunker, etc.), el harness invalida su caché automáticamente.
#  - BusinessMemory sigue disponible como utilidad de post-proceso:
#    detección temporal, promoción histórica, resolución de columnas y
#    validación de vistas usadas en SQL.
# -------------------------------------------------

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

from core.rag import obtener_candidatas_detalles, buscar_conocimiento_negocio
from core.rag_store import read_manifest, CHROMA_DIR, COLLECTION_NAME

logger = logging.getLogger(__name__)

DEFAULT_AGENTS_MD_PATH = Path(
    os.getenv("AGENTS_MD_PATH") or
    os.getenv("DEFAULT_AGENTS_MD_PATH") or
    Path(__file__).resolve().parent.parent / "AGENTS.md"
)

# R2: umbral coseno ∈ [0,1]. Calibrar con el golden set (hit@k) del RAG.
MIN_SCORE_THRESHOLD = float(os.getenv("HARNESS_MIN_SCORE", "0.20"))
MAX_FALLBACK_VIEWS = 3
MIN_CONTEXT_VIEWS = 3


# ============================================================================
# Normalización de preguntas y nombres de vistas
# ============================================================================

def _normalize_question(question: str) -> str:
    """Normaliza la pregunta para caché y keyword matching."""
    if not question:
        return ""
    q = question.lower().strip()
    replacements = [
        ("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u"),
        ("ñ", "n"), ("ü", "u"),
    ]
    for orig, repl in replacements:
        q = q.replace(orig, repl)
    q = re.sub(r"[¿?!¡.,;:\"'`()\[\]{}]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q


def _clean_view_name(view_name: str) -> str:
    """Elimina el prefijo 'semantic.' si existe."""
    if not view_name:
        return ""
    return view_name.lower().replace("semantic.", "").strip()


def _with_semantic_prefix(view_name: str) -> Optional[str]:
    """Garantiza que el nombre de vista tenga el prefijo 'semantic.'."""
    if not view_name:
        return None
    clean = _clean_view_name(view_name)
    return f"semantic.{clean}"


# ============================================================================
# (H1) Intención temporal estructurada — criterio ÚNICO compartido
# ============================================================================

_SNAPSHOT_KEYWORDS = ["hoy", "ayer", "actual", "snapshot", "dia actual", "en curso"]

_HISTORICAL_KEYWORDS = [
    "mes", "ano", "anio", "historico", "historial", "tendencia", "rango", "ultimos",
    "semana pasada", "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
    "agosto", "septiembre", "octubre", "noviembre", "diciembre", "trimestre", "semestre",
]

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _temporal_intent(question: str) -> Optional[str]:
    """'historical' | 'snapshot' | None."""
    q = _normalize_question(question)
    asks_hist = any(k in q for k in _HISTORICAL_KEYWORDS) or bool(_YEAR_RE.search(q))
    asks_snap = any(k in q for k in _SNAPSHOT_KEYWORDS)
    if asks_hist and not asks_snap:
        return "historical"
    if asks_snap and not asks_hist:
        return "snapshot"
    return None


def _is_snapshot_view(view_info: Optional["ViewInfo"]) -> bool:
    if not view_info:
        return False
    tipo = _normalize_question(view_info.tipo or "")
    filtro = _normalize_question(view_info.filtro_fecha or "")
    return (
        "current" in tipo
        or "snapshot" in tipo
        or "compare periods" in tipo
        or "no aplica" in filtro
    )


def _is_historical_view(view_info: Optional["ViewInfo"]) -> bool:
    if not view_info:
        return False
    tipo = _normalize_question(view_info.tipo or "")
    filtro = _normalize_question(view_info.filtro_fecha or "")
    return "historical" in tipo or "rango" in filtro


def _historical_variants_for(view_name: str, biz_mem: "BusinessMemory") -> List[str]:
    """
    Deriva candidatas históricas documentadas a partir de una vista snapshot,
    usando la CONVENCIÓN de nombres consultada contra el catálogo.
    """
    base = _clean_view_name(view_name)
    for suffix in ("_latest", "_current", "_day"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    candidatas = [f"{base}_history", f"{base}_historical", f"{_clean_view_name(view_name)}_history"]
    return [c for c in dict.fromkeys(candidatas) if biz_mem.get_view(c)]


def _ordered_documented_views(biz_mem: "BusinessMemory", temporal: Optional[str]) -> List[str]:
    """
    Vistas documentadas, priorizando las compatibles con la intención temporal.
    sorted() es estable: dentro de cada grupo se conserva el orden del archivo.
    """
    documented = biz_mem.list_views()
    if temporal == "historical":
        documented = sorted(documented, key=lambda v: not _is_historical_view(biz_mem.get_view(v)))
    elif temporal == "snapshot":
        documented = sorted(documented, key=lambda v: not _is_snapshot_view(biz_mem.get_view(v)))
    return documented


# ============================================================================
# ViewInfo y BusinessMemory
# ============================================================================

@dataclass
class ViewInfo:
    nombre: str
    descripcion: str = ""
    tipo: str = ""
    granularidad: str = ""
    filtro_fecha: str = ""
    columnas_fecha: List[str] = field(default_factory=list)
    metricas: Dict[str, str] = field(default_factory=dict)
    notas: List[str] = field(default_factory=list)

    def to_context_block(self) -> str:
        lineas = [f"Vista: {self.nombre}"]

        for campo, etiqueta in [
            (self.tipo, "Tipo"),
            (self.descripcion, "Descripción"),
            (self.granularidad, "Granularidad"),
            (self.filtro_fecha, "Filtro de fecha"),
        ]:
            if campo:
                lineas.append(f"{etiqueta}: {campo}")

        if self.columnas_fecha:
            lineas.append(f"Columnas de fecha: {', '.join(self.columnas_fecha)}")

        if self.metricas:
            lineas.append("Métricas:")
            for nombre_metrica, definicion in self.metricas.items():
                lineas.append(f"  - {nombre_metrica}: {definicion}")

        if self.notas:
            lineas.append("Notas:")
            for nota in self.notas:
                lineas.append(f"  - {nota}")

        return "\n".join(lineas)

    def has_metric(self, metric_name: str) -> bool:
        return metric_name.lower() in {k.lower() for k in self.metricas.keys()}


class BusinessMemory:
    def __init__(self, raw_md: str):
        self.raw_md = raw_md
        self.views: Dict[str, ViewInfo] = {}
        self._parse()

    @classmethod
    def from_file(cls, path: Path | str = DEFAULT_AGENTS_MD_PATH) -> "BusinessMemory":
        ruta = Path(path)
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontró {ruta.resolve()}")
        return cls(ruta.read_text(encoding="utf-8"))

    @staticmethod
    def _is_valid_view_name(name: str) -> bool:
        if not name or not name.strip():
            return False
        if re.match(r"^\d+(\.\d+)+\s+\w", name):
            return False
        if " " in name and "_" not in name:
            return False
        return True

    def _parse(self):
        lines = self.raw_md.splitlines()
        current_view: Optional[ViewInfo] = None
        section: Optional[str] = None

        for raw_line in lines:
            line = raw_line.rstrip()

            if line.startswith("### "):
                nombre_vista = line.replace("### ", "").strip()
                if not self._is_valid_view_name(nombre_vista):
                    current_view = None
                    section = None
                    continue

                current_view = ViewInfo(nombre=nombre_vista)
                self.views[nombre_vista] = current_view
                section = None
                continue

            if not current_view:
                continue

            if not line.strip():
                continue

            if self._is_section_header(line, "Métricas"):
                section = "metricas"
                continue

            if self._is_section_header(line, "Notas"):
                section = "notas"
                continue

            if kv := self._extract_kv(line, "Descripción"):
                current_view.descripcion = kv
                section = None
                continue

            if kv := self._extract_kv(line, "Tipo"):
                current_view.tipo = kv
                section = None
                continue

            if kv := self._extract_kv(line, "Granularidad"):
                current_view.granularidad = kv
                section = None
                continue

            if kv := self._extract_kv(line, "Filtro de fecha"):
                current_view.filtro_fecha = kv
                section = None
                continue

            if kv := self._extract_kv(line, "Columnas de fecha"):
                current_view.columnas_fecha = [c.strip() for c in kv.split(",") if c.strip()]
                section = None
                continue

            if section == "metricas":
                metrica = self._extract_list_item_backtick(line)
                if metrica:
                    nombre_metrica, definicion = metrica
                    current_view.metricas[nombre_metrica] = definicion
                continue

            if section == "notas":
                nota = self._extract_plain_list_item(line)
                if nota:
                    current_view.notas.append(nota)
                continue

            section = None

    @staticmethod
    def _is_section_header(line: str, section_name: str) -> bool:
        pattern = rf"-\s*\*\*{re.escape(section_name)}\*\*\s*:"
        return bool(re.match(pattern, line.strip()))

    @staticmethod
    def _extract_kv(line: str, field_name: str) -> Optional[str]:
        pattern = rf"-\s*\*\*{re.escape(field_name)}\*\*\s*:\s*(.*)"
        match = re.match(pattern, line.strip())
        return match.group(1).strip() if match else None

    @staticmethod
    def _extract_list_item_backtick(line: str) -> Optional[Tuple[str, str]]:
        match = re.match(r"\s*-\s*`([^`]+)`\s*:\s*(.+)", line)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return None

    @staticmethod
    def _extract_plain_list_item(line: str) -> Optional[str]:
        match = re.match(r"\s*-\s*(.+)", line)
        return match.group(1).strip() if match else None

    def get_view(self, view_name: str) -> Optional[ViewInfo]:
        return self.views.get(_clean_view_name(view_name))

    def list_views(self) -> List[str]:
        return list(self.views.keys())

    def to_full_context(self) -> str:
        blocks = [v.to_context_block() for v in self.views.values()]
        return "\n\n".join(blocks)


# ============================================================================
# (A1) Instancia compartida del catálogo
# ============================================================================

def _agents_md_signature() -> str:
    """Firma del archivo AGENTS.md. Clavea el caché de BusinessMemory."""
    try:
        return hashlib.sha256(DEFAULT_AGENTS_MD_PATH.read_bytes()).hexdigest()[:16]
    except Exception:
        return "na"


@lru_cache(maxsize=8)
def _load_biz_mem(sig: str) -> BusinessMemory:
    logger.info(f"[Harness] Cargando BusinessMemory (sig={sig})")
    return BusinessMemory.from_file()


def get_biz_mem() -> BusinessMemory:
    return _load_biz_mem(_agents_md_signature())


# ============================================================================
# (A3) Resolución inequívoca concepto→columna
# ============================================================================

def _normalize_column_text(text: str) -> str:
    if not text:
        return ""
    return (
        text.lower().strip().replace("_", " ")
        .replace("á", "a").replace("é", "e").replace("í", "i")
        .replace("ó", "o").replace("ú", "u")
    )


def resolve_column_unambiguous(concept: str, available: List[str]) -> Optional[str]:
    tokens = {t for t in _normalize_column_text(concept).split() if len(t) > 2}
    if not tokens:
        return None
    matches = [
        col for col in available
        if tokens <= set(_normalize_column_text(col).split())
        or set(_normalize_column_text(col).split()) <= tokens
    ]
    return matches[0] if len(matches) == 1 else None


# ============================================================================
# Seguridad dinámica basada en BusinessMemory
# ============================================================================

def is_view_allowed(view_name: str, biz_mem: BusinessMemory) -> bool:
    if not view_name:
        return False
    return biz_mem.get_view(view_name) is not None


def extract_views_from_sql(sql: str) -> Set[str]:
    if not sql:
        return set()
    pattern = r"(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)"
    matches = re.findall(pattern, sql, re.IGNORECASE)
    return set(matches)


def validate_sql_views(sql: str, biz_mem: BusinessMemory) -> Tuple[bool, List[str]]:
    used_views = extract_views_from_sql(sql)
    invalid = [v for v in used_views if not is_view_allowed(v, biz_mem)]
    return len(invalid) == 0, invalid


# ============================================================================
# Construcción de contexto semántico ALINEADA AL RAG
# ============================================================================

def build_harness_context(
    candidatas: List[Dict[str, Any]],
    biz_mem: Optional[BusinessMemory] = None,
    user_question: Optional[str] = None,
    include_rules: bool = True,
    rules_md: Optional[str] = None,
    business_knowledge: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    El RAG decide QUÉ vistas son relevantes Y nos entrega su contexto enriquecido
    (incluye definición, métricas, reglas y taxonomía asociada en c["context"]).
    El harness solo estructura el prompt final.

    BusinessMemory actúa como fallback/normalización cuando un candidato no
    trae contexto (p. ej. fallback al catálogo documentado).
    """
    if biz_mem is None:
        biz_mem = get_biz_mem()

    default_rules = (
        "Eres un asistente BI que consulta el esquema 'semantic' en PostgreSQL (Supabase).\n"
        "Reglas de oro:\n"
        "- NO recalcular métricas que la vista ya expone; usa la columna directamente.\n"
        "- Usar el formato 'YYYY-MM-DD' para filtros de fecha.\n"
        "- '_latest' / '_day' / 'current' son snapshots del día en curso: NO filtrar por fecha.\n"
        "- '_history' / '_historical' requieren rango de fechas obligatorio.\n"
        "- 'dashboard_*' traen periodos predefinidos; verificar si fijan el periodo.\n"
        "- 'compare_periods' ya calcula la comparativa; no agregar filtros ni joins manuales.\n"
        "- Si el usuario menciona un rango explícito (mes, año, 'últimos X días', 'ayer'), "
        "usa obligatoriamente una vista historical; PROHIBIDO usar _latest/_day/_week.\n"
        "- Métricas de transacciones (transacciones, ventas_hoy) vienen de fact_transacciones; "
        "métricas de productos/unidades vienen de fact_ventas.\n"
        "- Nunca inventar columnas. Si hay duda, validar con information_schema.columns."
    )

    semantic_parts: List[str] = []
    view_names: List[str] = []

    for c in candidatas:
        view_name = c.get("view_name") if isinstance(c, dict) else str(c)
        if not view_name:
            continue
        view_name_clean = _clean_view_name(view_name)
        view_names.append(view_name_clean)

        # ✅ USAR EL CONTEXTO ENRIQUECIDO DEL RAG si existe
        rag_context = c.get("context") if isinstance(c, dict) else None
        if rag_context:
            semantic_parts.append(rag_context.strip())
            continue

        # Fallback al BusinessMemory (p. ej. cuando candidatas vienen del catálogo documentado)
        view_info = biz_mem.get_view(view_name_clean)
        if view_info:
            semantic_parts.append(view_info.to_context_block())
        else:
            semantic_parts.append(
                f"Vista: {view_name_clean}\n"
                "Descripción: (no documentada en AGENTS.md)\n"
                "Métricas: (no definidas)"
            )

    semantic_context = "\n\n".join(semantic_parts)

    # ✅ Inyectar conocimiento de negocio recuperado (reglas, taxonomía, definiciones)
    knowledge_block = ""
    if business_knowledge:
        kb_parts = [
            item.get("content", "")
            for item in business_knowledge
            if item.get("content")
        ]
        if kb_parts:
            knowledge_block = (
                "CONOCIMIENTO DE NEGOCIO RELEVANTE (reglas, taxonomía, definiciones):\n\n"
                + "\n\n".join(kb_parts)
            )

    ambiguity_notes: List[str] = []
    if user_question and view_names:
        detector = AmbiguityDetector(biz_mem)
        ambiguity_notes = detector.analyze(user_question, view_names)

    rules_text = rules_md.strip() if rules_md else default_rules

    sections: List[str] = [rules_text]
    if knowledge_block:
        sections.append(knowledge_block)
    sections.append(f"MEMORIA SEMÁNTICA (vistas relevantes):\n\n{semantic_context}")

    context = "\n\n---\n\n".join(sections)

    if ambiguity_notes:
        notes_block = "\n".join(f"- {n}" for n in ambiguity_notes)
        context += f"\n\nNOTAS DE AMBIGÜEDAD:\n\n{notes_block}"

    return context


# ============================================================================
# AmbiguityDetector
# ============================================================================

class AmbiguityDetector:
    def __init__(self, biz_mem: BusinessMemory):
        self.biz_mem = biz_mem

    def analyze(self, user_question: str, candidatas: List[str]) -> List[str]:
        notes: List[str] = []
        notes.extend(self._metric_ambiguity(user_question, candidatas))
        notes.extend(self._temporal_ambiguity(user_question, candidatas))
        return notes

    def _metric_ambiguity(self, user_question: str, candidatas: List[str]) -> List[str]:
        notes: List[str] = []
        q_norm = _normalize_question(user_question)

        requested_metrics: Set[str] = set()
        for view_name in candidatas:
            view = self.biz_mem.get_view(view_name)
            if not view:
                continue
            for metric in view.metricas.keys():
                variants = {metric, metric.replace("_", " ")}
                if any(var in q_norm for var in variants):
                    requested_metrics.add(metric)

        if not requested_metrics:
            return notes

        for view_name in candidatas:
            view = self.biz_mem.get_view(view_name)
            if not view:
                continue

            for metric in requested_metrics:
                if view.has_metric(metric):
                    notes.append(
                        f"Usar el campo expuesto `{metric}` de `{view_name}`; "
                        "NO recalcularlo manualmente."
                    )
                else:
                    other_views = [
                        other
                        for other in candidatas
                        if other != view_name
                        and self.biz_mem.get_view(other)
                        and self.biz_mem.get_view(other).has_metric(metric)
                    ]
                    if other_views:
                        notes.append(
                            f"La métrica `{metric}` no está en `{view_name}`; "
                            f"considera usar {', '.join(f'`{v}`' for v in other_views)}."
                        )

        return notes

    def _temporal_ambiguity(self, user_question: str, candidatas: List[str]) -> List[str]:
        notes: List[str] = []
        intent = _temporal_intent(user_question)

        for view_name in candidatas:
            view = self.biz_mem.get_view(view_name)
            if not view:
                continue

            if intent == "snapshot" and _is_historical_view(view):
                notes.append(
                    f"La pregunta parece pedir datos de hoy/ayer, pero `{view_name}` "
                    "requiere un rango de fechas histórico. Considera una vista `_latest` o `_day`."
                )

            if intent == "historical" and _is_snapshot_view(view):
                notes.append(
                    f"La pregunta parece pedir histórico, pero `{view_name}` es un snapshot "
                    "sin filtro de fecha. Considera su versión `_history`."
                )

        return notes


# ============================================================================
# Firma del manifiesto del RAG — para hot-reload coherente
# ============================================================================

def _rag_manifest_sig() -> str:
    """
    Firma del manifiesto generado por indexer.py.
    Cuando el indexer reindexa, este valor cambia y el caché del harness
    se invalida automáticamente.
    """
    try:
        m = read_manifest(CHROMA_DIR, COLLECTION_NAME)
        if not m:
            return "no_manifest"
        return hashlib.sha1(json.dumps(m, sort_keys=True).encode()).hexdigest()[:16]
    except Exception:
        return "no_manifest"


# ============================================================================
# Contexto compilado por pregunta (cacheado)
# ============================================================================

@lru_cache(maxsize=128)
def _build_harness_context_cached_impl(
    normalized_question: str,
    md_sig: str,
    rag_sig: str,
) -> Dict[str, Any]:
    """
    Compila el paquete de contexto para una pregunta.
    La caché depende de:
      - la pregunta normalizada,
      - la firma de AGENTS.md,
      - la firma del manifiesto del RAG (reindex externo invalida el caché).
    """
    logger.info(f"[Harness] Compilando contexto (md_sig={md_sig}, rag_sig={rag_sig})")
    biz_mem = get_biz_mem()

    # H1/H2: intención temporal computada UNA vez
    temporal = _temporal_intent(normalized_question)
    capability_gap_historical = False

    # -------------------------------------------------------------- R1 + R2 + R3
    try:
        hits = obtener_candidatas_detalles(
            query=normalized_question,
            k=10,
            allowed_views=None,
            min_score_threshold=0.0,  # traer todo; el umbral se aplica abajo
        )
    except Exception as e:
        logger.warning(f"[Harness] Retriever falló: {e}")
        hits = []

    strong = [h for h in hits if h.get("score", 0.0) >= MIN_SCORE_THRESHOLD]

    if strong:
        candidatas_detalle: List[Dict[str, Any]] = strong
    elif hits:
        logger.warning(
            f"[Harness] Sin matches sobre umbral {MIN_SCORE_THRESHOLD}; "
            f"usando top-{MAX_FALLBACK_VIEWS} sin umbral como fallback."
        )
        candidatas_detalle = hits[:MAX_FALLBACK_VIEWS]
    else:
        logger.warning("[Harness] Retriever vacío; usando catálogo como último recurso.")
        documented = _ordered_documented_views(biz_mem, temporal)
        candidatas_detalle = [
            {
                "view_name": v,
                "score": 0.0,
                "original_score": 0.0,
                "metadata_boost": 0.0,
                "can_answer": True,
            }
            for v in documented[:MAX_FALLBACK_VIEWS]
        ]

    preferred_view: Optional[str] = (
        _clean_view_name(candidatas_detalle[0]["view_name"])
        if candidatas_detalle else None
    )

    # -------------------------------------------------------------- allowed_views
    allowed_views_set: Set[str] = set()
    for c in candidatas_detalle:
        vn = c.get("view_name")
        if vn:
            allowed_views_set.add(_clean_view_name(vn))

    if preferred_view:
        allowed_views_set.add(_clean_view_name(preferred_view))

    if len(allowed_views_set) < 2:
        for v in _ordered_documented_views(biz_mem, temporal):
            if len(allowed_views_set) >= MIN_CONTEXT_VIEWS:
                break
            allowed_views_set.add(v)

    allowed_views_clean = sorted(allowed_views_set)

    # -------------------------------------------------------------- H2 Promoción histórica
    if temporal == "historical":
        has_historical = any(
            _is_historical_view(biz_mem.get_view(v)) for v in allowed_views_clean
        )
        if not has_historical:
            promoted: List[str] = []
            for v in list(allowed_views_clean):
                promoted.extend(
                    hv for hv in _historical_variants_for(v, biz_mem)
                    if is_view_allowed(hv, biz_mem)
                )
            promoted = [p for p in dict.fromkeys(promoted)]

            if promoted:
                allowed_views_clean = sorted(set(allowed_views_clean) | set(promoted))
                preferred_view = promoted[0]
                logger.info(f"[Harness] Vistas históricas promovidas al allowlist: {promoted}")
                candidatas_detalle = [
                    {"view_name": preferred_view, "score": 1.0, "can_answer": True}
                ] + [
                    c for c in candidatas_detalle
                    if _clean_view_name(c.get("view_name")) != preferred_view
                ]
            else:
                capability_gap_historical = True
                logger.warning(
                    "[Harness] Pregunta histórica sin vista _history/_historical documentada. "
                    "capability_gap=True"
                )

    # -------------------------------------------------------------- Conocimiento de negocio
    business_knowledge: List[Dict[str, Any]] = []
    try:
        business_knowledge = buscar_conocimiento_negocio(
            query=normalized_question,
            k=6,
            min_score=0.0,
        )
    except Exception as e:
        logger.warning(f"[Harness] buscar_conocimiento_negocio falló: {e}")

    # -------------------------------------------------------------- Contexto semántico
    top_candidates = candidatas_detalle[:5]

    if not top_candidates and preferred_view:
        top_candidates = [{"view_name": preferred_view, "score": 0.0, "can_answer": True}]

    semantic_context = build_harness_context(
        candidatas=top_candidates,
        biz_mem=biz_mem,
        user_question=normalized_question,
        include_rules=True,
        business_knowledge=business_knowledge,
    )

    # Ambigüedad solo sobre vistas relevantes
    ambiguity_views = [
        _clean_view_name(c.get("view_name"))
        for c in top_candidates
        if c.get("view_name")
    ]
    if preferred_view and preferred_view not in ambiguity_views:
        ambiguity_views.append(preferred_view)

    if temporal == "historical" and not capability_gap_historical:
        ambiguity_views = [
            v for v in ambiguity_views
            if not _is_snapshot_view(biz_mem.get_view(v))
        ] or ambiguity_views

    detector = AmbiguityDetector(biz_mem)
    ambiguity_notes = detector.analyze(normalized_question, ambiguity_views)

    if capability_gap_historical:
        ambiguity_notes.append(
            "La pregunta requiere un periodo histórico, pero NINGUNA vista documentada "
            "ofrece histórico; las fuentes actuales son snapshots del día en curso."
        )

    return {
        "semantic_context": semantic_context,
        "allowed_views": [_with_semantic_prefix(v) for v in allowed_views_clean],
        "preferred_view": _with_semantic_prefix(preferred_view),
        "ambiguity_notes": ambiguity_notes,
        "capability_gap": capability_gap_historical,
        "temporal_intent": temporal,
        "all_documented_views": [_with_semantic_prefix(v) for v in biz_mem.list_views()],
    }


def build_harness_context_cached(question: str) -> Dict[str, Any]:
    return _build_harness_context_cached_impl(
        _normalize_question(question),
        _agents_md_signature(),
        _rag_manifest_sig(),
    )


# ============================================================================
# Ejemplo de uso
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    pregunta = "productos mas vendidos en merced en junio de 2026"
    context_pack = build_harness_context_cached(pregunta)

    print("temporal_intent:", context_pack["temporal_intent"])
    print("capability_gap:", context_pack["capability_gap"])
    print("preferred_view:", context_pack["preferred_view"])
    print("allowed_views:", context_pack["allowed_views"])
    print("ambiguity_notes:", context_pack["ambiguity_notes"])
    print("\n--- SEMANTIC CONTEXT ---\n")
    print(context_pack["semantic_context"][:2500], "...")
