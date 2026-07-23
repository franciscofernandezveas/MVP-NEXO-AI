"""
core/harness.py
"""

from __future__ import annotations

import logging
import re
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

from core.semantic_retriever import (
    obtener_candidatas_detalles,
    seleccionar_vista_principal,
    _detect_temporal_context,
)

logger = logging.getLogger(__name__)

DEFAULT_AGENTS_MD_PATH = Path(
    os.getenv("AGENTS_MD_PATH") or
    os.getenv("DEFAULT_AGENTS_MD_PATH") or
    Path(__file__).resolve().parent.parent / "AGENTS.md"
)


# ============================================================================
# Normalización de preguntas
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
        """
        Filtra títulos de sección del markdown que NO son vistas de base de datos.
        Ejemplos de falsas vistas: '1.1 Prioridad de Vistas', '4.2 Sintaxis de filtros de fecha'.
        """
        if not name or not name.strip():
            return False

        # Títulos de sección tipo "1.1 Algo", "4.2 Otra cosa"
        if re.match(r"^\d+(\.\d+)+\s+\w", name):
            return False

        # Nombres que parecen títulos de documentación: tienen espacios pero no guiones bajos
        # Las vistas reales usan snake_case o al menos no tienen espacios sueltos
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

                # NUEVO: saltar títulos de sección que no son vistas reales
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
        return self.views.get(view_name)

    def list_views(self) -> List[str]:
        return list(self.views.keys())

    def to_full_context(self) -> str:
        blocks = [v.to_context_block() for v in self.views.values()]
        return "\n\n".join(blocks)


# ============================================================================
# Seguridad dinámica basada en BusinessMemory
# ============================================================================

def is_view_allowed(view_name: str, biz_mem: BusinessMemory) -> bool:
    """
    Una vista es segura si existe documentada en AGENTS.md.
    Acepta nombres con o sin prefijo 'semantic.'.
    """
    if not view_name:
        return False
    clean = view_name.replace("semantic.", "").strip()
    return biz_mem.get_view(clean) is not None


def extract_views_from_sql(sql: str) -> Set[str]:
    """
    Extrae vistas referenciadas en una query SQL (solo esquema semantic).
    """
    if not sql:
        return set()
    pattern = r"(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)"
    matches = re.findall(pattern, sql, re.IGNORECASE)
    views = set()
    for match in matches:
        views.add(match.lower())
    return views


def validate_sql_views(sql: str, biz_mem: BusinessMemory) -> Tuple[bool, List[str]]:
    """
    Valida que todas las vistas usadas en el SQL estén documentadas.
    Devuelve (is_valid, list_of_errors).
    """
    used_views = extract_views_from_sql(sql)
    invalid = [v for v in used_views if not is_view_allowed(v, biz_mem)]
    return len(invalid) == 0, invalid


# ============================================================================
# Construcción de contexto con retriever vectorial
# ============================================================================

def build_harness_context(
    candidatas: List[Dict[str, Any]],
    biz_mem: Optional[BusinessMemory] = None,
    user_question: Optional[str] = None,
    include_rules: bool = True,
    rules_md: Optional[str] = None,
) -> str:
    """
    Construye el contexto semántico como string a partir de candidatas del retriever.
    """
    if biz_mem is None:
        biz_mem = BusinessMemory.from_file()

    default_rules = (
        "Eres un asistente BI que consulta el esquema 'semantic' en PostgreSQL (Supabase).\n"
        "Reglas de oro:\n"
        "- NO recalcular métricas que la vista ya expone.\n"
        "- Usar el formato 'YYYY-MM-DD' para filtros de fecha.\n"
        "- '_latest' / '_day' son snapshots actuales: NO filtrar por fecha.\n"
        "- '_history' / '_historical' requieren rango de fechas.\n"
        "- 'dashboard_*' traen periodos predefinidos; verificar antes de filtrar.\n"
        "- 'compare_periods' ya calcula la comparativa; no agregar filtros ni joins manuales.\n"
        "- Nunca inventar columnas. Si hay duda, validar con information_schema.columns."
    )

    semantic_parts: List[str] = []
    view_names: List[str] = []

    for c in candidatas:
        view_name = c.get("view_name") if isinstance(c, dict) else str(c)
        if not view_name:
            continue
        view_names.append(view_name)

        view_info = biz_mem.get_view(view_name)
        if view_info:
            semantic_parts.append(view_info.to_context_block())
        else:
            semantic_parts.append(
                f"Vista: {view_name}\n"
                "Descripción: (no documentada en AGENTS.md)\n"
                "Métricas: (no definidas)"
            )

    semantic_context = "\n\n".join(semantic_parts)

    # Notas de ambigüedad
    ambiguity_notes: List[str] = []
    if user_question and view_names:
        detector = AmbiguityDetector(biz_mem)
        ambiguity_notes = detector.analyze(user_question, view_names)

    if include_rules:
        rules_text = rules_md.strip() if rules_md else default_rules
        context = f"{rules_text}\n\n---\n\nMEMORIA SEMÁNTICA (vistas relevantes):\n\n{semantic_context}"
    else:
        context = f"MEMORIA SEMÁNTICA (vistas relevantes):\n\n{semantic_context}"

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
        q_norm = _normalize_question(user_question)

        snapshot_keywords = [
            "hoy", "ayer", "actual", "snapshot", "dia actual", "en curso"
        ]
        historical_keywords = [
            "mes", "ano", "anio", "historico", "tendencia", "rango", "ultimos",
            "semana pasada", "junio", "julio", "agosto", "septiembre", "octubre",
        ]

        asks_snapshot = any(k in q_norm for k in snapshot_keywords)
        asks_historical = any(k in q_norm for k in historical_keywords)

        for view_name in candidatas:
            view = self.biz_mem.get_view(view_name)
            if not view:
                continue

            filtro_norm = _normalize_question(view.filtro_fecha)
            tipo_norm = _normalize_question(view.tipo)

            is_snapshot = (
                "no aplica" in filtro_norm
                or "current" in tipo_norm
                or "snapshot" in tipo_norm
                or "compare periods" in tipo_norm
            )
            needs_history = "rango" in filtro_norm or "historical" in tipo_norm

            if asks_snapshot and needs_history and not asks_historical:
                notes.append(
                    f"La pregunta parece pedir datos de hoy/ayer, pero `{view_name}` "
                    "requiere un rango de fechas histórico. Considera una vista `_latest` o `_day`."
                )

            if asks_historical and is_snapshot and not asks_snapshot:
                notes.append(
                    f"La pregunta parece pedir histórico, pero `{view_name}` es un snapshot "
                    "sin filtro de fecha. Considera su versión `_history`."
                )

        return notes


# ============================================================================
# Fallback por keywords para elegir vista principal
# ============================================================================

def _fallback_preferred_view(question: str, allowed_views: List[str]) -> Optional[str]:
    """
    Si el retriever no elige una vista principal, usamos reglas de keyword simples.
    """
    q = _normalize_question(question)

    # Mapeo de intención a vistas candidatas (en orden de preferencia)
    keyword_map = [
        (["venta", "vendido", "vendidos", "unidades vendidas", "ingreso"], [
            "sales_producto_daily",
            "sales_review_day",
            "sales_review_day_history",
            "kpi_categorias_diario",
            "sales_week",
        ]),
        (["producto", "categoria", "categoría", "familia", "articulo", "artículo"], [
            "sales_producto_daily",
            "kpi_categorias_productos_sede",
            "kpi_categorias_diario",
        ]),
        (["sede", "local", "tienda", "sucursal", "plaza"], [
            "sales_review_locales",
            "sales_review_locales_latest",
            "kpi_categorias_productos_sede",
        ]),
        (["fidelizacion", "canje", "puntos", "fidelización"], [
            "kpi_fidelizacion_detalle",
            "dashboard_canjes_resumen",
        ]),
        (["cortesia", "cortesía", "gratis", "regalo"], [
            "kpi_cortesia_detalle",
            "dashboard_cortesias_resumen",
        ]),
        (["hora", "horario", "momento del dia", "picos"], [
            "mart_operacion_hora",
        ]),
    ]

    for keywords, candidate_views in keyword_map:
        if any(k in q for k in keywords):
            for cv in candidate_views:
                if cv in allowed_views:
                    return cv

    # Fallback general: primera vista de ventas, o la primera permitida
    sales_views = [v for v in allowed_views if "sales" in v or "kpi" in v]
    if sales_views:
        return sales_views[0]

    return allowed_views[0] if allowed_views else None


# ============================================================================
# Función cacheada que espera el orquestador
# ============================================================================

@lru_cache(maxsize=128)
def _build_harness_context_cached_impl(normalized_question: str) -> Dict[str, Any]:
    """
    Implementación cacheada. Usa el retriever vectorial real.
    """
    biz_mem = BusinessMemory.from_file()

    # Recuperar candidatas del retriever vectorial con k amplio
    candidatas_detalle = obtener_candidatas_detalles(
        query=normalized_question,
        k=10,
        allowed_views=None,
        min_score_threshold=-999.0,
        column_hints={"original_query": normalized_question},
    )

    # Si no hay candidatas, fallback a todas las vistas documentadas (ahora solo reales)
    if not candidatas_detalle:
        logger.warning("Retriever no devolvió candidatas. Fallback a todas las vistas de AGENTS.md")
        candidatas_detalle = [
            {
                "view_name": v,
                "score": 0.0,
                "original_score": 0.0,
                "metadata_boost": 0.0,
            }
            for v in biz_mem.list_views()
        ]

    # Determinar vista principal
    vista_principal_obj = seleccionar_vista_principal(
        query=normalized_question,
        column_hints={"original_query": normalized_question},
        allowed_views=None,
        min_score_threshold=-999.0,
    )
    preferred_view = vista_principal_obj["view_name"] if vista_principal_obj else None

    # allowed_views = todas las candidatas recuperadas + todas las vistas documentadas reales
    allowed_views_set: Set[str] = set()
    for c in candidatas_detalle:
        vn = c.get("view_name")
        if vn:
            allowed_views_set.add(vn)

    # Asegurar que la vista principal esté incluida
    if preferred_view and preferred_view not in allowed_views_set:
        allowed_views_set.add(preferred_view)

    # Asegurar que todas las vistas documentadas reales estén permitidas
    for documented_view in biz_mem.list_views():
        allowed_views_set.add(documented_view)

    allowed_views = sorted(allowed_views_set)

    # NUEVO: si no hay vista principal, usar fallback por keywords
    if not preferred_view:
        preferred_view = _fallback_preferred_view(normalized_question, allowed_views)
        logger.info(f"[Harness] Fallback preferred_view: {preferred_view}")

    # Construir contexto semántico solo para las top candidatas
    top_candidates = candidatas_detalle[:5] if len(candidatas_detalle) >= 5 else candidatas_detalle

    # NUEVO: si las top candidates están vacías o son malas, usar la preferred_view
    if not top_candidates and preferred_view:
        top_candidates = [{"view_name": preferred_view, "score": 0.0}]

    semantic_context = build_harness_context(
        candidatas=top_candidates,
        biz_mem=biz_mem,
        user_question=normalized_question,
        include_rules=True,
    )

    # NUEVO: ambigüedad solo sobre top candidates, no sobre todas las vistas
    ambiguity_views = [c.get("view_name") for c in top_candidates if c.get("view_name")]
    if preferred_view and preferred_view not in ambiguity_views:
        ambiguity_views.append(preferred_view)

    detector = AmbiguityDetector(biz_mem)
    ambiguity_notes = detector.analyze(normalized_question, ambiguity_views)

    return {
        "semantic_context": semantic_context,
        "allowed_views": [f"semantic.{v}" for v in allowed_views],
        "preferred_view": f"semantic.{preferred_view}" if preferred_view else None,
        "ambiguity_notes": ambiguity_notes,
        "all_documented_views": [f"semantic.{v}" for v in biz_mem.list_views()],
    }


def build_harness_context_cached(question: str) -> Dict[str, Any]:
    normalized = _normalize_question(question)
    return _build_harness_context_cached_impl(normalized)


# ============================================================================
# Ejemplo de uso
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    pregunta = "¿Cuántos canjes de fidelización hubo por sede el mes pasado?"
    context_pack = build_harness_context_cached(pregunta)

    print("preferred_view:", context_pack["preferred_view"])
    print("allowed_views:", context_pack["allowed_views"])
    print("ambiguity_notes:", context_pack["ambiguity_notes"])
    print("\n--- SEMANTIC CONTEXT ---\n")
    print(context_pack["semantic_context"][:2000], "...")
