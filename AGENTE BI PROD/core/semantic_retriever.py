# core/semantic_retriever.py
# -------------------------------------------------
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import re
from datetime import datetime
import os

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db")))
COLLECTION_NAME = "semantic_views"

_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
_vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=_embeddings,
    persist_directory=str(CHROMA_DIR),
)

# ----------------------------------------------------------------------------
# FIX 2: Health probe al importar (Fail loud / Warn si el índice está vacío)
# ----------------------------------------------------------------------------
try:
    _DOC_COUNT = _vector_store._collection.count()
except Exception:
    _DOC_COUNT = -1

if _DOC_COUNT <= 0:
    logger.warning(
        f"[Retriever] ⚠️ Colección '{COLLECTION_NAME}' vacía o inaccesible en {CHROMA_DIR}. "
        f"La selección semántica está degradada: el sistema operará solo con fallback léxico."
    )
else:
    logger.info(f"[Retriever] Colección '{COLLECTION_NAME}' lista: {_DOC_COUNT} vistas indexadas.")


# ============================================================================
# Normalización de prefijo "semantic." en filtros Chroma
# ============================================================================

def _normalize_view_name(view: str) -> List[str]:
    """Devuelve las variantes con y sin prefijo 'semantic.' de una vista."""
    clean = view.strip()
    if clean.startswith("semantic."):
        unprefixed = clean[len("semantic."):]
        return list({clean, unprefixed})
    return list({clean, f"semantic.{clean}"})


def _validate_chroma_filter(allowed_views: Optional[List[str]]) -> Dict[str, Any]:
    """Valida y construye el filtro para Chroma, tolerante al prefijo 'semantic.'."""
    if not allowed_views:
        return {}
    valid_views = set()
    for view in allowed_views:
        if isinstance(view, str) and view.strip():
            valid_views.update(_normalize_view_name(view))
    if not valid_views:
        return {}
    return {"view_name": {"$in": list(valid_views)}}


def _detect_temporal_context(query: str) -> Dict[str, Any]:
    """Detecta contexto temporal en la query para seleccionar vista adecuada."""
    context = {
        "is_historical": False,
        "is_future": False,
        "is_current": False,
        "needs_specific_date": False,
        "year_mentioned": None,
        "month_mentioned": False,
        "day_mentioned": False,
    }

    q = query.lower()
    current_year = datetime.now().year

    years = re.findall(r'\b(20\d{2})\b', query)
    if years:
        context["year_mentioned"] = years
        for y in years:
            yi = int(y)
            if yi >= current_year + 1:
                context["is_future"] = True
                context["needs_specific_date"] = True
            else:
                # FIX 5: Año explícito (incluido el en curso) ⇒ consulta de rango, no snapshot.
                # "Ventas del año 2026" pide agregación anual, no el estado del día de hoy.
                context["is_historical"] = True
                context["needs_specific_date"] = True

    month_names = [
        "enero", "febrero", "marzo", "abril", "mayo", "junio",
        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
    ]
    if any(m in q for m in month_names):
        context["month_mentioned"] = True
        context["needs_specific_date"] = True
        if not any(w in q for w in ["hoy", "ayer", "actual"]):
            context["is_historical"] = True

    historical_keywords = [
        "histórico", "historico", "históricas", "historicas", "pasado", "anterior",
        "histórica", "mes pasado", "año anterior", "tendencia", "evolución"
    ]
    if any(kw in q for kw in historical_keywords):
        context["is_historical"] = True

    date_patterns = [
        r'\b(20\d{2})[-/\s](0?[1-9]|1[0-2])[-/\s](0?[1-9]|[12]\d|3[01])\b',
        r'\b(0?[1-9]|[12]\d|3[01])[-/\s](0?[1-9]|1[0-2])[-/\s](20\d{2})\b',
    ]
    for pattern in date_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            context["needs_specific_date"] = True
            break

    current_keywords = ["hoy", "ayer", "actual", "today", "presente", "diario", "día", "dia"]
    if any(kw in q for kw in current_keywords):
        context["is_current"] = True
        context["day_mentioned"] = True

    logger.debug(f"Contexto temporal detectado para query '{query}': {context}")
    return context


def _detect_dimensions_in_query(query: str) -> List[str]:
    """Detecta dimensiones o columnas explícitamente solicitadas en la query."""
    if not query:
        return []

    q = query.lower().strip()
    q = q.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    q = re.sub(r"[¿?!¡.,;:\"'`()\[\]{}]", " ", q)

    dims = []

    if any(k in q for k in [
        "producto", "productos", "sku", "articulo", "artículo", "item",
        "mas vendido", "más vendido", "ranking", "top", "capuccino", "cappuccino"
    ]):
        dims.extend(["producto", "descripcion", "descripción", "nombre_producto"])

    if any(k in q for k in [
        "sede", "sucursal", "local", "tienda", "plaza", "ubicacion", "ubicación",
        "merced", "tajamar", "bolsillo", "lo contador", "san pablo"
    ]):
        dims.extend(["sucursal", "nombre_sede", "sede", "local", "tienda"])

    if any(k in q for k in ["categoria", "categoría", "categorias", "categorías"]):
        dims.extend(["categoria", "categoría", "categoria_nueva"])

    if any(k in q for k in [
        "fecha", "dia", "día", "mes", "año", "ano", "semana", "hoy", "ayer",
        "enero", "febrero", "marzo", "abril", "mayo", "junio",
        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
    ]):
        dims.extend(["fecha", "fecha_completa", "fecha_venta", "mes"])

    if any(k in q for k in ["hora", "horario", "franja", "pico", "demanda"]):
        dims.extend(["hora", "franja_horaria", "nombre_dia_semana"])

    return list(dict.fromkeys(dims))


def _extract_required_columns(column_hints: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Extrae dimensiones y métricas requeridas de los column_hints."""
    if not column_hints:
        return {"dimensions": [], "metrics": []}

    dims: List[str] = []
    metrics: List[str] = []

    if isinstance(column_hints.get("dimensions"), list):
        dims.extend(column_hints["dimensions"])
    if isinstance(column_hints.get("metrics"), list):
        metrics.extend(column_hints["metrics"])
    if column_hints.get("metric"):
        metrics.append(column_hints["metric"])

    flat_to_dim = [
        ("location", ["sede", "sucursal", "local", "tienda", "plaza", "ubicacion", "ubicación"]),
        ("product", ["producto", "sku", "articulo", "artículo", "item"]),
        ("category", ["categoria", "categoría"]),
    ]
    for key, _ in flat_to_dim:
        val = column_hints.get(key)
        if isinstance(val, str) and val.strip():
            dims.append(val)
        elif isinstance(val, list):
            dims.extend(v for v in val if isinstance(v, str) and v.strip())

    return {
        "dimensions": list(dict.fromkeys(dims)),
        "metrics": list(dict.fromkeys(metrics)),
    }


def _column_matches(candidate: str, requested: str) -> bool:
    """Comprueba si una columna candidata satisface una dimensión solicitada."""
    c = candidate.lower().strip().replace("_", " ")
    r = requested.lower().strip().replace("_", " ")
    return r in c or c in r or r.replace(" ", "") == c.replace(" ", "")


def _apply_implicit_temporal_rules(
    query_temporal: Dict[str, Any],
    view_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Ajusta la metadata de la vista considerando reglas implícitas de negocio."""
    adjusted = dict(view_metadata)
    temporal_type = adjusted.get("temporal_type", "general")
    time_scope = adjusted.get("time_scope", "unknown")
    view_name = adjusted.get("view_name", "")
    view_lower = view_name.lower()

    is_daily_snapshot = (
        temporal_type == "current"
        or time_scope == "daily"
        or "latest" in view_lower
        or "day" in view_lower
    )

    if query_temporal.get("is_current") and is_daily_snapshot:
        adjusted["supports_date_filter"] = True
        adjusted["implicit_date_scope"] = "current_day"

    if query_temporal.get("is_historical") or query_temporal.get("needs_specific_date"):
        if temporal_type == "current" and "historical" not in view_lower:
            adjusted["supports_date_filter"] = False
            adjusted["implicit_date_scope"] = "none"
        elif temporal_type == "historical":
            adjusted["supports_date_filter"] = True

    return adjusted


def _distance_to_similarity(distance: float) -> float:
    """
    FIX 3: Chroma devuelve DISTANCIA (menor = mejor). 
    Convertimos a similitud ∈ (0, 1].
    """
    try:
        d = max(float(distance), 0.0)
    except (TypeError, ValueError):
        return 0.0
    return 1.0 / (1.0 + d)


def _backfill_metadata_from_catalog(metadata: Dict[str, Any], biz_mem: Any) -> Dict[str, Any]:
    """FIX 6: Rellena campos faltantes del índice vectorial con el catálogo de BusinessMemory."""
    view = biz_mem.get_view(metadata.get("view_name", ""))
    if not view:
        return metadata
    m = dict(metadata)
    if not m.get("metrics"):
        m["metrics"] = list(view.metricas.keys())
    if not m.get("dimensions"):
        m["dimensions"] = list(view.columnas_fecha)
    if not m.get("temporal_type") or m.get("temporal_type") == "general":
        tipo = (view.tipo or "").lower()
        filtro = (view.filtro_fecha or "").lower()
        if "historical" in tipo or "rango" in filtro:
            m["temporal_type"] = "historical"
        elif "current" in tipo or "snapshot" in tipo or "no aplica" in filtro:
            m["temporal_type"] = "current"
    if "supports_location_filter" not in m:
        cols = [c.lower() for c in list(view.metricas.keys()) + list(view.columnas_fecha)]
        m["supports_location_filter"] = any(
            k in " ".join(cols) for k in ("sucursal", "sede", "local", "tienda", "plaza")
        )
    if not m.get("purpose"):
        m["purpose"] = view.descripcion
    return m


def _calculate_metadata_score(
    query_temporal: Dict[str, Any],
    view_metadata: Dict[str, Any],
    query: str = "",
    column_hints: Optional[Dict[str, Any]] = None,
) -> float:
    """Calcula score adicional basado en metadata, contexto temporal y dimensiones."""
    score = 0.0

    meta = _apply_implicit_temporal_rules(query_temporal, view_metadata)

    temporal_type = meta.get("temporal_type", "general")
    time_scope = meta.get("time_scope", "unknown")
    implicit_scope = meta.get("implicit_date_scope", "none")

    if query_temporal.get("is_historical") and temporal_type == "historical":
        score += 2.0
    elif query_temporal.get("is_current") and temporal_type == "current":
        score += 2.0
    elif query_temporal.get("is_current") and implicit_scope == "current_day":
        score += 1.8
    elif query_temporal.get("needs_specific_date") and temporal_type == "historical":
        score += 1.5

    if query_temporal.get("is_historical") and temporal_type == "current" and implicit_scope != "current_day":
        score -= 2.0
    elif query_temporal.get("is_current") and temporal_type == "historical":
        score -= 0.5

    if "daily" in time_scope and query_temporal.get("day_mentioned"):
        score += 0.5

    required = _extract_required_columns(column_hints)
    requested_dims = required["dimensions"] or _detect_dimensions_in_query(query)
    requested_metrics = required["metrics"]

    metrics = meta.get("metrics", []) or []
    dimensions = meta.get("dimensions", []) or []
    available_columns = [str(d) for d in dimensions] + [str(m) for m in metrics]

    for dim in requested_dims:
        has_dim = any(
            _column_matches(avail_col, dim)
            for avail_col in available_columns
        )
        if not has_dim:
            score -= 2.0
            logger.debug(
                f"Penalizando vista '{meta.get('view_name')}' por falta de dimensión '{dim}'"
            )
        else:
            score += 1.0

    for met in requested_metrics:
        has_metric = any(
            _column_matches(avail_col, met)
            for avail_col in available_columns
        )
        if not has_metric:
            score -= 1.5
            logger.debug(
                f"Penalizando vista '{meta.get('view_name')}' por falta de métrica '{met}'"
            )
        else:
            score += 0.8

    return score


def _calculate_compatibility_score(
    column_hints: Optional[Dict[str, Any]],
    view_metadata: Dict[str, Any]
) -> Dict[str, str]:
    """Calcula compatibilidad semántica entre hints del planner y metadata de la vista."""
    compatibility = {
        "metric": "ok",
        "date_range": "ok",
        "location": "ok",
        "product": "ok",
        "category": "ok",
    }

    safe_hints = column_hints or {}
    query_temporal = _detect_temporal_context(safe_hints.get("original_query", ""))
    adjusted_meta = _apply_implicit_temporal_rules(query_temporal, view_metadata)

    metrics = adjusted_meta.get("metrics", []) or []
    dimensions = adjusted_meta.get("dimensions", []) or []
    available_columns = [str(d).lower() for d in dimensions] + [str(m).lower() for m in metrics]

    required = _extract_required_columns(safe_hints)
    requested_dims = required["dimensions"] or _detect_dimensions_in_query(safe_hints.get("original_query", ""))
    requested_metrics = required["metrics"]

    metric_hint = requested_metrics[0] if requested_metrics else safe_hints.get("metric", "")
    if metric_hint:
        if isinstance(metrics, list) and metrics:
            metric_found = any(
                isinstance(view_metric, str)
                and (
                    metric_hint.lower() in view_metric.lower()
                    or view_metric.lower() in metric_hint.lower()
                )
                for view_metric in metrics
            )
            if not metric_found:
                compatibility["metric"] = "missing"
        else:
            compatibility["metric"] = "missing"

    location_hint = safe_hints.get("location", "")
    if location_hint:
        supports_location = adjusted_meta.get("supports_location_filter", False)
        if not supports_location:
            compatibility["location"] = "missing"
        elif isinstance(dimensions, list):
            location_keywords = ["sucursal", "nombre_sede", "sede", "local", "tienda", "plaza", "ubicacion", "ubicación"]
            has_location = any(
                any(keyword in str(dim).lower() for keyword in location_keywords)
                for dim in dimensions
            )
            if not has_location:
                compatibility["location"] = "missing"

    product_keywords = ["producto", "descripcion", "descripción", "nombre_producto", "sku", "articulo", "artículo", "item"]
    if any(d in requested_dims for d in ["producto", "descripcion", "descripción", "sku", "articulo", "artículo", "item"]):
        has_product = any(
            any(pk in col for pk in product_keywords)
            for col in available_columns
        )
        if not has_product:
            compatibility["product"] = "missing"

    if "categoria" in requested_dims or "categoría" in requested_dims:
        has_category = any(
            "categoria" in col or "categoría" in col
            for col in available_columns
        )
        if not has_category:
            compatibility["category"] = "missing"

    date_hint = safe_hints.get("date_range", "")
    if not date_hint and query_temporal.get("is_current"):
        date_hint = "current"
    elif not date_hint and query_temporal.get("needs_specific_date"):
        date_hint = "specific"

    if date_hint:
        supports_date = adjusted_meta.get("supports_date_filter", False)
        if not supports_date:
            compatibility["date_range"] = "missing"
        else:
            implicit_scope = adjusted_meta.get("implicit_date_scope", "none")
            if implicit_scope not in ("current_day", "historical_range"):
                if isinstance(dimensions, list):
                    time_keywords = ["fecha", "date", "time", "periodo", "mes", "año", "dia", "día", "hora"]
                    has_time = any(
                        any(keyword in str(dim).lower() for keyword in time_keywords)
                        for dim in dimensions
                    )
                    if not has_time:
                        compatibility["date_range"] = "missing"

    return compatibility


def _generate_compatibility_reason(compatibility: Dict[str, str], view_name: str) -> str:
    """Genera una razón explicativa para la compatibilidad."""
    missing = [k for k, v in compatibility.items() if v == "missing"]

    if not missing:
        return f"La vista '{view_name}' puede responder todos los filtros requeridos."

    reasons_map = {
        "metric": "no contiene la métrica solicitada",
        "location": "no soporta filtros por ubicación",
        "date_range": "no soporta filtros temporales",
        "product": "no tiene desglose por producto",
        "category": "no tiene desglose por categoría",
    }

    reasons = [reasons_map.get(k, f"falta {k}") for k in missing]
    return f"La vista '{view_name}' {' y '.join(reasons)}."


def obtener_candidatas_vistas(
    query: str,
    k: int = 5,
    allowed_views: Optional[List[str]] = None,
    min_score_threshold: float = 0.0,
    column_hints: Optional[Dict[str, Any]] = None,
    biz_mem: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Devuelve `k` vistas candidatas como diccionarios con orden y similitud corregidos."""
    search_kwargs = {"k": k}
    filter_dict = _validate_chroma_filter(allowed_views)
    if filter_dict:
        search_kwargs["filter"] = filter_dict

    try:
        docs_with_scores = _vector_store.similarity_search_with_score(query, **search_kwargs)
        logger.debug(f"Recuperados {len(docs_with_scores)} documentos con scores crudos")
    except Exception as e:
        logger.warning(f"similarity_search_with_score falló: {e}")
        logger.info("Intentando con as_retriever...")
        retriever = _vector_store.as_retriever(search_kwargs=search_kwargs)
        docs = retriever.invoke(query)
        docs_with_scores = [(doc, 0.0) for doc in docs]

    temporal_context = _detect_temporal_context(query)

    candidates = []
    for doc, score in docs_with_scores:
        # FIX 3: Conversión correcta de Distancia a Similitud ∈ (0, 1]
        similarity = _distance_to_similarity(score)
        
        # Backfill opcional de metadata vacía
        meta = doc.metadata
        if biz_mem is not None:
            meta = _backfill_metadata_from_catalog(meta, biz_mem)

        metadata_boost = _calculate_metadata_score(
            temporal_context, meta, query, column_hints
        )
        
        # Escalar el boost para que sea proporcional a la similitud (0 a 1)
        adjusted_score = similarity + (metadata_boost * 0.25)

        if adjusted_score < min_score_threshold:
            continue

        candidates.append({
            "view_name": meta.get("view_name", ""),
            "context": doc.page_content,
            "score": adjusted_score,
            "original_score": similarity,
            "raw_distance": float(score) if score is not None else None,
            "metadata_boost": metadata_boost,
            "temporal_context": temporal_context,
            "view_metadata": meta
        })

    # FIX 3: Orden descendente correcto (mayor similitud y boost primero)
    candidates.sort(key=lambda x: x["score"] or 0.0, reverse=True)
    logger.debug(f"Retornando {len(candidates)} candidatos después de ajuste de scores")
    return candidates


def obtener_candidatas_detalles(
    query: str,
    k: int = 5,
    allowed_views: Optional[List[str]] = None,
    min_score_threshold: float = 0.0,
    column_hints: Optional[Dict[str, Any]] = None,
    biz_mem: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Devuelve vistas candidatas con metadata detallada y compatibilidad semántica."""
    candidates = obtener_candidatas_vistas(
        query=query,
        k=k,
        allowed_views=allowed_views,
        min_score_threshold=min_score_threshold,
        column_hints=column_hints,
        biz_mem=biz_mem,
    )

    enriched_hints = dict(column_hints) if column_hints else {}
    enriched_hints["original_query"] = query

    detailed_candidates = []
    for candidate in candidates:
        metadata = candidate["view_metadata"]

        metrics = metadata.get("metrics", []) if isinstance(metadata.get("metrics"), list) else []
        dimensions = metadata.get("dimensions", []) if isinstance(metadata.get("dimensions"), list) else []
        usage_examples = metadata.get("usage_examples", []) if isinstance(metadata.get("usage_examples"), list) else []

        compatibility = _calculate_compatibility_score(enriched_hints, metadata)
        compatibility_reason = _generate_compatibility_reason(compatibility, candidate["view_name"])
        can_answer = all(v == "ok" for v in compatibility.values())

        detailed_candidates.append({
            "view_name": candidate["view_name"],
            "score": candidate["score"],
            "original_score": candidate["original_score"],
            "raw_distance": candidate.get("raw_distance"),
            "metadata_boost": candidate["metadata_boost"],
            "purpose": metadata.get("purpose", ""),
            "grain": metadata.get("grain", "desconocido"),
            "metrics": metrics,
            "dimensions": dimensions,
            "domain": metadata.get("domain", "general"),
            "notes": metadata.get("notes", ""),
            "keywords": metadata.get("keywords", ""),
            "context": candidate["context"],
            "temporal_type": metadata.get("temporal_type", "general"),
            "time_scope": metadata.get("time_scope", "unknown"),
            "temporal_context": candidate["temporal_context"],
            "usage_examples": usage_examples,
            "compatibility": compatibility,
            "compatibility_reason": compatibility_reason,
            "can_answer": can_answer
        })

    detailed_candidates.sort(key=lambda x: x["score"], reverse=True)

    # FIX 1: Log truthful — muestra conteo y diagnóstico si la colección está vacía
    raw_count = len(candidates)
    if not detailed_candidates:
        logger.warning(
            f"Retriever: 0 candidatas para '{query}' "
            f"(docs crudos de Chroma: {raw_count} — si es 0, la colección está vacía "
            f"o CHROMA_DIR no apunta al índice: {CHROMA_DIR})"
        )
    else:
        logger.info(f"Top 3 candidatos para '{query}':")
        for i, candidate in enumerate(detailed_candidates[:3]):
            status = "✅" if candidate["can_answer"] else "❌"
            logger.info(
                f"  {i+1}. {status} {candidate['view_name']} "
                f"(score: {candidate['score']:.3f}, sim: {candidate['original_score']:.3f}, boost: {candidate['metadata_boost']:+.2f}) "
                f"- {candidate['compatibility_reason']}"
            )

    return detailed_candidates


def payload_to_column_hints(payload: Any, original_query: str = "") -> Dict[str, Any]:
    """Convierte un SQLPayload-like en column_hints para el semantic_retriever."""
    metrics = getattr(payload, "metrics", None) or []
    dimensions = getattr(payload, "dimensions", None) or []

    hints: Dict[str, Any] = {
        "metrics": metrics,
        "dimensions": dimensions,
        "original_query": original_query
                         or getattr(payload, "question", "")
                         or getattr(payload, "query", ""),
        "filters_description": getattr(payload, "filters_description", ""),
    }

    if metrics:
        hints["metric"] = metrics[0]

    loc = next(
        (d for d in dimensions if any(k in d.lower() for k in ["sede", "sucursal", "local", "tienda", "plaza", "ubicacion", "ubicación"])),
        ""
    )
    prod = next(
        (d for d in dimensions if any(k in d.lower() for k in ["producto", "sku", "articulo", "artículo", "item"])),
        ""
    )
    cat = next(
        (d for d in dimensions if any(k in d.lower() for k in ["categoria", "categoría"])),
        ""
    )

    if loc:
        hints["location"] = loc
    if prod:
        hints["product"] = prod
    if cat:
        hints["category"] = cat

    time_window = getattr(payload, "time_window", None)
    if time_window:
        hints["date_range"] = time_window

    return hints


def seleccionar_vista_principal(
    query: str,
    column_hints: Optional[Dict[str, Any]] = None,
    allowed_views: Optional[List[str]] = None,
    min_score_threshold: float = 0.0,
    precomputed_candidates: Optional[List[Dict[str, Any]]] = None,
    biz_mem: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """
    Selecciona la vista principal más relevante para una consulta.
    Soporta `precomputed_candidates` para evitar doble cómputo de embeddings (FIX 4).
    """
    if precomputed_candidates is not None:
        candidates = precomputed_candidates
    else:
        candidates = obtener_candidatas_detalles(
            query=query,
            k=10,
            allowed_views=allowed_views,
            min_score_threshold=min_score_threshold,
            column_hints=column_hints,
            biz_mem=biz_mem,
        )

    if not candidates:
        logger.info(f"No se encontraron vistas candidatas para: '{query}'")
        return None

    compatible = [c for c in candidates if c.get("can_answer")]

    if compatible:
        vista_principal = compatible[0]
    else:
        vista_principal = candidates[0]
        logger.warning(
            f"Ninguna vista candidata es completamente compatible para '{query}'. "
            f"Usando la mejor disponible: {vista_principal['view_name']}"
        )

    status = "✅" if vista_principal["can_answer"] else "❌"
    logger.info(
        f"Vista principal seleccionada para '{query}': {status} {vista_principal['view_name']} "
        f"(score: {vista_principal['score']:.3f}) - {vista_principal['compatibility_reason']}"
    )

    # Fallbacks solo si la vista principal NO puede responder
    if not vista_principal.get("can_answer"):
        temporal_context = _detect_temporal_context(query)
        required = _extract_required_columns(column_hints)
        requested_dims = required["dimensions"] or _detect_dimensions_in_query(query)

        product_keywords = ["producto", "descripcion", "descripción", "sku", "articulo", "artículo", "item"]
        if any(any(pk in d.lower() for pk in product_keywords) for d in requested_dims):
            for candidate in candidates[1:]:
                if candidate.get("can_answer"):
                    logger.info(
                        f"Fallback por compatibilidad de producto: {candidate['view_name']} "
                        f"(score: {candidate['score']:.3f})"
                    )
                    return candidate

        if temporal_context.get("is_historical") or temporal_context.get("needs_specific_date"):
            if vista_principal.get("temporal_type") == "current":
                for candidate in candidates[1:]:
                    if candidate.get("temporal_type") in ("historical", "periodic") and candidate.get("can_answer"):
                        logger.info(
                            f"Reemplazando vista 'current' por histórica compatible: {candidate['view_name']} "
                            f"(score: {candidate['score']:.3f})"
                        )
                        return candidate

        if temporal_context.get("is_current"):
            if vista_principal.get("temporal_type") == "historical":
                for candidate in candidates[1:]:
                    if (candidate.get("temporal_type") == "current" or candidate.get("implicit_date_scope") == "current_day"):
                        if candidate.get("can_answer"):
                            logger.info(
                                f"Reemplazando vista histórica por snapshot actual compatible: {candidate['view_name']} "
                                f"(score: {candidate['score']:.3f})"
                            )
                            return candidate

    return vista_principal