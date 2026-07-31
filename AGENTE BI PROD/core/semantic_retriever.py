# core/semantic_retriever.py
# -------------------------------------------------
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
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


def _validate_chroma_filter(allowed_views: Optional[List[str]]) -> Dict[str, Any]:
    """Valida y construye el filtro para Chroma."""
    if not allowed_views:
        return {}
    valid_views = [view for view in allowed_views if isinstance(view, str) and view.strip()]
    if not valid_views:
        return {}
    return {"view_name": {"$in": valid_views}}


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
    
    # Detectar años específicos
    years = re.findall(r'\b(20\d{2})\b', query)
    if years:
        context["year_mentioned"] = years
        future_years = [int(y) for y in years if int(y) >= current_year + 1]
        historical_years = [int(y) for y in years if int(y) < current_year]
        if future_years:
            context["is_future"] = True
            context["needs_specific_date"] = True
        if historical_years:
            context["is_historical"] = True
            context["needs_specific_date"] = True
    
    # Detectar meses como palabra (con o sin año) → implícitamente histórico o al menos requiere fecha
    month_names = [
        "enero", "febrero", "marzo", "abril", "mayo", "junio",
        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
    ]
    if any(m in q for m in month_names):
        context["month_mentioned"] = True
        context["needs_specific_date"] = True
        if not any(w in q for w in ["hoy", "ayer", "actual"]):
            context["is_historical"] = True
    
    # Detectar palabras clave históricas
    historical_keywords = [
        "histórico", "historico", "históricas", "historicas", "pasado", "anterior",
        "histórica", "mes pasado", "año anterior", "tendencia", "evolución"
    ]
    if any(kw in q for kw in historical_keywords):
        context["is_historical"] = True
    
    # Detectar fechas específicas (formatos numéricos)
    date_patterns = [
        r'\b(20\d{2})[-/\s](0?[1-9]|1[0-2])[-/\s](0?[1-9]|[12]\d|3[01])\b',
        r'\b(0?[1-9]|[12]\d|3[01])[-/\s](0?[1-9]|1[0-2])[-/\s](20\d{2})\b',
    ]
    for pattern in date_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            context["needs_specific_date"] = True
            break
    
    # Detectar "hoy", "ayer", "diario", "actual" → snapshot actual
    current_keywords = ["hoy", "ayer", "actual", "today", "presente", "diario", "día", "dia"]
    if any(kw in q for kw in current_keywords):
        context["is_current"] = True
        context["day_mentioned"] = True
    
    logger.debug(f"Contexto temporal detectado para query '{query}': {context}")
    return context


# ============================================================================
# NUEVO: Detección de dimensiones solicitadas en la query
# ============================================================================

def _detect_dimensions_in_query(query: str) -> List[str]:
    """
    Detecta dimensiones o columnas explícitamente solicitadas en la query.
    Devuelve lista de nombres de columnas candidatas a buscar en la vista.
    """
    if not query:
        return []
    
    q = query.lower().strip()
    q = q.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    q = re.sub(r"[¿?!¡.,;:\"'`()\[\]{}]", " ", q)
    
    dims = []
    
    # Producto / SKU / artículo
    if any(k in q for k in [
        "producto", "productos", "sku", "articulo", "artículo", "item",
        "mas vendido", "más vendido", "ranking", "top", "capuccino", "cappuccino"
    ]):
        dims.extend(["producto", "descripcion", "descripción", "nombre_producto"])
    
    # Sede / local / sucursal / plaza
    if any(k in q for k in [
        "sede", "sucursal", "local", "tienda", "plaza", "ubicacion", "ubicación",
        "merced", "tajamar", "bolsillo", "lo contador", "san pablo"
    ]):
        dims.extend(["sucursal", "nombre_sede", "sede", "local", "tienda"])
    
    # Categoría
    if any(k in q for k in ["categoria", "categoría", "categorias", "categorías"]):
        dims.extend(["categoria", "categoría", "categoria_nueva"])
    
    # Fecha
    if any(k in q for k in [
        "fecha", "dia", "día", "mes", "año", "ano", "semana", "hoy", "ayer",
        "enero", "febrero", "marzo", "abril", "mayo", "junio",
        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
    ]):
        dims.extend(["fecha", "fecha_completa", "fecha_venta", "mes"])
    
    # Franja / hora
    if any(k in q for k in ["hora", "horario", "franja", "pico", "demanda"]):
        dims.extend(["hora", "franja_horaria", "nombre_dia_semana"])
    
    return list(dict.fromkeys(dims))  # preservar orden, eliminar duplicados


def _column_matches(candidate: str, requested: str) -> bool:
    """
    Comprueba si una columna candidata satisface una dimensión solicitada.
    Soporta coincidencia exacta, subcadena y variantes.
    """
    c = candidate.lower().strip().replace("_", " ")
    r = requested.lower().strip().replace("_", " ")
    return r in c or c in r or r.replace(" ", "") == c.replace(" ", "")


def _apply_implicit_temporal_rules(
    query_temporal: Dict[str, Any],
    view_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Ajusta la metadata de la vista considerando reglas implícitas de negocio.
    """
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


def _calculate_metadata_score(
    query_temporal: Dict[str, Any],
    view_metadata: Dict[str, Any],
    query: str = ""
) -> float:
    """
    Calcula score adicional basado en metadata, contexto temporal y dimensiones requeridas.
    AHORA con penalización fuerte por dimensiones faltantes.
    """
    score = 0.0
    
    meta = _apply_implicit_temporal_rules(query_temporal, view_metadata)
    
    temporal_type = meta.get("temporal_type", "general")
    time_scope = meta.get("time_scope", "unknown")
    implicit_scope = meta.get("implicit_date_scope", "none")
    
    # Boost para alineación temporal
    if query_temporal.get("is_historical") and temporal_type == "historical":
        score += 2.0
    elif query_temporal.get("is_current") and temporal_type == "current":
        score += 2.0
    elif query_temporal.get("is_current") and implicit_scope == "current_day":
        score += 1.8
    elif query_temporal.get("needs_specific_date") and temporal_type == "historical":
        score += 1.5
    
    # Penalizaciones temporales
    if query_temporal.get("is_historical") and temporal_type == "current" and implicit_scope != "current_day":
        score -= 2.0
    elif query_temporal.get("is_current") and temporal_type == "historical":
        score -= 0.5
    
    # Boost por granularidad temporal
    if "daily" in time_scope and query_temporal.get("day_mentioned"):
        score += 0.5
    
    # =========================================================================
    # NUEVO: Penalización drástica por dimensiones faltantes
    # =========================================================================
    requested_dims = _detect_dimensions_in_query(query)
    if requested_dims:
        metrics = meta.get("metrics", []) or []
        dimensions = meta.get("dimensions", []) or []
        available_columns = [str(d) for d in dimensions] + [str(m) for m in metrics]
        
        for dim in requested_dims:
            has_dim = any(
                _column_matches(avail_col, dim)
                for avail_col in available_columns
            )
            if not has_dim:
                score -= 5.0  # penalización alta: mejor no elegir esta vista
                logger.debug(
                    f"Penalizando vista '{meta.get('view_name')}' por falta de dimensión '{dim}'"
                )
            else:
                # Bonus leve por tener la dimensión exacta
                score += 0.5
    
    return score


def _calculate_compatibility_score(
    column_hints: Dict[str, Any],
    view_metadata: Dict[str, Any]
) -> Dict[str, str]:
    """
    Calcula compatibilidad semántica entre hints del planner y metadata de la vista.
    AHORA valida dimensiones reales de la query.
    """
    compatibility = {
        "metric": "ok",
        "date_range": "ok",
        "location": "ok",
        "product": "ok",
        "category": "ok",
    }
    
    query_temporal = _detect_temporal_context(column_hints.get("original_query", ""))
    adjusted_meta = _apply_implicit_temporal_rules(query_temporal, view_metadata)
    
    # Recopilar columnas disponibles
    metrics = adjusted_meta.get("metrics", []) or []
    dimensions = adjusted_meta.get("dimensions", []) or []
    available_columns = [str(d).lower() for d in dimensions] + [str(m).lower() for m in metrics]
    view_name = adjusted_meta.get("view_name", "")
    
    # Validar métrica
    metric_hint = (column_hints or {}).get("metric", "")
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
    
    # Validar ubicación
    location_hint = (column_hints or {}).get("location", "")
    if location_hint:
        supports_location = adjusted_meta.get("supports_location_filter", False)
        if not supports_location:
            compatibility["location"] = "missing"
        elif isinstance(dimensions, list):
            location_keywords = ["sucursal", "nombre_sede", "sede", "local", "tienda", "plaza", "ubicacion"]
            has_location = any(
                any(keyword in str(dim).lower() for keyword in location_keywords)
                for dim in dimensions
            )
            if not has_location:
                compatibility["location"] = "missing"
    
    # Validar producto
    requested_dims = _detect_dimensions_in_query(column_hints.get("original_query", ""))
    product_keywords = ["producto", "descripcion", "descripción", "nombre_producto", "sku", "articulo"]
    if any(d in requested_dims for d in ["producto", "descripcion", "descripción"]):
        has_product = any(
            any(pk in col for pk in product_keywords)
            for col in available_columns
        )
        if not has_product:
            compatibility["product"] = "missing"
    
    # Validar categoría
    if "categoria" in requested_dims or "categoría" in requested_dims:
        has_category = any(
            "categoria" in col or "categoría" in col
            for col in available_columns
        )
        if not has_category:
            compatibility["category"] = "missing"
    
    # Validar fecha con reglas implícitas
    date_hint = (column_hints or {}).get("date_range", "")
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
) -> List[Dict[str, Any]]:
    """Devuelve `k` vistas candidatas como diccionarios."""
    search_kwargs = {"k": k}
    filter_dict = _validate_chroma_filter(allowed_views)
    if filter_dict:
        search_kwargs["filter"] = filter_dict

    try:
        docs_with_scores = _vector_store.similarity_search_with_score(query, **search_kwargs)
        logger.debug(f"Recuperados {len(docs_with_scores)} documentos con scores")
    except Exception as e:
        logger.warning(f"similarity_search_with_score falló: {e}")
        logger.info("Intentando con as_retriever...")
        retriever = _vector_store.as_retriever(search_kwargs=search_kwargs)
        docs = retriever.invoke(query)
        docs_with_scores = [(doc, 0.0) for doc in docs]

    temporal_context = _detect_temporal_context(query)
    
    candidates = []
    for doc, score in docs_with_scores:
        score_value = float(score) if score is not None else 0.0
        # NUEVO: pasar query para penalización de dimensiones
        metadata_boost = _calculate_metadata_score(temporal_context, doc.metadata, query)
        adjusted_score = score_value + metadata_boost
        
        if adjusted_score < min_score_threshold:
            continue

        candidates.append({
            "view_name": doc.metadata.get("view_name", ""),
            "context": doc.page_content,
            "score": adjusted_score,
            "original_score": score_value,
            "metadata_boost": metadata_boost,
            "temporal_context": temporal_context,
            "view_metadata": doc.metadata
        })

    candidates.sort(key=lambda x: x["score"] or 0.0, reverse=True)
    logger.debug(f"Retornando {len(candidates)} candidatos después de ajuste de scores")
    return candidates


def obtener_candidatas_detalles(
    query: str,
    k: int = 5,
    allowed_views: Optional[List[str]] = None,
    min_score_threshold: float = 0.0,
    column_hints: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Devuelve vistas candidatas con información detallada de metadata Y compatibilidad semántica.
    """
    candidates = obtener_candidatas_vistas(query, k, allowed_views, min_score_threshold)
    
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
    
    logger.info(f"Top 3 candidatos para '{query}':")
    for i, candidate in enumerate(detailed_candidates[:3]):
        status = "✅" if candidate["can_answer"] else "❌"
        logger.info(f"  {i+1}. {status} {candidate['view_name']} (score: {candidate['score']:.3f}) - {candidate['compatibility_reason']}")
    
    return detailed_candidates


def seleccionar_vista_principal(
    query: str,
    column_hints: Optional[Dict[str, Any]] = None,
    allowed_views: Optional[List[str]] = None,
    min_score_threshold: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """
    Selecciona la vista principal más relevante para una consulta.
    CON VALIDACIÓN EXPLÍCITA DE COMPATIBILIDAD Y FALLBACKS.
    """
    candidates = obtener_candidatas_detalles(
        query=query,
        k=10,  # Ampliar para encontrar mejores alternativas
        allowed_views=allowed_views,
        min_score_threshold=min_score_threshold,
        column_hints=column_hints
    )
    
    if not candidates:
        logger.info(f"No se encontraron vistas candidatas para: '{query}'")
        return None
    
    # Separar compatibles e incompatibles
    compatible = [c for c in candidates if c.get("can_answer")]
    incompatible = [c for c in candidates if not c.get("can_answer")]
    
    # Preferir la primera compatible
    if compatible:
        vista_principal = compatible[0]
    else:
        # Si ninguna es compatible, tomar la de mayor score de todas
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
    
    # Fallbacks temporales adicionales
    temporal_context = _detect_temporal_context(query)
    requested_dims = _detect_dimensions_in_query(query)
    
    # Si la mejor vista no puede responder por producto, buscar alternativa
    if not vista_principal.get("can_answer") and any(d in requested_dims for d in ["producto", "descripcion", "descripción"]):
        for candidate in candidates[1:]:
            if candidate.get("can_answer"):
                logger.info(
                    f"Fallback por compatibilidad de producto: {candidate['view_name']} "
                    f"(score: {candidate['score']:.3f})"
                )
                return candidate
    
    # Fallback histórico vs current
    if temporal_context.get("is_historical") or temporal_context.get("needs_specific_date"):
        if vista_principal.get("temporal_type") == "current":
            for candidate in candidates[1:]:
                if candidate.get("temporal_type") in ("historical", "periodic"):
                    if candidate.get("can_answer"):
                        logger.info(
                            f"Reemplazando vista 'current' por histórica compatible: {candidate['view_name']} "
                            f"(score: {candidate['score']:.3f})"
                        )
                        return candidate
    
    # Fallback current vs histórico
    if temporal_context.get("is_current"):
        if vista_principal.get("temporal_type") == "historical":
            for candidate in candidates[1:]:
                if candidate.get("temporal_type") == "current" or candidate.get("implicit_date_scope") == "current_day":
                    if candidate.get("can_answer"):
                        logger.info(
                            f"Reemplazando vista histórica por snapshot actual compatible: {candidate['view_name']} "
                            f"(score: {candidate['score']:.3f})"
                        )
                        return candidate
    
    return vista_principal


# ------------------------------------------------------------------
# NUEVO: RESOLUCIÓN SEMÁNTICA SOBRE CATÁLOGO DOCUMENTADO (BusinessMemory)
# ------------------------------------------------------------------

# NOTA: Lazy singleton para evitar import circular con core.harness.
# core.harness importa funciones de este módulo durante su carga,
# por lo que no podemos importar BusinessMemory a nivel de módulo aquí.
_biz_mem_catalog = None


def _get_biz_mem_catalog():
    """Devuelve instancia lazy/singleton de BusinessMemory."""
    global _biz_mem_catalog
    if _biz_mem_catalog is None:
        from core.harness import BusinessMemory
        _biz_mem_catalog = BusinessMemory.from_file()
    return _biz_mem_catalog


def _normalize_for_catalog(text: str) -> str:
    if not text:
        return ""
    return (
        text.lower()
        .strip()
        .replace("_", " ")
        .replace("á", "a").replace("é", "e").replace("í", "i")
        .replace("ó", "o").replace("ú", "u")
    )


_SEMANTIC_COLUMN_MAP = {
    "producto": ["producto", "descripcion", "descripción", "nombre_producto", "articulo", "artículo", "sku", "productos"],
    "sucursal": ["sucursal", "nombre_sede", "sede", "local", "tienda", "plaza", "ubicacion", "ubicación", "sucursales"],
    "categoria": ["categoria", "categoría", "categoria_nueva", "categorias", "categorías"],
    "subcategoria": ["subcategoria", "subcategoría"],
    "fecha": ["fecha", "fecha_completa", "fecha_venta", "mes", "anio", "año"],
    "venta_total": ["venta_total", "ventas", "ventas_totales", "ventas_total", "subtotal_diario", "ingreso", "total_ventas"],
    "unidades": ["unidades", "cantidad", "unidades_totales", "unidades_vendidas", "unidades_total"],
    "transacciones": ["transacciones", "total_transacciones", "numero_transacciones", "transacciones_totales"],
    "ticket_promedio": ["ticket_promedio", "ticket_promedio_sede"],
}



def get_view_columns(view_name: str) -> List[str]:
    """
    Devuelve las columnas reales (métricas + fechas) documentadas para una vista.
    """
    clean = view_name.replace("semantic.", "").strip()
    info = _get_biz_mem_catalog().get_view(clean)
    if not info:
        return []
    return list(info.metricas.keys()) + info.columnas_fecha


def column_exists_in_view(view_name: str, column_name: str) -> bool:
    """
    Verifica si una columna semántica existe en la vista, con mapeo flexible.
    """
    requested = _normalize_for_catalog(column_name)
    available = [_normalize_for_catalog(c) for c in get_view_columns(view_name)]

    for avail in available:
        if requested == avail or requested in avail or avail in requested:
            return True

    for variant in _SEMANTIC_COLUMN_MAP.get(requested, [requested]):
        v = _normalize_for_catalog(variant)
        for avail in available:
            if v == avail or v in avail or avail in v:
                return True
    return False


def resolve_column(view_name: str, semantic_name: str) -> Optional[str]:
    """
    Devuelve el nombre REAL de la columna en la vista que mejor coincida.
    """
    requested = _normalize_for_catalog(semantic_name)
    available = get_view_columns(view_name)
    available_norm = [_normalize_for_catalog(c) for c in available]

    # Exacto
    for i, avail in enumerate(available_norm):
        if requested == avail:
            return available[i]

    # Subcadena
    for i, avail in enumerate(available_norm):
        if requested in avail or avail in requested:
            return available[i]

    # Mapeo semántico
    for variant in _SEMANTIC_COLUMN_MAP.get(requested, [requested]):
        v = _normalize_for_catalog(variant)
        for i, avail in enumerate(available_norm):
            if v == avail or v in avail or avail in v:
                return available[i]
    return None


def find_compatible_view(task: Any, allowed_views: List[str]) -> Optional[str]:
    """
    Encuentra la primera vista que contenga todas las métricas,
    dimensiones y columnas de filtro.
    """
    required: Set[str] = set()
    for m in (task.metrics or []):
        required.add(_normalize_for_catalog(m))
    for d in (task.dimensions or []):
        required.add(_normalize_for_catalog(d))
    for f in (task.filters or []):
        required.add(_normalize_for_catalog(f.column))

    if not required:
        return None

    candidates = task.candidate_views or allowed_views
    for view_full in candidates:
        view_name = view_full.replace("semantic.", "").strip()
        available = {_normalize_for_catalog(c) for c in get_view_columns(view_name)}
        missing = [
            col for col in required
            if not any(col == avail or col in avail or avail in col for avail in available)
        ]
        if not missing:
            return view_full
    return None
