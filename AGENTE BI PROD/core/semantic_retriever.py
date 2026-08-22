# core/semantic_retriever.py
# -------------------------------------------------
# Versión simplificada: El retriever SOLO hace retrieval vectorial.
# La inteligencia semántica vive en:
#   - BusinessMemory (catálogo = única fuente de verdad)
#   - Planner (tiene el catálogo en el prompt)
#   - SQL Agent (valida columnas exactas + retry)
# -------------------------------------------------
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import re
import os
import hashlib
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db")))
COLLECTION_NAME = "semantic_views"

_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Inicializar vector store con manejo de errores
try:
    _vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    logger.info(f"[Retriever] Chroma inicializado en {CHROMA_DIR}")
except Exception as e:
    logger.error(f"[Retriever] Error al inicializar Chroma: {e}")
    raise


# ============================================================================
# INICIALIZACIÓN PEREZOSA (lazy loading)
# ============================================================================

@lru_cache(maxsize=1)
def _get_doc_count() -> int:
    """Obtiene el conteo de documentos con caché."""
    try:
        return _vector_store._collection.count()
    except Exception:
        return -1


def _ensure_collection_initialized() -> bool:
    """Verifica e inicializa la colección si está vacía (solo una vez)."""
    count = _get_doc_count()
    if count > 0:
        logger.info(f"[Retriever] Colección lista: {count} documentos")
        return True
    
    logger.warning(f"[Retriever] Colección vacía en {CHROMA_DIR}. Iniciando indexación...")
    
    try:
        # Buscar AGENTS.md
        agents_path = Path(os.getenv("AGENTS_MD_PATH", str(BASE_DIR / "AGENTS.md")))
        
        if not agents_path.exists():
            alternative_paths = [
                BASE_DIR / "AGENTS.md",
                Path("/app/AGENTS.md"),
                Path("/app/AGENTE BI PROD/AGENTS.md"),
                Path("/data/AGENTS.md"),
                Path.cwd() / "AGENTS.md",
            ]
            for alt_path in alternative_paths:
                if alt_path.exists():
                    agents_path = alt_path
                    break
            else:
                logger.error("No se encontró AGENTS.md")
                return False
        
        # Indexar documentos (función simple de parsing)
        documents = _build_documents_from_agents_md(agents_path)
        if documents:
            _vector_store.add_documents(documents=documents, ids=[d.id for d in documents])
            logger.info(f"✅ Indexación completada: {len(documents)} documentos")
            return True
    except Exception as e:
        logger.error(f"Error en indexación: {e}", exc_info=True)
    
    return False


# ============================================================================
# PARSING SIMPLE DE AGENTS.md (sin heurísticas complejas)
# ============================================================================

def _build_documents_from_agents_md(agents_path: Path) -> List[Document]:
    """Extrae vistas semánticas de AGENTS.md de forma simple."""
    
    VALID_VIEWS = {
        'sales_review_day', 'sales_review_day_history', 'sales_review_locales_latest',
        'sales_review_locales', 'sales_week', 'sales_producto_daily',
        'mart_operacion_hora', 'dashboard_canjes_resumen', 'kpi_fidelizacion_detalle',
        'dashboard_cortesias_resumen', 'kpi_cortesia_detalle', 'kpi_categorias_diario',
        'dashboard_participacion_categorias', 'kpi_categorias_productos_sede'
    }
    
    if not agents_path.exists():
        return []
    
    raw = agents_path.read_text(encoding="utf-8-sig")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    
    # Extraer sección 3
    section_3_match = re.search(r'## 3\. Vistas Semánticas.*?(?=## 4\.|$)', raw, re.DOTALL)
    if not section_3_match:
        return []
    
    section_3 = section_3_match.group(0)
    parts = re.split(r'\n(?=###\s+)', section_3)
    
    documents = []
    seen = set()
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        first_line = part.splitlines()[0].strip()
        m = re.match(r'^###\s+([A-Za-z0-9_]+)', first_line)
        if not m:
            continue
        
        view_name = m.group(1).strip()
        if view_name not in VALID_VIEWS or view_name in seen:
            continue
        seen.add(view_name)
        
        description = "\n".join(part.splitlines()[1:]).strip()
        
        # Extraer métricas simples (formato: - `nombre`: descripción)
        metrics = re.findall(r'-\s*`([^`]+)`\s*:', description)
        
        # Filtrar palabras que no son métricas (lista simple)
        non_metrics = {'descripción', 'tipo', 'granularidad', 'filtro fecha', 'filas', 'nota'}
        metrics = [m for m in metrics if m.lower() not in non_metrics]
        
        if not metrics:
            metrics = [f"metrica_{view_name}"]
        
        # Crear documento con metadatos simples
        doc_id = hashlib.sha256(f"semantic.{view_name}".encode()).hexdigest()
        
        documents.append(Document(
            page_content=(
                f"VISTA SEMÁNTICA: semantic.{view_name}\n"
                f"DESCRIPCIÓN COMPLETA:\n{description}\n"
                f"MÉTRICAS: {', '.join(metrics[:15])}"
            ),
            metadata={
                "view_name": f"semantic.{view_name}",
                "metrics": ", ".join(metrics[:15]),
                "keywords": description[:500],
            },
            id=doc_id
        ))
    
    logger.info(f"Total de documentos creados: {len(documents)}")
    return documents


# Inicialización al importar (con lazy loading)
_initialized = _ensure_collection_initialized()


# ============================================================================
# FUNCIONES DE RETRIEVAL (simples, sin heurísticas)
# ============================================================================

def _normalize_view_name(view: str) -> List[str]:
    """Devuelve variantes con y sin prefijo 'semantic.'."""
    clean = view.strip()
    if clean.startswith("semantic."):
        return list({clean, clean[len("semantic."):]})
    return list({clean, f"semantic.{clean}"})


def _validate_chroma_filter(allowed_views: Optional[List[str]]) -> Dict[str, Any]:
    """Construye filtro para Chroma."""
    if not allowed_views:
        return {}
    valid_views = set()
    for view in allowed_views:
        if isinstance(view, str) and view.strip():
            valid_views.update(_normalize_view_name(view))
    if not valid_views:
        return {}
    return {"view_name": {"$in": list(valid_views)}}


def _distance_to_similarity(distance: float) -> float:
    """Convierte distancia de Chroma a similitud ∈ (0, 1]."""
    try:
        d = max(float(distance), 0.0)
    except (TypeError, ValueError):
        return 0.0
    return 1.0 / (1.0 + d)


def obtener_candidatas_vistas(
    query: str,
    k: int = 5,
    allowed_views: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Devuelve vistas candidatas basadas SOLO en similitud vectorial.
    La inteligencia semántica la hace el Planner con el catálogo.
    """
    search_kwargs = {"k": k}
    filter_dict = _validate_chroma_filter(allowed_views)
    if filter_dict:
        search_kwargs["filter"] = filter_dict

    try:
        docs_with_scores = _vector_store.similarity_search_with_score(query, **search_kwargs)
    except Exception as e:
        logger.warning(f"Búsqueda vectorial falló: {e}")
        return []

    candidates = []
    for doc, score in docs_with_scores:
        similarity = _distance_to_similarity(score)
        
        candidates.append({
            "view_name": doc.metadata.get("view_name", ""),
            "context": doc.page_content,
            "score": similarity,
            "view_metadata": doc.metadata,
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    if candidates:
        logger.info(f"Top {min(3, len(candidates))} candidatos para '{query[:50]}...':")
        for i, c in enumerate(candidates[:3]):
            logger.info(f"  {i+1}. {c['view_name']} (score: {c['score']:.3f})")
    else:
        logger.warning(f"0 candidatas para '{query[:50]}...' (colección vacía o sin matches)")
    
    return candidates


def payload_to_column_hints(payload: Any, original_query: str = "") -> Dict[str, Any]:
    """Convierte payload a hints simples para el retriever."""
    return {
        "metrics": getattr(payload, "metrics", None) or [],
        "dimensions": getattr(payload, "dimensions", None) or [],
        "original_query": original_query or getattr(payload, "task", ""),
    }


def seleccionar_vista_principal(
    query: str,
    allowed_views: Optional[List[str]] = None,
    precomputed_candidates: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Selecciona la vista con mayor similitud vectorial.
    El Planner decide si es realmente compatible usando el catálogo.
    """
    candidates = precomputed_candidates or obtener_candidatas_vistas(
        query=query,
        k=5,
        allowed_views=allowed_views,
    )
    
    if not candidates:
        return None
    
    return candidates[0]