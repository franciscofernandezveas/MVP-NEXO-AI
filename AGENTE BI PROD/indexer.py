# indexer.py
# Ubicación: AGENTE BI PROD\indexer.py

import hashlib
import os
import re
from pathlib import Path

from dotenv import load_dotenv

# CARGA EXPLÍCITA DEL .ENV (override=True fuerza recarga)
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# === CONFIGURACIÓN DE RUTAS ===
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "AGENTS.md"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "semantic_views"

# Lista de vistas válidas según la taxonomía del AGENTS.md
VALID_VIEWS = {
    'sales_review_day',
    'sales_review_day_history',
    'sales_review_locales_latest',
    'sales_review_locales',
    'sales_week',
    'sales_producto_daily',
    'mart_operacion_hora',
    'dashboard_canjes_resumen',
    'kpi_fidelizacion_detalle',
    'dashboard_cortesias_resumen',
    'kpi_cortesia_detalle',
    'kpi_categorias_diario',
    'dashboard_participacion_categorias',
    'kpi_categorias_productos_sede'
}


def _make_id(view_name: str) -> str:
    # Usar el hash completo para evitar colisiones
    return hashlib.sha256(view_name.encode()).hexdigest()


def _extract_field(text: str, field_name: str) -> str:
    """Extrae un campo del markdown tipo **Campo**: valor o - **Campo**: valor."""
    pattern = rf'(?:-\s*)?\*\*{re.escape(field_name)}\*\*[:：]\s*(.*?)(?=\n\s*(?:-\s*)?\*\*|\Z)'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _extract_list(text: str, field_name: str) -> list[str]:
    """Extrae una lista separada por comas de un campo markdown."""
    raw = _extract_field(text, field_name)
    if not raw:
        return []
    items = [item.strip() for item in raw.replace("\n", " ").split(",") if item.strip()]
    return items


def _extract_metrics_from_bullets(description: str) -> list[str]:
    """Extrae métricas de las viñetas que tienen formato - `nombre_metrica`: descripción"""
    metrics = []
    # Buscar patrones como: - `ventas_hoy`: descripción
    pattern = r'-\s*`([^`]+)`\s*:'
    matches = re.findall(pattern, description)
    
    # Filtrar palabras que no son métricas
    non_metric_words = {
        'descripción', 'tipo', 'granularidad', 'filtro fecha', 'grano',
        'filas', 'nota', 'fecha', 'fecha_key'
    }
    
    for match in matches:
        if match.lower() not in non_metric_words:
            metrics.append(match)
    
    return metrics


def _parse_agents_md(file_path: Path) -> list[Document]:
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de catálogo: {file_path}")

    raw = file_path.read_text(encoding="utf-8-sig")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    
    # Extraer solo la sección 3 (Vistas Semánticas)
    section_3_match = re.search(r'## 3\. Vistas Semánticas.*?(?=## 4\.|$)', raw, re.DOTALL)
    if not section_3_match:
        raise ValueError("No se encontró la sección 3 'Vistas Semánticas' en AGENTS.md")
    
    section_3 = section_3_match.group(0)
    
    # Dividir por bloques de encabezados de nivel 3 (###)
    parts = re.split(r'\n(?=###\s+)', section_3)
    documents = []
    seen_views = set()

    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        first_line = part.splitlines()[0].strip()
        m_view = re.match(r'^###\s+([A-Za-z0-9_]+)', first_line)
        if not m_view:
            continue
            
        view_name = m_view.group(1).strip()
        
        # Verificar si es una vista válida
        if view_name not in VALID_VIEWS:
            print(f"[SKIP] No es una vista válida: {view_name}")
            continue
        
        # Verificar si ya vimos esta vista
        if view_name in seen_views:
            print(f"[WARN] Vista duplicada encontrada: {view_name}. Omitiendo...")
            continue
        seen_views.add(view_name)
        
        description = "\n".join(part.splitlines()[1:]).strip()
        
        # Extraer campos
        purpose = _extract_field(description, "Descripción") or _extract_field(description, "Propósito")
        grain = _extract_field(description, "Granularidad") or _extract_field(description, "Grain")
        
        # Extraer métricas de las viñetas
        metrics = _extract_metrics_from_bullets(description)
        
        # Si no hay métricas de viñetas, intentar con listas
        if not metrics:
            metrics = _extract_list(description, "Métricas") or _extract_list(description, "Metricas")
        
        # Garantizar que metrics nunca esté vacío
        if not metrics:
            metrics = [f"metrica_{view_name}"]
        
        # Extraer dimensiones (para vistas que las tienen)
        dimensions = _extract_list(description, "Dimensiones") or _extract_list(description, "Cols")
        
        # Si no hay dimensiones explícitas, inferir de las métricas
        if not dimensions:
            # Buscar dimensiones en las métricas
            dimension_candidates = []
            for metric in metrics:
                if any(dim_word in metric.lower() for dim_word in ['sede', 'sucursal', 'local', 'producto', 'categoria', 'fecha']):
                    if 'sede' in metric.lower() or 'sucursal' in metric.lower():
                        dimension_candidates.append('sede')
                    if 'producto' in metric.lower():
                        dimension_candidates.append('producto')
                    if 'categoria' in metric.lower():
                        dimension_candidates.append('categoria')
                    if 'fecha' in metric.lower():
                        dimension_candidates.append('fecha')
            
            if not dimension_candidates:
                dimensions = ['general']
            else:
                dimensions = list(set(dimension_candidates))
        
        # Valores por defecto para campos vacíos
        if not purpose:
            purpose = f"Vista {view_name}"
        if not grain:
            grain = "general"
        
        view_lower = view_name.lower()
        purpose_lower = purpose.lower()
        grain_lower = grain.lower()
        dims_lower = [d.lower() for d in dimensions]
        
        # Clasificación temporal
        temporal_type = "general"
        if any(w in view_lower for w in ["latest", "hoy", "actual", "today", "_day"]):
            temporal_type = "current"
        elif any(w in view_lower for w in ["history", "historico", "histórico", "tendencia"]):
            temporal_type = "historical"
        elif any(w in view_lower for w in ["week", "semana", "comparativa"]):
            temporal_type = "periodic"
        elif any(w in view_lower for w in ["dashboard"]):
            temporal_type = "dashboard"
        
        # Clasificación de granularidad temporal
        time_scope = "unknown"
        if any(w in grain_lower for w in ["día", "diario", "fecha", "daily"]):
            time_scope = "daily"
        elif any(w in grain_lower for w in ["semana", "semanal", "weekly"]):
            time_scope = "weekly"
        elif any(w in grain_lower for w in ["mes", "mensual", "monthly"]):
            time_scope = "monthly"
        elif any(w in grain_lower for w in ["hora", "horaria", "hourly"]):
            time_scope = "hourly"
        
        # Detección de filtros
        supports_date_filter = any(d in dims_lower for d in ["fecha", "date", "periodo", "mes", "año", "dia", "día"])
        supports_location_filter = any(d in dims_lower for d in ["sede", "local", "sucursal", "ubicacion", "ubicación", "store", "branch"])
        
        keywords_str = f"{view_name} {purpose} " + " ".join(metrics)
        
        page_content = (
            f"VISTA SEMÁNTICA: semantic.{view_name}\n"
            f"NOMBRE TÉCNICO: {view_name}\n"
            f"CONTEXTO DE NEGOCIO Y REGLAS:\n{description}\n"
            f"KEYWORDS PARA BÚSQUEDA: {keywords_str}"
        )
        
        doc_id = _make_id(f"semantic.{view_name}")
        documents.append(Document(
            page_content=page_content,
            metadata={
                "view_name": f"semantic.{view_name}",
                "schema": "semantic",
                "keywords": keywords_str,
                "purpose": purpose,
                "grain": grain,
                "metrics": metrics[:10],  # Limitar a 10 métricas para evitar sobrecarga
                "dimensions": dimensions[:10],  # Limitar a 10 dimensiones
                "notes": "",
                "temporal_type": temporal_type,
                "time_scope": time_scope,
                "supports_date_filter": supports_date_filter,
                "supports_location_filter": supports_location_filter,
            },
            id=doc_id
        ))
        print(f"[DEBUG] Indexada: semantic.{view_name} | temporal: {temporal_type} | scope: {time_scope} | metrics: {len(metrics)} | dims: {len(dimensions)}")
    
    if not documents:
        raise ValueError(
            f"No se parseó ninguna vista desde {file_path}. Revisa la estructura del archivo AGENTS.md."
        )
    
    print(f"\n✅ Total de vistas parseadas: {len(documents)}")
    return documents


def indexar():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY no encontrada. Revisa que tu archivo .env tenga:\n"
            'OPENAI_API_KEY=sk-...\n'
            f"O defínela como variable de entorno antes de ejecutar. (Buscando en: {env_path})"
        )

    print(f"📖 Leyendo catálogo desde: {DATA_PATH}")
    docs = _parse_agents_md(DATA_PATH)
    print(f"🧩 Vistas parseadas: {len(docs)}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    
    vector_store.add_documents(documents=docs, ids=[d.id for d in docs])
    
    # Limpieza de obsoletos
    current_ids = {d.id for d in docs}
    try:
        all_ids = set(vector_store._collection.get()["ids"])
        to_delete = list(all_ids - current_ids)
        if to_delete:
            vector_store._collection.delete(ids=to_delete)
            print(f"🗑️ Vistas obsoletas eliminadas: {len(to_delete)}")
    except Exception as e:
        print(f"[WARN] No se pudo limpiar obsoletos: {e}")
    
    count = vector_store._collection.count()
    print(f"✅ Indexación completada. Total documentos en Chroma: {count}")


if __name__ == "__main__":
    indexar()