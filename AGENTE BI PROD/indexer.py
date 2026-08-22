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
DATA_PATH = Path(os.getenv("AGENTS_MD_PATH", str(BASE_DIR / "AGENTS.md")))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db")))
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
    """Genera un ID único usando SHA-256 completo."""
    return hashlib.sha256(view_name.encode()).hexdigest()


def _extract_field(text: str, field_name: str) -> str:
    """Extrae un campo del markdown tipo **Campo**: valor."""
    pattern = rf'(?:-\s*)?\*\*{re.escape(field_name)}\*\*[:：]\s*(.*?)(?=\n\s*(?:-\s*)?\*\*|\Z)'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _extract_metrics_from_bullets(description: str) -> list[str]:
    """Extrae métricas de las viñetas que tienen formato - `nombre_metrica`: descripción"""
    metrics = []
    # Buscar patrones como: - `ventas_hoy`: descripción
    pattern = r'-\s*`([^`]+)`\s*:'
    matches = re.findall(pattern, description)
    
    # Filtrar palabras que no son métricas
    non_metric_words = {
        'descripción', 'descripcion', 'tipo', 'granularidad', 'filtro fecha',
        'grano', 'filas', 'nota', 'fecha', 'fecha_key', 'fecha_completa',
        'fecha_venta', 'nombre_sede', 'sucursal', 'nombre_dia_semana',
        'nombre_dia', 'hora', 'mes'
    }
    
    for match in matches:
        if match.lower() not in non_metric_words:
            metrics.append(match)
    
    return metrics


def _infer_dimensions(metrics: list[str], description: str) -> list[str]:
    """Infiere dimensiones basándose en métricas y descripción."""
    dimensions = set()
    
    # Palabras clave para dimensiones
    dim_keywords = {
        'sede': ['sede', 'sucursal', 'local', 'tienda', 'plaza', 'ubicacion', 'ubicación', 'store', 'branch'],
        'producto': ['producto', 'sku', 'articulo', 'artículo', 'item', 'descripcion'],
        'categoria': ['categoria', 'categoría', 'categoria_nueva', 'subcategoria'],
        'fecha': ['fecha', 'fecha_completa', 'fecha_venta', 'fecha_key', 'mes', 'periodo'],
        'hora': ['hora', 'franja_horaria', 'horario'],
    }
    
    # Buscar en métricas
    for metric in metrics:
        metric_lower = metric.lower()
        for dim, keywords in dim_keywords.items():
            if any(kw in metric_lower for kw in keywords):
                dimensions.add(dim)
    
    # Buscar en descripción
    description_lower = description.lower()
    for dim, keywords in dim_keywords.items():
        if any(kw in description_lower for kw in keywords):
            dimensions.add(dim)
    
    # Si no se encontraron dimensiones, usar 'general'
    if not dimensions:
        dimensions.add('general')
    
    return list(dimensions)


def _infer_temporal_type(view_name: str, description: str) -> str:
    """Determina el tipo temporal de la vista."""
    combined = f"{view_name} {description}".lower()
    
    if any(w in combined for w in ["latest", "hoy", "actual", "today", "snapshot", "current"]):
        return "current"
    elif any(w in combined for w in ["histor", "history", "tendencia", "evolución", "time_series", "histórico"]):
        return "historical"
    elif any(w in combined for w in ["week", "semana", "comparativa", "compare"]):
        return "periodic"
    elif any(w in combined for w in ["dashboard"]):
        return "dashboard"
    else:
        return "general"


def _infer_time_scope(description: str) -> str:
    """Determina el alcance temporal de la vista."""
    desc_lower = description.lower()
    
    if any(w in desc_lower for w in ["hora", "horaria", "hourly"]):
        return "hourly"
    elif any(w in desc_lower for w in ["día", "diario", "fecha", "daily", "dia"]):
        return "daily"
    elif any(w in desc_lower for w in ["semana", "semanal", "weekly"]):
        return "weekly"
    elif any(w in desc_lower for w in ["mes", "mensual", "monthly"]):
        return "monthly"
    else:
        return "unknown"


def _parse_agents_md(file_path: Path) -> list[Document]:
    """Parsea el archivo AGENTS.md y crea documentos para Chroma."""
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
        
        # Si no hay métricas, usar un placeholder
        if not metrics:
            metrics = [f"metrica_{view_name}"]
        
        # Inferir dimensiones
        dimensions = _infer_dimensions(metrics, description)
        
        # Valores por defecto
        if not purpose:
            purpose = f"Vista {view_name}"
        if not grain:
            grain = "general"
        
        # Clasificación temporal
        temporal_type = _infer_temporal_type(view_name, description)
        time_scope = _infer_time_scope(description)
        
        # Detección de filtros
        dims_lower = [d.lower() for d in dimensions]
        supports_date_filter = any(d in dims_lower for d in ["fecha", "date", "periodo", "mes", "año", "dia", "día", "hora"])
        supports_location_filter = any(d in dims_lower for d in ["sede", "local", "sucursal", "ubicacion", "ubicación", "store", "branch"])
        
        # Construir keywords
        keywords_str = f"{view_name} {purpose} " + " ".join(metrics[:5])
        
        # Crear contenido
        page_content = (
            f"VISTA SEMÁNTICA: semantic.{view_name}\n"
            f"NOMBRE TÉCNICO: {view_name}\n"
            f"CONTEXTO DE NEGOCIO Y REGLAS:\n{description}\n"
            f"KEYWORDS PARA BÚSQUEDA: {keywords_str}"
        )
        
        # *** IMPORTANTE: Convertir listas a strings para Chroma ***
        metrics_str = ", ".join(metrics[:15])
        dimensions_str = ", ".join(dimensions[:10])
        
        # Crear ID único
        doc_id = _make_id(f"semantic.{view_name}")
        
        # Crear documento con metadatos válidos
        documents.append(Document(
            page_content=page_content,
            metadata={
                "view_name": f"semantic.{view_name}",
                "schema": "semantic",
                "keywords": keywords_str,
                "purpose": purpose[:500],
                "grain": grain,
                "metrics": metrics_str,  # String en lugar de lista
                "dimensions": dimensions_str,  # String en lugar de lista
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
    """Función principal para indexar las vistas semánticas en Chroma."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY no encontrada. Revisa que tu archivo .env tenga:\n"
            'OPENAI_API_KEY=sk-...\n'
            f"O defínela como variable de entorno antes de ejecutar. (Buscando en: {env_path})"
        )

    print(f"📖 Leyendo catálogo desde: {DATA_PATH}")
    print(f"📁 Directorio Chroma: {CHROMA_DIR}")
    
    docs = _parse_agents_md(DATA_PATH)
    print(f"🧩 Vistas parseadas: {len(docs)}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    
    # Agregar documentos
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
    
    # Verificación rápida
    if count > 0:
        print("\n📊 Verificación de documentos indexados:")
        try:
            all_docs = vector_store._collection.get()
            for i, (doc_id, metadata) in enumerate(zip(all_docs["ids"], all_docs["metadatas"])):
                view_name = metadata.get("view_name", "unknown")
                metrics_count = len(metadata.get("metrics", "").split(",")) if metadata.get("metrics") else 0
                print(f"  {i+1}. {view_name} | ID: {doc_id[:12]}... | métricas: {metrics_count}")
        except Exception as e:
            print(f"  [WARN] No se pudo verificar: {e}")


if __name__ == "__main__":
    indexar()