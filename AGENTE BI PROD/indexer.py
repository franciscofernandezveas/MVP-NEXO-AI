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
DATA_PATH = BASE_DIR / "data.txt"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "semantic_views"


def _make_id(view_name: str) -> str:
    return hashlib.sha256(view_name.encode()).hexdigest()[:16]


def _extract_field(text: str, field_name: str) -> str:
    """Extrae un campo del markdown tipo **Campo**: valor."""
    pattern = rf'\*\*{re.escape(field_name)}\*\*[:：]\s*(.*?)(?=\n\*\*|\Z)'
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


def _parse_data_txt(file_path: Path) -> list[Document]:
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró {file_path}")

    raw = file_path.read_text(encoding="utf-8-sig")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = re.sub(r'\n##\s*##\s*', '\n## ', raw)
    
    parts = re.split(r'\n##\s+', raw)
    documents = []

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.lower().startswith("descripción de vistas"):
            continue
        
        lines = [ln.rstrip() for ln in part.splitlines() if ln.strip()]
        if not lines:
            continue
        
        view_name = lines[0].strip().strip('*').strip()
        if not re.match(r'^[A-Za-z0-9_]+$', view_name):
            m = re.match(r'([A-Za-z0-9_]+)', view_name)
            if m:
                view_name = m.group(1)
            else:
                print(f"[WARN] Saltando bloque sin nombre de vista válido: {lines[0][:60]}")
                continue
        
        description = "\n".join(lines[1:]).strip()
        
        # Extraer campos estructurados del markdown
        purpose = _extract_field(description, "Propósito")
        grain = _extract_field(description, "Granularidad")
        metrics = _extract_list(description, "Métricas")
        dimensions = _extract_list(description, "Dimensiones")
        keywords = _extract_list(description, "Palabras clave")
        notes = _extract_field(description, "Notas")
        
        # Heurísticas de clasificación temporal
        view_lower = view_name.lower()
        purpose_lower = purpose.lower()
        grain_lower = grain.lower()
        dims_lower = [d.lower() for d in dimensions]
        
        temporal_type = "general"
        if any(w in view_lower for w in ["latest", "hoy", "actual", "today"]):
            temporal_type = "current"
        elif any(w in view_lower or w in purpose_lower for w in ["historico", "histórico", "tendencia", "historial", "time_series"]):
            temporal_type = "historical"
        elif any(w in view_lower or w in purpose_lower for w in ["semana", "comparativa", "vs", "anterior", "pasada"]):
            temporal_type = "periodic"
        
        time_scope = "unknown"
        if any(w in grain_lower for w in ["día", "diario", "fecha", "daily"]):
            time_scope = "daily"
        elif any(w in grain_lower for w in ["semana", "semanal", "weekly"]):
            time_scope = "weekly"
        elif any(w in grain_lower for w in ["mes", "mensual", "monthly"]):
            time_scope = "monthly"
        elif any(w in grain_lower for w in ["hora", "horaria", "hourly"]):
            time_scope = "hourly"
        
        supports_date_filter = any(d in dims_lower for d in ["fecha", "date", "periodo", "mes", "año", "dia", "día"])
        supports_location_filter = any(d in dims_lower for d in ["sede", "local", "sucursal", "ubicacion", "ubicación", "store", "branch"])
        
        # Construir keywords string para búsqueda
        keywords_str = " ".join(keywords) if keywords else view_name.replace('_', ' ')
        
        page_content = (
            f"VISTA SEMÁNTICA: semantic.{view_name}\n"
            f"NOMBRE TÉCNICO: {view_name}\n"
            f"CONTEXTO DE NEGOCIO Y REGLAS:\n{description}\n"
            f"KEYWORDS PARA BÚSQUEDA: {keywords_str}"
        )
        
        documents.append(Document(
            page_content=page_content,
            metadata={
                "view_name": view_name,
                "schema": "semantic",
                "keywords": keywords_str,
                "purpose": purpose,
                "grain": grain,
                "metrics": metrics,
                "dimensions": dimensions,
                "notes": notes,
                "temporal_type": temporal_type,
                "time_scope": time_scope,
                "supports_date_filter": supports_date_filter,
                "supports_location_filter": supports_location_filter,
            },
            id=_make_id(view_name)
        ))
        print(f"[DEBUG] Indexada: semantic.{view_name} | temporal: {temporal_type} | scope: {time_scope} | date_filter: {supports_date_filter} | loc_filter: {supports_location_filter}")
    
    if not documents:
        raise ValueError(
            "No se parseó ninguna vista. Revisa que cada vista inicie con '## nombre_vista'."
        )
    
    return documents


def indexar():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY no encontrada. Revisa que tu archivo .env tenga:\n"
            'OPENAI_API_KEY=sk-...\n'
            f"O defínela como variable de entorno antes de ejecutar. (Buscando en: {env_path})"
        )

    print(f"📖 Leyendo: {DATA_PATH}")
    docs = _parse_data_txt(DATA_PATH)
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
