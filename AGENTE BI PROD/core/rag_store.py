# core/rag_store.py
# -------------------------------------------------
# Configuración y manifiesto del índice de conocimiento (AGENTS.md → Chroma).
#
# Fuente ÚNICA de verdad para:
#   · rutas/constantes del store (colección, modelo de embeddings, espacio HNSW),
#   · resolución de la ubicación de AGENTS.md,
#   · formato y comparación del manifiesto de versionado.
#
# NO abre ni modifica la colección:
#   · escritura → indexer.py
#   · lectura   → core/rag.py
# Ambos importan de aquí → imposible que diverjan.
# -------------------------------------------------
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.chunking import CHUNKING_VERSION

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db")))
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "bi_knowledge")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
HNSW_SPACE = "cosine"


def resolve_agents_md() -> Path:
    """Política de búsqueda del AGENTS.md (misma que usaba el retriever anterior)."""
    candidates = [
        Path(os.getenv("AGENTS_MD_PATH") or os.getenv("DEFAULT_AGENTS_MD_PATH") or ""),
        BASE_DIR / "AGENTS.md",
        Path("/app/AGENTS.md"),
        Path("/app/AGENTE BI PROD/AGENTS.md"),
        Path("/data/AGENTS.md"),
        Path.cwd() / "AGENTS.md",
    ]
    for p in candidates:
        if p and str(p) != "." and p.exists():
            return p
    return BASE_DIR / "AGENTS.md"  # para mensaje de error legible


DEFAULT_AGENTS_MD_PATH = resolve_agents_md()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------- manifiesto ----------------
def manifest_path(persist_directory: Path, collection_name: str) -> Path:
    return Path(persist_directory) / f"{collection_name}.manifest.json"


def read_manifest(persist_directory: Path, collection_name: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(manifest_path(persist_directory, collection_name)
                          .read_text(encoding="utf-8"))
    except Exception:
        return None


def write_manifest(persist_directory: Path, collection_name: str,
                   manifest: Dict[str, Any]) -> None:
    p = Path(persist_directory)
    p.mkdir(parents=True, exist_ok=True)
    manifest_path(p, collection_name).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def desired_manifest(agents_md_path: Path, embedding_model: str) -> Dict[str, Any]:
    """Lo que el índice DEBERÍA tener. Si difiere del manifest en disco → stale."""
    return {
        "source_sha256": sha256_file(agents_md_path),
        "embedding_model": embedding_model,
        "chunking_version": CHUNKING_VERSION,
        "hnsw_space": HNSW_SPACE,
    }


def manifest_mismatches(current: Optional[Dict[str, Any]],
                        desired: Dict[str, Any]) -> List[str]:
    """Razones legibles de por qué el índice está desactualizado."""
    if not current:
        return ["sin manifiesto (índice nunca construido o anterior al versionado)"]
    out = []
    for key, want in desired.items():
        got = current.get(key)
        if got != want:
            out.append(f"{key}: vigente '{str(got)[:12]}…' ≠ requerido '{str(want)[:12]}…'")
    return out
