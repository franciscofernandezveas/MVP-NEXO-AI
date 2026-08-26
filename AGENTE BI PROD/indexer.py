# indexer.py
# Ubicación: AGENTE BI PROD/indexer.py
# -------------------------------------------------
# Indexador del business memory — ÚNICO componente autorizado a ESCRIBIR el índice.
#
#   python indexer.py                 → indexa solo si hay cambios (idempotente)
#   python indexer.py reindex --force → rebuild completo
#   python indexer.py status          → estado del índice (sin API key)
#   python indexer.py verify          → inspección de contenido (sin API key)
#
# Flujo recomendado: correr en deploy/CI o tras editar AGENTS.md.
# core/rag.py NUNCA reindexa: en runtime solo lee y advierte si está stale.
# -------------------------------------------------
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from core.chunking import CHUNKING_VERSION, build_chunks, chunk_stats
from core.rag_store import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_AGENTS_MD_PATH,
    EMBEDDING_MODEL,
    HNSW_SPACE,
    desired_manifest,
    manifest_mismatches,
    read_manifest,
    write_manifest,
)

logger = logging.getLogger(__name__)


def _load_env() -> None:
    """Carga .env solo cuando se ejecuta como indexador (sin side-effects al importar)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)
    except ImportError:
        pass  # las variables pueden venir del entorno


def _require_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY no encontrada. Define la variable de entorno o "
            "agrégala al .env junto a indexer.py."
        )


def _raw_count_and_space() -> Tuple[Optional[int], Optional[str]]:
    """Count + hnsw:space leídos con cliente crudo (no requiere OPENAI_API_KEY)."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        col = client.get_collection(COLLECTION_NAME)
        return col.count(), (col.metadata or {}).get("hnsw:space")
    except Exception:
        return None, None


def index_status(source: Optional[Path] = None) -> Dict[str, Any]:
    """Estado del índice vs lo requerido. Usable desde CLI, CI o healthchecks."""
    agents_md = Path(source) if source else DEFAULT_AGENTS_MD_PATH
    count, space = _raw_count_and_space()
    current = read_manifest(CHROMA_DIR, COLLECTION_NAME)
    if agents_md.exists():
        desired = desired_manifest(agents_md, EMBEDDING_MODEL)
        mismatches = manifest_mismatches(current, desired)
    else:
        desired = None
        mismatches = [f"AGENTS.md no encontrado (buscado en {agents_md})"]
    if space != HNSW_SPACE:
        mismatches.append(f"hnsw:space de la colección: '{space}' ≠ '{HNSW_SPACE}'")
    if not count:
        mismatches.append("colección vacía o inexistente")
    return {
        "collection": COLLECTION_NAME,
        "chroma_dir": str(CHROMA_DIR),
        "source": str(agents_md),
        "doc_count": count,
        "hnsw_space": space,
        "manifest": current,
        "desired": desired,
        "is_current": not mismatches,
        "stale_reasons": mismatches,
        "fix": "python indexer.py reindex --force",
    }


def index_documents(force: bool = False, source: Optional[Path] = None) -> Dict[str, Any]:
    """
    Pipeline de escritura (idempotente):
      1. Si manifest vigente == requerido y colección sana → skip.
      2. Si no → build_chunks → recrear colección (espacio coseno garantizado)
         → add_documents → escribir manifiesto.
    Devuelve stats consumibles programáticamente.
    """
    _load_env()
    _require_api_key()

    agents_md = Path(source) if source else DEFAULT_AGENTS_MD_PATH
    if not agents_md.exists():
        raise FileNotFoundError(f"No se encontró AGENTS.md (buscado en {agents_md})")

    status = index_status(source=agents_md)
    if not force and status["is_current"]:
        logger.info(f"[Indexer] Índice vigente ({status['doc_count']} chunks). Sin cambios.")
        return {"action": "skipped", "reason": "índice vigente", **status}

    logger.info(f"[Indexer] Reindexando. Motivos: {status['stale_reasons'] or ['--force']}")
    logger.info(f"[Indexer] Fuente: {agents_md} | Colección: {COLLECTION_NAME} | "
                f"Modelo: {EMBEDDING_MODEL} | Chunker: {CHUNKING_VERSION}")

    docs = build_chunks(agents_md.read_text(encoding="utf-8-sig"))
    if not docs:
        logger.error("[Indexer] build_chunks devolvió 0 documentos.")
        return {"action": "error", "reason": "0 chunks generados", **status}

    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # Recreación completa: garantiza hnsw:space correcto y cero basura obsoleta
    # (collection_metadata solo aplica al CREAR la colección).
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": HNSW_SPACE},
    )
    try:
        store.delete_collection()
    except Exception as e:
        logger.debug(f"[Indexer] delete_collection: {e}")
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": HNSW_SPACE},
    )
    store.add_documents(documents=docs, ids=[d.id for d in docs])

    manifest = {
        **desired_manifest(agents_md, EMBEDDING_MODEL),
        "doc_count": len(docs),
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "chunk_stats": chunk_stats(docs),
    }
    write_manifest(CHROMA_DIR, COLLECTION_NAME, manifest)

    stats = {"action": "reindexed", "doc_count": len(docs), "manifest": manifest}
    logger.info(f"[Indexer] Indexación completada: {len(docs)} chunks.")
    return stats


# Compat retro: hooks/scripts que llamaban indexar() del indexer anterior
def indexar(force: bool = False) -> Dict[str, Any]:
    return index_documents(force=force)


def _cmd_verify() -> None:
    """Inspección del contenido indexado (sin API key)."""
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_collection(COLLECTION_NAME)
    print(f"Colección '{COLLECTION_NAME}': {col.count()} chunks | "
          f"hnsw:space={((col.metadata or {}).get('hnsw:space'))}")
    for doc_type in ("view", "metric", "table_row", "section"):
        try:
            res = col.get(where={"doc_type": doc_type},
                          include=["metadatas", "documents"], limit=1)
            if res["ids"]:
                md = res["metadatas"][0] or {}
                label = md.get("view_name") or md.get("column") or md.get("section") or "?"
                preview = (res["documents"][0] or "")[:120].replace("\n", " ")
                print(f"  · {doc_type:10s} ej: {label}")
                print(f"      {preview}…")
        except Exception as e:
            print(f"  · {doc_type:10s} (sin datos o error: {e})")


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Indexador del business memory (único que escribe el índice)")
    sub = parser.add_subparsers(dest="command")

    p_re = sub.add_parser("reindex", help="Indexa si hay cambios (o siempre con --force)")
    p_re.add_argument("--force", action="store_true", help="Rebuild completo")
    p_re.add_argument("--source", type=Path, default=None,
                      help="Ruta alternativa de AGENTS.md")
    sub.add_parser("status", help="Estado del índice vs lo requerido")
    sub.add_parser("verify", help="Inspección del contenido indexado")

    args = parser.parse_args(argv)
    command = args.command or "reindex"  # default retro-compatible: `python indexer.py`

    if command == "reindex":
        stats = index_documents(force=getattr(args, "force", False),
                                source=getattr(args, "source", None))
        print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))
    elif command == "status":
        print(json.dumps(index_status(), indent=2, ensure_ascii=False, default=str))
    elif command == "verify":
        _cmd_verify()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
