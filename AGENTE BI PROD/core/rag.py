# core/rag.py
# -------------------------------------------------
# RAG sobre AGENTS.md — capa de RECUPERACIÓN (solo lectura).
#
#  - La INDEXACIÓN vive en indexer.py (index offline / serve online).
#    Este módulo NUNCA escribe en la colección: si el índice está stale
#    lo reporta en los logs y sirve en modo degradado.
#  - Hot-reload: si el manifiesto cambia en disco (indexer corrió fuera
#    del proceso), reabre la colección e invalida el índice léxico.
#  - Config y manifiesto compartidos con el indexer vía core/rag_store
#    (fuente única → imposible divergencia como la del indexer legado).
#  - Búsqueda híbrida: coseno (Chroma) + BM25 (corpus en memoria) fusionados
#    con Reciprocal Rank Fusion (Cormack et al., 2009). Sin listas de
#    términos de negocio: el IDF de BM25 maneja stopwords estadísticamente.
#  - Selección de vistas por AGREGACIÓN DE EVIDENCIA sobre chunks
#    view / table_row / metric y sus co-referencias estructurales.
#  - API compatible con el retriever anterior: harness.py solo cambia el import.
#
#  Dependencia opcional: rank_bm25 (pip install rank_bm25).
#  Si falta, degrada a búsqueda solo-vectorial con warning.
# -------------------------------------------------
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from core.rag_store import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_AGENTS_MD_PATH,
    EMBEDDING_MODEL,
    HNSW_SPACE,
    desired_manifest,
    manifest_mismatches,
    manifest_path,
    read_manifest,
)

logger = logging.getLogger(__name__)

# Constante estándar de Reciprocal Rank Fusion (Cormack et al., 2009)
_RRF_K = 60


# ------------------------------------------------------------------
# 1) Utilidades de recuperación
# ------------------------------------------------------------------
def _tokenize(text: str) -> List[str]:
    """Tokenización para BM25. Sin stopword lists: el IDF las penaliza solo."""
    return re.findall(r"[a-z0-9áéíóúñü_]+", text.lower())


def _content_key(text: str) -> str:
    """Clave estable por contenido (alinea hits vectoriales y léxicos en el RRF)."""
    return hashlib.sha1(text.encode()).hexdigest()


def _normalize_view_name(view: str) -> List[str]:
    clean = (view or "").strip()
    if not clean:
        return []
    if clean.startswith("semantic."):
        return list({clean, clean[len("semantic."):]} )
    return list({clean, f"semantic.{clean}"})


def _build_filter(
    doc_types: Optional[Union[str, Sequence[str]]] = None,
    view_names: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, Any]]:
    clauses: List[Dict[str, Any]] = []
    if doc_types:
        dts = [doc_types] if isinstance(doc_types, str) else list(doc_types)
        clauses.append({"doc_type": dts[0]} if len(dts) == 1
                       else {"$or": [{"doc_type": dt} for dt in dts]})
    if view_names:
        variants = sorted({v for name in view_names for v in _normalize_view_name(name)})
        if variants:
            clauses.append({"view_name": {"$in": variants}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _cosine_similarity(distance: Any) -> float:
    """Chroma cosine distance = 1 - cos → similitud ∈ [0,1] interpretable."""
    try:
        return max(0.0, min(1.0, 1.0 - float(distance)))
    except (TypeError, ValueError):
        return 0.0


# ------------------------------------------------------------------
# 2) Store de solo lectura con búsqueda híbrida y hot-reload
# ------------------------------------------------------------------
class BusinessRAG:
    """
    Acceso de lectura al índice construido por indexer.py.

    No reindexa. index_health() reporta el estado y get_rag() advierte cómo
    corregirlo. La primera apertura puede crear el CONTENEDOR vacío de la
    colección (limitación de Chroma) — nunca indexa documentos.
    """

    def __init__(self, persist_directory: Path, collection_name: str,
                 embedding_model: str, agents_md_path: Path):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.agents_md_path = Path(agents_md_path)
        self._embeddings = OpenAIEmbeddings(model=embedding_model)
        self._store = self._open_store()
        # Índice léxico lazy (se construye en el primer search_hybrid)
        self._bm25 = None
        self._bm25_corpus: List[Tuple[str, Document]] = []
        self._bm25_unavailable = False
        # Hot-reload: firma del manifiesto observado por este proceso
        self._reload_lock = threading.Lock()
        self._manifest_sig = self._current_manifest_sig()

    @classmethod
    def from_env(cls) -> "BusinessRAG":
        return cls(CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, DEFAULT_AGENTS_MD_PATH)

    def _open_store(self) -> Chroma:
        # collection_metadata solo aplica si la colección se CREA aquí (contenedor
        # vacío). Si indexer.py ya la creó, se abre con su configuración (coseno).
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(self.persist_directory),
            collection_metadata={"hnsw:space": HNSW_SPACE},
        )

    def _count(self) -> int:
        try:
            return self._store._collection.count()
        except Exception:
            return 0

    # ---------------- salud del índice (reporta, no corrige) ----------------
    def index_health(self) -> Dict[str, Any]:
        """Estado del índice vs lo requerido. La corrección es del indexer."""
        current = read_manifest(self.persist_directory, self.collection_name)
        mismatches: List[str] = []
        if self.agents_md_path.exists():
            desired = desired_manifest(self.agents_md_path, self.embedding_model)
            mismatches = manifest_mismatches(current, desired)
        else:
            desired = None
            mismatches = [f"AGENTS.md no encontrado (buscado en {self.agents_md_path})"]

        meta = self._store._collection.metadata or {}
        space = meta.get("hnsw:space")
        count = self._count()
        if count > 0 and space != HNSW_SPACE:
            mismatches.append(f"hnsw:space de la colección: '{space}' ≠ '{HNSW_SPACE}'")
        if count == 0:
            mismatches.append("colección vacía o inexistente")

        return {
            "collection": self.collection_name,
            "persist_directory": str(self.persist_directory),
            "doc_count": count,
            "hnsw_space": space,
            "manifest": current,
            "desired": desired,
            "is_current": not mismatches,
            "stale_reasons": mismatches,
            "fix": "python indexer.py reindex --force",
        }

    # ---------------- hot-reload (indexer corrió fuera del proceso) ----------------
    def _current_manifest_sig(self) -> Optional[int]:
        try:
            return manifest_path(self.persist_directory, self.collection_name).stat().st_mtime_ns
        except OSError:
            return None

    def _ensure_fresh(self) -> None:
        """
        Si el manifiesto en disco cambió (reindex externo), reabre la colección
        y descarta el índice léxico cacheado. Costo por llamada: un stat().
        """
        with self._reload_lock:
            sig = self._current_manifest_sig()
            if sig != self._manifest_sig:
                logger.info("[RAG] Manifiesto cambió en disco → hot-reload "
                            "(reabrir colección + invalidar BM25).")
                try:
                    self._store = self._open_store()
                except Exception as e:
                    logger.warning(f"[RAG] Reapertura falló (¿reindex en curso?): {e}")
                    return  # sirve con el store previo; el próximo intento reintenta
                self._bm25 = None
                self._bm25_corpus = []
                self._manifest_sig = sig

    # ---------------- acceso puntual ----------------
    def fetch_view_document(self, view_name: str) -> Optional[Document]:
        """Recupera el chunk 'view' completo por nombre (definición canónica).
        Se usa cuando una vista gana por evidencia indirecta pero su chunk no
        rankeó: el agente SQL necesita sus columnas igualmente."""
        try:
            res = self._store._collection.get(
                where={"view_name": view_name},
                include=["documents", "metadatas"],
                limit=1,
            )
            docs = res.get("documents") or []
            metas = res.get("metadatas") or []
            if docs:
                return Document(page_content=docs[0], metadata=metas[0] or {})
        except Exception as e:
            logger.debug(f"[RAG] fetch_view_document('{view_name}'): {e}")
        return None

    # ---------------- índice léxico (BM25) ----------------
    def _ensure_bm25(self) -> None:
        if self._bm25 is not None or self._bm25_unavailable:
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("[RAG] rank_bm25 no instalado → búsqueda solo vectorial. "
                           "Instala con: pip install rank_bm25")
            self._bm25_unavailable = True
            return
        # Corpus pequeño (decenas de chunks) → se materializa desde Chroma
        data = self._store.get(include=["documents", "metadatas"])
        self._bm25_corpus = [
            (_content_key(t or ""), Document(page_content=t or "", metadata=md or {}))
            for t, md in zip(data.get("documents", []), data.get("metadatas", []))
        ]
        self._bm25 = (BM25Okapi([_tokenize(d.page_content) for _, d in self._bm25_corpus])
                      if self._bm25_corpus else None)

    # ---------------- búsqueda vectorial (base) ----------------
    def search(
        self,
        query: str,
        k: int = 5,
        doc_types: Optional[Union[str, Sequence[str]]] = None,
        view_names: Optional[Sequence[str]] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        self._ensure_fresh()
        kwargs: Dict[str, Any] = {"k": k}
        if f := _build_filter(doc_types, view_names):
            kwargs["filter"] = f
        try:
            hits = self._store.similarity_search_with_score(query, **kwargs)
        except Exception as e:
            logger.warning(f"[RAG] Búsqueda falló: {e}")
            return []

        results = []
        for doc, distance in hits:
            sim = _cosine_similarity(distance)
            if sim < min_score:
                continue
            results.append({"document": doc, "score": sim, "raw_distance": float(distance)})
        results.sort(key=lambda h: h["score"], reverse=True)
        return results

    # ---------------- búsqueda híbrida (vector + BM25, RRF) ----------------
    def search_hybrid(
        self,
        query: str,
        k: int = 8,
        doc_types: Optional[Union[str, Sequence[str]]] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Fusiona ranking vectorial (coseno) y léxico (BM25) con RRF.
          · 'score' → similitud coseno ∈ [0,1] (0.0 si el hit llegó solo por BM25)
          · 'rrf'   → evidencia fusionada (mayor = mejor)
        """
        self._ensure_fresh()
        fetch = max(k * 3, 16)
        vec_hits = self.search(query, k=fetch, doc_types=doc_types, min_score=min_score)
        vec_by_key = {_content_key(h["document"].page_content): h for h in vec_hits}

        rankings: List[List[str]] = [list(vec_by_key)]
        self._ensure_bm25()
        if self._bm25 is not None:
            dts = ({doc_types} if isinstance(doc_types, str)
                   else set(doc_types) if doc_types else None)
            scores = self._bm25.get_scores(_tokenize(query))
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            lex_rank = [
                self._bm25_corpus[i][0]
                for i in order
                if scores[i] > 0
                and (dts is None or self._bm25_corpus[i][1].metadata.get("doc_type") in dts)
            ][:fetch]
            rankings.append(lex_rank)

        fused: Dict[str, float] = defaultdict(float)
        for ranking in rankings:
            for rank, key in enumerate(ranking):
                fused[key] += 1.0 / (_RRF_K + rank + 1)

        corpus_lookup = dict(self._bm25_corpus)
        out: List[Dict[str, Any]] = []
        for key, rrf in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]:
            doc = vec_by_key.get(key, {}).get("document") or corpus_lookup.get(key)
            if doc is None:
                continue
            out.append({
                "document": doc,
                "rrf": rrf,
                "score": vec_by_key.get(key, {}).get("score", 0.0),
            })
        return out


# ------------------------------------------------------------------
# 3) Singleton thread-safe (sin side-effects en import)
# ------------------------------------------------------------------
_INSTANCE: Optional[BusinessRAG] = None
_LOCK = threading.Lock()


def get_rag() -> BusinessRAG:
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                if not os.getenv("OPENAI_API_KEY"):
                    logger.error(
                        "[RAG] OPENAI_API_KEY no está configurada. "
                        "El lector de embeddings requiere la misma API key que el indexer. "
                        "Actívala con: set OPENAI_API_KEY=... (PowerShell) o agrégala al .env."
                    )
                rag = BusinessRAG.from_env()
                health = rag.index_health()
                if health["doc_count"] == 0:
                    logger.error(f"[RAG] Índice vacío. Construye el índice con: {health['fix']}")
                elif not health["is_current"]:
                    logger.warning(
                        f"[RAG] Índice desactualizado: {health['stale_reasons']}. "
                        f"Sirviendo en modo degradado. Para actualizar: {health['fix']}"
                    )
                else:
                    logger.info(f"[RAG] Índice vigente ({health['doc_count']} chunks).")
                _INSTANCE = rag
    return _INSTANCE


# ------------------------------------------------------------------
# 4) API compatible con el semantic_retriever anterior
#    (harness.py solo debe cambiar el import)
# ------------------------------------------------------------------
def obtener_candidatas_vistas(
    query: str,
    k: int = 5,
    allowed_views: Optional[List[str]] = None,
    min_score_threshold: float = 0.0,
    column_hints: Optional[Dict[str, Any]] = None,   # compat, no usado
    biz_mem: Optional[Any] = None,                   # compat, no usado
) -> List[Dict[str, Any]]:
    """
    Selección de vistas por evidencia fusionada:
      búsqueda híbrida sobre (view + table_row + metric) → RRF por chunk →
      agregación por vista (cada chunk aporta a su vista propia o a las
      vistas que referencia estructuralmente).
    """
    rag = get_rag()
    hits = rag.search_hybrid(
        query,
        k=max(k * 4, 16),
        doc_types=("view", "table_row", "metric"),
        min_score=min_score_threshold,
    )

    allowed: set = set()
    for name in (allowed_views or []):
        allowed |= set(_normalize_view_name(name))

    def _views_of(md: Dict[str, Any]) -> List[str]:
        vs = [md["view_name"]] if md.get("view_name") else []
        for key in ("related_views", "views"):
            vs += [v.strip() for v in (md.get(key) or "").split(",") if v.strip()]
        return sorted({v.removeprefix("semantic.") for v in vs})

    per_view: Dict[str, Dict[str, Any]] = {}
    for h in hits:
        for v in _views_of(h["document"].metadata):
            if allowed and not (set(_normalize_view_name(v)) & allowed):
                continue
            slot = per_view.setdefault(v, {"rrf": 0.0, "view_doc": None, "evidence": []})
            slot["rrf"] += h["rrf"]
            if h["document"].metadata.get("doc_type") == "view":
                slot["view_doc"] = h["document"]
            else:
                slot["evidence"].append(h["document"])

    if not per_view:
        logger.warning(f"[RAG] 0 candidatas sobre umbral {min_score_threshold} "
                       f"para '{query[:50]}...'")
        return []

    max_rrf = max(s["rrf"] for s in per_view.values())
    candidates: List[Dict[str, Any]] = []
    for name, slot in sorted(per_view.items(), key=lambda kv: kv[1]["rrf"], reverse=True)[:k]:
        # Si el chunk de la vista no rankeó, recuperar su definición canónica:
        # la evidencia la seleccionó y el agente SQL necesita sus columnas.
        doc = (slot["view_doc"]
               or rag.fetch_view_document(f"semantic.{name}")
               or (slot["evidence"][0] if slot["evidence"] else None))
        if doc is None:
            continue
        context = doc.page_content
        if slot["view_doc"] is not None and slot["evidence"]:
            context += ("\n\nEvidencia adicional:\n" + "\n".join(
                f"- {e.page_content.splitlines()[-1]}" for e in slot["evidence"][:2]))
        score = slot["rrf"] / max_rrf  # evidencia fusionada normalizada ∈ (0,1]
        if score < min_score_threshold:
            continue
        candidates.append({
            "view_name": f"semantic.{name}",
            "context": context,
            "score": score,
            "original_score": score,
            "raw_distance": 1.0 - score,
            "metadata_boost": 0.0,
            "temporal_context": {},
            "view_metadata": doc.metadata,
        })

    if candidates:
        logger.info(f"[RAG] Top {min(3, len(candidates))} para '{query[:50]}...': "
                    + " | ".join(f"{c['view_name']} ({c['score']:.3f})" for c in candidates[:3]))
    else:
        logger.warning(f"[RAG] 0 candidatas sobre umbral {min_score_threshold} "
                       f"para '{query[:50]}...'")
    return candidates


def obtener_candidatas_detalles(
    query: str,
    k: int = 5,
    allowed_views: Optional[List[str]] = None,
    min_score_threshold: float = 0.0,
    column_hints: Optional[Dict[str, Any]] = None,
    biz_mem: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    candidates = obtener_candidatas_vistas(
        query=query, k=k, allowed_views=allowed_views,
        min_score_threshold=min_score_threshold,
        column_hints=column_hints, biz_mem=biz_mem,
    )
    out = []
    for c in candidates:
        md = c["view_metadata"]
        fields = json.loads(md.get("fields_json") or "{}")
        out.append({
            **c,
            "fields": fields,  # expone TODO lo declarado en el documento (genérico)
            "purpose": next(iter(fields.values()), md.get("keywords", ""))[:200],
            "grain": fields.get("Granularidad") or fields.get("Filas") or "desconocido",
            "metrics": [m for m in md.get("metrics", "").split(", ") if m],
            "dimensions": [],
            "domain": "general",
            "notes": fields.get("Nota", ""),
            "keywords": md.get("keywords", ""),
            "temporal_type": fields.get("Tipo", "general"),
            "time_scope": fields.get("Filtro fecha", "unknown"),
            "usage_examples": [],
            "compatibility": {k2: "ok" for k2 in ("metric", "date_range", "location", "product", "category")},
            "compatibility_reason": f"La vista '{c['view_name']}' puede responder la consulta.",
            "can_answer": True,
        })
    return out


def seleccionar_vista_principal(
    query: str,
    column_hints: Optional[Dict[str, Any]] = None,
    allowed_views: Optional[List[str]] = None,
    min_score_threshold: float = 0.0,
    precomputed_candidates: Optional[List[Dict[str, Any]]] = None,
    biz_mem: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    candidates = precomputed_candidates or obtener_candidatas_detalles(
        query=query, k=5, allowed_views=allowed_views,
        min_score_threshold=min_score_threshold,
        column_hints=column_hints, biz_mem=biz_mem,
    )
    if not candidates:
        logger.info(f"[RAG] Sin vistas candidatas para: '{query}'")
        return None
    best = candidates[0]
    logger.info(f"[RAG] Vista principal: {best['view_name']} (score: {best['score']:.3f})")
    return best


def payload_to_column_hints(payload: Any, original_query: str = "") -> Dict[str, Any]:
    return {
        "metrics": getattr(payload, "metrics", None) or [],
        "dimensions": getattr(payload, "dimensions", None) or [],
        "original_query": original_query or getattr(payload, "task", "") or getattr(payload, "question", ""),
    }


# ------------------------------------------------------------------
# 5) Conocimiento de negocio (reglas, definiciones, taxonomía)
# ------------------------------------------------------------------
def buscar_conocimiento_negocio(
    query: str,
    k: int = 4,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """Recupera reglas, definiciones, métricas y filas de taxonomía (híbrido)
    para inyectar en planner/supervisor/sql_agent."""
    hits = get_rag().search_hybrid(
        query, k=k, doc_types=("metric", "section", "table_row"), min_score=min_score)
    return [{
        "content": h["document"].page_content,
        "score": h["score"],
        "rrf": h["rrf"],
        "doc_type": h["document"].metadata.get("doc_type"),
        "section": h["document"].metadata.get("section", ""),
    } for h in hits]


# ------------------------------------------------------------------
# 6) CLI de consulta: stats / search / candidates
#    (la indexación se hace con indexer.py, NO desde aquí)
# ------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """
    CLI standalone. Carga .env para poder correr fuera de la app principal.
    La app en producción carga sus variables de entorno por su propio entrypoint.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env",
                    override=True)
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        description="Consulta del índice RAG (solo lectura). "
                    "Para indexar: python indexer.py")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stats", help="Salud del índice (manifiesto, espacio, count)")
    p_search = sub.add_parser("search", help="Búsqueda híbrida cruda")
    p_search.add_argument("query")
    p_search.add_argument("--k", type=int, default=5)
    p_search.add_argument("--min-score", type=float, default=0.0)
    p_cand = sub.add_parser("candidates", help="Selección de vistas (híbrida + agregación)")
    p_cand.add_argument("query")
    p_cand.add_argument("--k", type=int, default=5)
    p_cand.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args(argv)

    rag = BusinessRAG.from_env()
    if args.command == "stats":
        print(json.dumps(rag.index_health(), indent=2, ensure_ascii=False, default=str))
    elif args.command == "search":
        for h in rag.search_hybrid(args.query, k=args.k, min_score=args.min_score):
            print(f"[rrf={h['rrf']:.4f} cos={h['score']:.3f}] "
                  f"({h['document'].metadata.get('doc_type')}) "
                  f"{h['document'].metadata.get('view_name') or h['document'].metadata.get('section')}")
            print("   " + h["document"].page_content[:160].replace("\n", " ") + "...")
    elif args.command == "candidates":
        for c in obtener_candidatas_vistas(args.query, k=args.k,
                                           min_score_threshold=args.min_score):
            first = next((l for l in c["context"].splitlines() if l.strip()), "")
            print(f"[{c['score']:.3f}] {c['view_name']}")
            print(f"   {first[:160]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
