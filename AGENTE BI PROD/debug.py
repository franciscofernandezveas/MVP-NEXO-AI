# debug_rag.py  (raíz del proyecto)
import json
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# ---------- 0. Resolución de rutas (¿indexador y lector miran lo mismo?) ----------
from core.rag_store import CHROMA_DIR, COLLECTION_NAME, DEFAULT_AGENTS_MD_PATH, EMBEDDING_MODEL
print("\n== 0. PATHS ==")
print("CHROMA_DIR :", CHROMA_DIR)
print("AGENTS.md  :", DEFAULT_AGENTS_MD_PATH, "| existe:", DEFAULT_AGENTS_MD_PATH.exists())
print("COLLECTION :", COLLECTION_NAME, "| embedding model:", EMBEDDING_MODEL)

# ---------- 1. Salud del índice ----------
from core.rag import get_rag
rag = get_rag()
health = rag.index_health()
print("\n== 1. HEALTH ==")
print(json.dumps({k: v for k, v in health.items() if k != "manifest"},
                 indent=2, ensure_ascii=False, default=str))

# ---------- 2. ¿Qué quedó realmente indexado? ----------
print("\n== 2. CONTENIDO INDEXADO ==")
data = rag._store.get(include=["documents", "metadatas"])
types = Counter((md or {}).get("doc_type") for md in data.get("metadatas", []))
print("doc_types:", dict(types))
enriched = sum(1 for t in data.get("documents", [])
               if "Intenciones y preguntas de negocio asociadas" in (t or ""))
print(f"chunks 'view' enriquecidos con taxonomía: {enriched}")
print("  → si doc_types no tiene 'table_row' o enriched == 0: el índice NO es v2")

# ---------- 3. ¿BM25 activo? ----------
rag._ensure_bm25()
print("\n== 3. BM25 ==", "OK" if rag._bm25 is not None else "NO → pip install rank_bm25")

# ---------- 4. Sondeo de las queries que fallaban ----------
from core.rag import obtener_candidatas_vistas
print("\n== 4. SONDEOS ==")
for q in ["qué producto se vendió más en junio",
          "unidades vendidas por sucursal últimos 90 días"]:
    print(f"\nQUERY: {q}")
    print(" -- hits híbridos (doc-level):")
    for h in rag.search_hybrid(q, k=6, doc_types=("view", "table_row", "metric")):
        md = h["document"].metadata
        label = md.get("view_name") or md.get("column") or md.get("section") or "?"
        prev = h["document"].page_content[:100].replace("\n", " ")
        print(f"    [rrf={h['rrf']:.4f} cos={h['score']:.3f}] ({md.get('doc_type')}) {label}")
        print(f"        {prev}…")
    cands = obtener_candidatas_vistas(q, k=5)
    print(" -- candidatas agregadas:",
          [(c["view_name"], round(c["score"], 3)) for c in cands] or "NINGUNA")
