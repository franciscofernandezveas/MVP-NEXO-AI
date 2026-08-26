# core/rag.py
# -------------------------------------------------
# RAG sobre AGENTS.md — reemplazo de semantic_retriever.py
#
#  - Chunking estructural (heading tree): SIN listas hardcodeadas.
#      · heading snake_case  → doc_type="view"     (p.ej. ### sales_review_day)
#      · resto de headings   → doc_type="section"  (reglas, definiciones, etc.)
#      · columnas `- \`col\`: def` dentro de vistas → doc_type="metric" (puente)
#      · filas de tablas que mapean a vistas       → doc_type="intent" (v2)
#        (la columna-vista se detecta por CONTENIDO: identificadores snake_case
#         que existen como headings de vista en el documento — nunca por nombre)
#  - Vistas enriquecidas con las intenciones/preguntas típicas de la taxonomía
#    → el chunk 'view' matchea lenguaje natural del usuario, no solo jerga técnica.
#  - Índice versionado por manifiesto (sha256 del archivo + modelo + versión
#    de chunker) → reindex automático SOLO cuando hay cambios reales.
#  - Espacio coseno → score ∈ [0,1] interpretable; min_score se APLICA de verdad.
#  - Candidatas de vista con FUSIÓN de evidencias (view + intent votan por vista;
#    los chunks 'metric' NO votan — sirven solo para lookup por nombre técnico).
#  - Carga .env local al CONSTRUIR la instancia (nunca al importar):
#    override=False → en producción ganan las variables reales del entorno.
# -------------------------------------------------
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db")))
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "bi_knowledge")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")

# Bumpear cuando cambie la lógica de chunking → fuerza reindex
CHUNKING_VERSION = "v2"

# Chunks de sección largos se subdividen preservando el breadcrumb
MAX_SECTION_CHARS = 1800


def _resolve_agents_md() -> Path:
    """Misma política de búsqueda que usaba el retriever anterior."""
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


DEFAULT_AGENTS_MD_PATH = _resolve_agents_md()


def _load_env() -> None:
    """Carga el .env del proyecto si existe (desarrollo local).

    - Ruta explícita vía BASE_DIR → inmune al cwd (funciona igual desde
      core/, desde la raíz del proyecto o desde el CLI).
    - override=False → en producción (Railway) las variables reales del
      entorno tienen prioridad sobre el archivo.
    - Se invoca al CONSTRUIR la instancia, no al importar el módulo →
      mantiene la regla de diseño "sin side-effects en import".
    """
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


# ------------------------------------------------------------------
# 1) Chunking estructural — heading tree genérico
# ------------------------------------------------------------------
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]{1,}$")
_COLUMN_ITEM_RE = re.compile(r"^\s*-\s*`([^`\s]+)`\s*:\s*(.+)$")


@dataclass
class _Node:
    level: int
    title: str
    lines: List[str] = field(default_factory=list)
    children: List["_Node"] = field(default_factory=list)


def _parse_heading_tree(md_text: str) -> Tuple[List[str], List[_Node]]:
    preamble: List[str] = []
    root = _Node(level=0, title="__root__")
    stack: List[_Node] = [root]
    for line in md_text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            node = _Node(level=len(m.group(1)), title=m.group(2).strip())
            while stack and stack[-1].level >= node.level:
                stack.pop()
            stack[-1].children.append(node)
            stack.append(node)
        elif len(stack) == 1:
            preamble.append(line)
        else:
            stack[-1].lines.append(line)
    return preamble, root.children


def _is_view_heading(title: str) -> bool:
    """Vista = título completo en snake_case. '3. Vistas Semánticas' no califica."""
    return bool(_IDENTIFIER_RE.fullmatch(title.strip()))


def _slug(text: str, max_len: int = 64) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_")
    return (s or hashlib.sha1(text.encode()).hexdigest()[:12])[:max_len]


# ---------------- tablas GFM (genérico) ----------------
_CELL_SEP_RE = re.compile(r"^:?-+:?$")


def _split_table_row(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_table_separator(line: str) -> bool:
    s = line.strip()
    if not s.startswith("|"):
        return False
    cells = _split_table_row(s)
    return len(cells) >= 2 and all(_CELL_SEP_RE.match(c) for c in cells if c != "")


def _clean_ident(text: str) -> str:
    """Quita backticks/asteriscos para DETECTAR identificadores (no para contenido)."""
    return re.sub(r"[`*]", "", text).strip()


def _extract_tables(
    lines: List[str],
) -> Tuple[List[str], List[Tuple[List[str], List[List[str]], List[str]]]]:
    """Separa tablas GFM (header + separador + filas) del resto del texto.

    Devuelve (lineas_sin_tablas, [(headers, rows, raw_lines)]).
    """
    rest: List[str] = []
    tables: List[Tuple[List[str], List[List[str]], List[str]]] = []
    i = 0
    while i < len(lines):
        if (lines[i].strip().startswith("|")
                and i + 1 < len(lines)
                and _is_table_separator(lines[i + 1])):
            headers = _split_table_row(lines[i])
            raw = [lines[i], lines[i + 1]]
            j = i + 2
            rows: List[List[str]] = []
            while j < len(lines) and lines[j].strip().startswith("|") and lines[j].strip():
                rows.append(_split_table_row(lines[j]))
                raw.append(lines[j])
                j += 1
            if rows:
                tables.append((headers, rows, raw))
            else:
                rest.extend(raw)   # tabla vacía → no es tabla real
            i = j
        else:
            rest.append(lines[i])
            i += 1
    return rest, tables


def _detect_view_column(
    rows: List[List[str]], ncols: int, known_views: set
) -> Optional[int]:
    """La columna-vista es aquella donde la mayoría de celdas son identificadores
    que EXISTEN como headings de vista en el documento.
    Detección por contenido — nunca por nombre de columna."""
    best_i, best_ratio = None, 0.0
    for i in range(ncols):
        vals = [_clean_ident(r[i]) for r in rows if i < len(r)]
        vals = [v for v in vals if v]
        if not vals:
            continue
        ratio = sum(v in known_views for v in vals) / len(vals)
        if ratio > best_ratio:
            best_i, best_ratio = i, ratio
    return best_i if best_ratio >= 0.5 else None


def _intent_rows(
    tables: List[Tuple[List[str], List[List[str]], List[str]]],
    known_views: set,
) -> List[Tuple[str, Dict[str, str]]]:
    """Filas de tablas que mapean a vistas conocidas → (view, {header: valor})."""
    out: List[Tuple[str, Dict[str, str]]] = []
    for headers, rows, _raw in tables:
        vcol = _detect_view_column(rows, len(headers), known_views)
        if vcol is None:
            continue
        for row in rows:
            view = _clean_ident(row[vcol]) if vcol < len(row) else ""
            if view not in known_views:
                continue
            cells = {headers[i]: row[i].strip()
                     for i in range(min(len(headers), len(row))) if headers[i]}
            out.append((view, cells))
    return out


def _cell(cells: Dict[str, str], needle: str) -> str:
    """Busca el valor de una columna por fragmento de encabezado (contrato del doc)."""
    for h, v in cells.items():
        if needle in h.lower():
            return v
    return ""


def _split_long(text: str, max_chars: int) -> List[str]:
    """Subdivide por párrafos; cada parte conserva el encabezado (breadcrumb + heading)."""
    if len(text) <= max_chars:
        return [text]
    header, _, rest = text.partition("\n\n")
    parts: List[str] = []
    buf = header
    for block in rest.split("\n\n"):
        candidate = f"{buf}\n\n{block}"
        if len(candidate) > max_chars and buf != header:
            parts.append(buf)
            buf = f"{header}\n\n{block}"
        else:
            buf = candidate
    if buf.strip():
        parts.append(buf)
    return parts


def build_chunks(md_text: str) -> List[Document]:
    """
    Convierte AGENTS.md en documentos listos para indexar.
    Toda la clasificación se DERIVA de la estructura del archivo — nada hardcodeado.

    Tres pasadas sobre el árbol:
      0) catálogo de vistas conocidas (headings snake_case)
      1) filas de tablas que mapean intenciones → vistas conocidas (intents)
      2) emisión de documentos (view enriquecida / section / intent / metric)
    """
    raw = md_text.replace("\r\n", "\n").replace("\r", "\n")
    preamble, roots = _parse_heading_tree(raw)

    # --- Pasada 0: catálogo de vistas conocidas ---
    known_views: set = set()

    def collect_view_names(node: _Node) -> None:
        if _is_view_heading(node.title):
            known_views.add(node.title)
        for child in node.children:
            collect_view_names(child)

    for r in roots:
        collect_view_names(r)

    # --- Pasada 1: intents por vista (para enriquecer chunks 'view') ---
    intent_index: Dict[str, List[Dict[str, str]]] = {}

    def collect_intents(node: _Node) -> None:
        _, tables = _extract_tables(node.lines)
        for view, cells in _intent_rows(tables, known_views):
            intent_index.setdefault(view, []).append(cells)
        for child in node.children:
            collect_intents(child)

    for r in roots:
        collect_intents(r)

    # --- Pasada 2: emisión ---
    docs: List[Document] = []
    seen_ids: set = set()
    section_seq = 0
    column_index: Dict[str, Dict[str, Any]] = {}

    def add(content: str, doc_type: str, doc_id: str, **metadata) -> None:
        if doc_id in seen_ids:
            doc_id = f"{doc_id}:{hashlib.sha1(content.encode()).hexdigest()[:10]}"
        seen_ids.add(doc_id)
        docs.append(Document(page_content=content.strip(), metadata={
            "doc_type": doc_type, **metadata}, id=doc_id))

    if any(l.strip() for l in preamble):
        add("\n".join(preamble).strip(), "section", "section:preamble", section="__preamble__")

    def visit(node: _Node, path: List[str]) -> None:
        nonlocal section_seq
        crumb = " > ".join(path + [node.title])
        body_lines, tables = _extract_tables(node.lines)
        body = "\n".join(body_lines).strip()

        if _is_view_heading(node.title):
            cols = []
            for line in node.lines:
                m = _COLUMN_ITEM_RE.match(line)
                if m and _IDENTIFIER_RE.fullmatch(m.group(1)):
                    col_name, definition = m.group(1), m.group(2).strip()
                    cols.append(col_name)
                    entry = column_index.setdefault(
                        col_name, {"definition": definition, "views": []})
                    entry["views"].append(f"semantic.{node.title}")

            # Enriquecimiento: inyectar intenciones/preguntas típicas de la taxonomía
            # (lenguaje natural del usuario DENTRO del texto vectorizado).
            extras = intent_index.get(node.title, [])
            enriched = body
            intent_names: List[str] = []
            if extras:
                block = ["", "**Intenciones y lenguaje típico de usuarios (mapeo de taxonomía):**"]
                for cells in extras:
                    kv = "; ".join(f"{h}: {v}" for h, v in cells.items() if v)
                    block.append(f"- {kv}")
                    first_val = next(iter(cells.values()), "")
                    if first_val:
                        intent_names.append(_clean_ident(first_val))
                enriched = f"{body}\n" + "\n".join(block)

            first = extras[0] if extras else {}
            add(
                f"{crumb}\n\n### {node.title}\n{enriched}",
                "view",
                f"view:{_slug(node.title)}",
                section=path[-1] if path else "",
                view_name=f"semantic.{node.title}",       # compat con metadata anterior
                metrics=", ".join(cols[:20]),             # compat con metadata anterior
                keywords=body[:500],                      # compat con metadata anterior
                intents=", ".join(intent_names)[:500],
                date_policy=_cell(first, "filtro de fecha"),
                date_column=_cell(first, "columna de fecha"),
            )
        else:
            # Filas de tabla que mapean a vistas → un chunk 'intent' por fila.
            # (Así el lenguaje natural de la taxonomía es recuperable y no queda
            #  sepultado en un chunk-section gigante con 14 intenciones mezcladas.)
            for view, cells in _intent_rows(tables, known_views):
                lines = [crumb, f"Mapeo intención de usuario → vista `semantic.{view}`:"]
                lines += [f"- {h}: {v}" for h, v in cells.items() if v]
                content = "\n".join(lines)
                add(
                    content,
                    "intent",
                    f"intent:{_slug(view)}:{hashlib.sha1(content.encode()).hexdigest()[:8]}",
                    section=node.title,
                    view_name=f"semantic.{view}",
                    intent=_clean_ident(next(iter(cells.values()), "")),
                    date_policy=_cell(cells, "filtro de fecha"),
                    date_column=_cell(cells, "columna de fecha"),
                )
            # Tablas sin mapeo a vistas se conservan como texto del body.
            body_text = body
            if tables and not _intent_rows(tables, known_views):
                for _h, _r, raw_block in tables:
                    body_text = f"{body_text}\n\n" + "\n".join(raw_block) if body_text \
                        else "\n".join(raw_block)

            # Sección con body propio, u hoja sin hijos (evitar docs vacíos de contenedores)
            if body_text or not node.children:
                header_text = f"{crumb}\n{'#' * node.level} {node.title}"
                full = f"{header_text}\n\n{body_text}" if body_text else header_text
                for part in _split_long(full, MAX_SECTION_CHARS):
                    section_seq += 1
                    add(part, "section",
                        f"section:{_slug(node.title)}:{section_seq}",
                        section=node.title)

        for child in node.children:
            visit(child, path + [node.title])

    for root_node in roots:
        visit(root_node, [])

    # Chunks puente columna → vistas (lookup por nombre de métrica; NO votan candidatas)
    for col, info in sorted(column_index.items()):
        views = sorted(set(info["views"]))
        add(
            f"Columna/métrica `{col}`: {info['definition']}\n"
            f"Disponible en las vistas: {', '.join(views)}.",
            "metric",
            f"metric:{_slug(col)}",
            section="__metrics__",
            column=col,
            views=", ".join(views),
        )

    logger.info(
        f"[RAG] Chunks construidos: "
        f"{sum(1 for d in docs if d.metadata['doc_type']=='view')} vistas, "
        f"{sum(1 for d in docs if d.metadata['doc_type']=='intent')} intents, "
        f"{sum(1 for d in docs if d.metadata['doc_type']=='metric')} métricas, "
        f"{sum(1 for d in docs if d.metadata['doc_type']=='section')} secciones."
    )
    return docs


# ------------------------------------------------------------------
# 2) Store versionado con manifiesto
# ------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


class BusinessRAG:
    def __init__(self, persist_directory: Path, collection_name: str,
                 embedding_model: str, agents_md_path: Path):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.agents_md_path = Path(agents_md_path)
        self._embeddings = OpenAIEmbeddings(model=embedding_model)
        self._store = self._open_store()

    @classmethod
    def from_env(cls) -> "BusinessRAG":
        _load_env()   # ANTES de leer cualquier variable de entorno
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY no definida.\n"
                f"  - Se buscó .env en: {BASE_DIR / '.env'}\n"
                "  - En producción: configura la variable en el entorno (Railway)."
            )
        # Resolver la config AQUÍ y no desde las constantes de módulo:
        # CHROMA_DIR / COLLECTION_NAME / EMBEDDING_MODEL / DEFAULT_AGENTS_MD_PATH
        # se evalúan en import time y NO verían valores que vengan solo del .env.
        return cls(
            persist_directory=Path(os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db"))),
            collection_name=os.getenv("RAG_COLLECTION", "bi_knowledge"),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
            agents_md_path=_resolve_agents_md(),
        )

    def _open_store(self) -> Chroma:
        # collection_metadata solo aplica al CREAR la colección; el rebuild la recrea.
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(self.persist_directory),
            collection_metadata={"hnsw:space": "cosine"},
        )

    # ---------------- manifiesto ----------------
    def _manifest_path(self) -> Path:
        return self.persist_directory / f"{self.collection_name}.manifest.json"

    def _read_manifest(self) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(self._manifest_path().read_text(encoding="utf-8"))
        except Exception:
            return None

    def _desired_manifest(self) -> Dict[str, Any]:
        return {
            "source_sha256": _sha256_file(self.agents_md_path),
            "embedding_model": self.embedding_model,
            "chunking_version": CHUNKING_VERSION,
        }

    def _count(self) -> int:
        try:
            return self._store._collection.count()
        except Exception:
            return 0

    # ---------------- ciclo de vida ----------------
    def ensure_indexed(self) -> bool:
        """Idempotente: reindexa SOLO si el archivo, el modelo o el chunker cambiaron."""
        if not self.agents_md_path.exists():
            logger.error(f"[RAG] No se encontró {self.agents_md_path}")
            return False
        if self._read_manifest() == self._desired_manifest() and self._count() > 0:
            logger.info(f"[RAG] Índice vigente ({self._count()} chunks).")
            return True
        logger.info("[RAG] AGENTS.md cambió o índice inválido → reindexando.")
        return self.rebuild() > 0

    def rebuild(self) -> int:
        docs = build_chunks(self.agents_md_path.read_text(encoding="utf-8-sig"))
        if not docs:
            logger.warning("[RAG] build_chunks devolvió 0 documentos.")
            return 0
        try:
            self._store.delete_collection()
        except Exception as e:
            logger.debug(f"[RAG] delete_collection: {e}")
        self._store = self._open_store()
        self._store.add_documents(documents=docs, ids=[d.id for d in docs])

        manifest = {
            **self._desired_manifest(),
            "doc_count": len(docs),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._manifest_path().write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"[RAG] Indexación completada: {len(docs)} chunks.")
        return len(docs)

    # ---------------- búsqueda ----------------
    def search(
        self,
        query: str,
        k: int = 5,
        doc_types: Optional[Union[str, Sequence[str]]] = None,
        view_names: Optional[Sequence[str]] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
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

    def search_view_candidates(
        self,
        query: str,
        k: int = 5,
        allowed_views: Optional[Sequence[str]] = None,
        min_score: float = 0.0,
        intent_weight: float = 1.0,
        oversample: int = 4,
    ) -> List[Dict[str, Any]]:
        """Candidatas de vista con FUSIÓN de evidencias.

        Votan por cada vista sus chunks 'view' (definición técnica enriquecida)
        e 'intent' (lenguaje natural de la taxonomía). Score de una vista =
        mejor score ponderado entre sus evidencias (max, no suma → sin sesgo
        por volumen de chunks). Los chunks 'metric' NO votan: sirven para
        lookup por nombre técnico, pero rankear vistas con ellos arrastraba
        métricas genéricas (ventas_hoy/ayer) hacia vistas incorrectas.
        """
        hits = self.search(query, k=max(k * oversample, 12),
                           doc_types=("view", "intent"), view_names=allowed_views)
        agg: Dict[str, Dict[str, Any]] = {}
        for h in hits:
            meta = h["document"].metadata
            v = meta.get("view_name")
            if not v:
                continue
            s = h["score"] * (intent_weight if meta.get("doc_type") == "intent" else 1.0)
            e = agg.setdefault(v, {
                "view_name": v, "score": 0.0, "raw_distance": None,
                "view_doc": None, "fallback_doc": None, "evidence": [],
            })
            if s > e["score"]:
                e["score"] = s
                e["raw_distance"] = h["raw_distance"]
                e["fallback_doc"] = h["document"]
            if meta.get("doc_type") == "view" and e["view_doc"] is None:
                e["view_doc"] = h["document"]
            e["evidence"].append(meta.get("doc_type", "?"))

        out: List[Dict[str, Any]] = []
        for e in sorted(agg.values(), key=lambda x: -x["score"]):
            if e["score"] < min_score:
                continue
            # El contexto siempre es el chunk 'view' (definición completa);
            # el chunk 'intent' solo queda de respaldo si la vista no rankeó directa.
            e["document"] = e["view_doc"] or e["fallback_doc"]
            out.append(e)
        return out[:k]


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
                rag = BusinessRAG.from_env()
                rag.ensure_indexed()
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
    """Ahora con FUSIÓN de evidencias: votan chunks 'view' + 'intent' por vista."""
    cands = get_rag().search_view_candidates(
        query, k=k, allowed_views=allowed_views, min_score=min_score_threshold)
    candidates = [{
        "view_name": c["view_name"],
        "context": c["document"].page_content if c["document"] else "",
        "score": c["score"],
        "original_score": c["score"],
        "raw_distance": c["raw_distance"],
        "metadata_boost": 0.0,
        "temporal_context": {},
        "view_metadata": c["document"].metadata if c["document"] else {},
        "evidence": c["evidence"],
    } for c in cands]

    if candidates:
        logger.info(f"[RAG] Top {min(3, len(candidates))} para '{query[:50]}...': "
                    + " | ".join(f"{c['view_name']} ({c['score']:.3f})" for c in candidates[:3]))
    else:
        logger.warning(f"[RAG] 0 candidatas sobre umbral {min_score_threshold} para '{query[:50]}...'")
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
    return [{
        **c,
        "purpose": c["view_metadata"].get("keywords", "")[:200],
        "grain": "desconocido",
        "metrics": [m for m in c["view_metadata"].get("metrics", "").split(", ") if m],
        "dimensions": [],
        "domain": "general",
        "notes": "",
        "keywords": c["view_metadata"].get("keywords", ""),
        "temporal_type": "general",
        "time_scope": "unknown",
        "usage_examples": [],
        "compatibility": {k2: "ok" for k2 in ("metric", "date_range", "location", "product", "category")},
        "compatibility_reason": f"La vista '{c['view_name']}' puede responder la consulta.",
        "can_answer": True,
    } for c in candidates]


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
# 5) API nueva: conocimiento de negocio (reglas, definiciones, métricas)
# ------------------------------------------------------------------
def buscar_conocimiento_negocio(
    query: str,
    k: int = 4,
    min_score: float = 0.25,
) -> List[Dict[str, Any]]:
    """Recupera definiciones y reglas (chunks 'metric' + 'section') para planner/supervisor."""
    hits = get_rag().search(query, k=k, doc_types=("metric", "section"), min_score=min_score)
    return [{
        "content": h["document"].page_content,
        "score": h["score"],
        "doc_type": h["document"].metadata.get("doc_type"),
        "section": h["document"].metadata.get("section", ""),
    } for h in hits]


# ------------------------------------------------------------------
# 6) CLI: reindex / stats / search / views
# ------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Gestor del índice RAG (AGENTS.md)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("reindex", help="Reconstruye el índice completo")
    sub.add_parser("stats", help="Muestra el manifiesto actual")

    p_search = sub.add_parser("search", help="Prueba una consulta (todos los doc_types)")
    p_search.add_argument("query")
    p_search.add_argument("--k", type=int, default=5)
    p_search.add_argument("--min-score", type=float, default=0.0)
    p_search.add_argument("--doc-types", default=None,
                          help="csv para filtrar, ej: view,intent")

    p_views = sub.add_parser("views", help="Candidatas de vista con fusión (como las ve el agente)")
    p_views.add_argument("query")
    p_views.add_argument("--k", type=int, default=5)
    p_views.add_argument("--min-score", type=float, default=0.0)

    args = parser.parse_args(argv)

    rag = BusinessRAG.from_env()
    if args.command == "reindex":
        print(f"Indexados {rag.rebuild()} chunks desde {rag.agents_md_path}")
    elif args.command == "stats":
        print(json.dumps(rag._read_manifest() or {}, indent=2, ensure_ascii=False))
        print("chunks en colección:", rag._count())
    elif args.command == "search":
        rag.ensure_indexed()
        dt = [s.strip() for s in args.doc_types.split(",")] if args.doc_types else None
        for h in rag.search(args.query, k=args.k, doc_types=dt, min_score=args.min_score):
            print(f"[{h['score']:.3f}] ({h['document'].metadata.get('doc_type')}) "
                  f"{h['document'].metadata.get('view_name') or h['document'].metadata.get('section')}")
            print("   " + h["document"].page_content[:160].replace("\n", " ") + "...")
    elif args.command == "views":
        rag.ensure_indexed()
        for c in rag.search_view_candidates(args.query, k=args.k, min_score=args.min_score):
            doc = c["document"]
            print(f"[{c['score']:.3f}] {c['view_name']}  evidencia: {', '.join(c['evidence'])}")
            if doc is not None:
                print("   " + doc.page_content[:160].replace("\n", " ") + "...")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
