# core/chunking.py
# -------------------------------------------------
# Chunking estructural del business memory (Markdown → Documents).
#
# Toda la clasificación se DERIVA de la sintaxis del archivo:
#   · heading snake_case        → doc_type="view"
#   · resto de headings         → doc_type="section"
#   · columnas `- \`col\`: def`   → doc_type="metric"  (puente columna→vistas)
#   · filas de tablas Markdown  → doc_type="table_row" (intención↔vista)
#   · bullets '- **Campo**: v'  → metadata fields_json (datos declarados)
#   · co-referencias `\`vista\``    → metadata related_views + auto-enriquecimiento
#
# Cero reglas de negocio hardcodeadas.
# Usado por indexer.py (escritura). core/rag.py NO lo importa en runtime.
# -------------------------------------------------
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Bumpear cuando cambie la lógica de chunking → el manifiesto fuerza reindex
CHUNKING_VERSION = "v2"

# Chunks de sección largos se subdividen preservando el breadcrumb
MAX_SECTION_CHARS = 1800

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]{1,}$")
_COLUMN_ITEM_RE = re.compile(r"^\s*-\s*`([^`\s]+)`\s*:\s*(.+)$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:\-|]+\|?\s*$")
_BOLD_FIELD_RE = re.compile(r"^\s*-\s*\*\*([^*]+)\*\*\s*[:：]\s*(.+?)\s*$")
_BACKTICK_REF_RE = re.compile(r"`(?:semantic\.)?([a-z][a-z0-9_]{1,})`")


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


def _iter_tables(lines: List[str]):
    """Detecta tablas Markdown por pura sintaxis. Yield (headers, rows)."""
    i = 0
    while i < len(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if "|" in lines[i] and _TABLE_SEP_RE.match(nxt):
            headers = [c.strip().strip("`") for c in lines[i].strip().strip("|").split("|")]
            rows: List[List[str]] = []
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                cells = [c.strip().replace("**", "")
                         for c in lines[i].strip().strip("|").split("|")]
                if any(cells):
                    rows.append(cells)
                i += 1
            yield headers, rows
        else:
            i += 1


def _row_as_text(crumb: str, headers: List[str], cells: List[str]) -> str:
    pairs = " · ".join(f"{h}: {c}" for h, c in zip(headers, cells) if h and c)
    return f"{crumb}\nFila de tabla — {pairs}"


def _bold_fields(lines: List[str]) -> Dict[str, str]:
    """Bullets '- **Campo**: valor' → dict. Genérico: captura TODOS los campos."""
    return {m.group(1): m.group(2)
            for line in lines if (m := _BOLD_FIELD_RE.match(line))}


def build_chunks(md_text: str) -> List[Document]:
    """
    Convierte AGENTS.md en documentos listos para indexar.
    Clasificación 100% derivada de la estructura del archivo.
    """
    raw = md_text.replace("\r\n", "\n").replace("\r", "\n")
    preamble, roots = _parse_heading_tree(raw)

    docs: List[Document] = []
    seen_ids: set = set()
    section_seq = 0
    table_seq = 0
    column_index: Dict[str, Dict[str, Any]] = {}
    view_docs: Dict[str, Document] = {}
    table_chunks: List[Document] = []

    def add(content: str, doc_type: str, doc_id: str, **metadata) -> Document:
        if doc_id in seen_ids:
            doc_id = f"{doc_id}:{hashlib.sha1(content.encode()).hexdigest()[:10]}"
        seen_ids.add(doc_id)
        doc = Document(page_content=content.strip(),
                       metadata={"doc_type": doc_type, **metadata}, id=doc_id)
        docs.append(doc)
        return doc

    if any(l.strip() for l in preamble):
        add("\n".join(preamble).strip(), "section", "section:preamble", section="__preamble__")

    def visit(node: _Node, path: List[str]) -> None:
        nonlocal section_seq, table_seq
        crumb = " > ".join(path + [node.title])
        body = "\n".join(node.lines).strip()

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

            fields = _bold_fields(node.lines)  # granularidad, tipo, filtro, notas...
            view_docs[node.title] = add(
                f"{crumb}\n\n### {node.title}\n{body}",
                "view",
                f"view:{_slug(node.title)}",
                section=path[-1] if path else "",
                view_name=f"semantic.{node.title}",   # compat con metadata anterior
                metrics=", ".join(cols[:20]),         # compat con metadata anterior
                keywords=body[:500],                  # compat con metadata anterior
                fields_json=json.dumps(fields, ensure_ascii=False),
            )
        else:
            # Filas de tabla → chunks propios (recuperación fina intención↔vista)
            for headers, rows in _iter_tables(node.lines):
                for cells in rows:
                    table_seq += 1
                    table_chunks.append(add(
                        _row_as_text(crumb, headers, cells),
                        "table_row",
                        f"table:{_slug(node.title)}:{table_seq}",
                        section=node.title,
                    ))
            # Sección con body propio, u hoja sin hijos (evita docs vacíos)
            if body or not node.children:
                header_text = f"{crumb}\n{'#' * node.level} {node.title}"
                full = f"{header_text}\n\n{body}" if body else header_text
                for part in _split_long(full, MAX_SECTION_CHARS):
                    section_seq += 1
                    add(part, "section",
                        f"section:{_slug(node.title)}:{section_seq}",
                        section=node.title)

        for child in node.children:
            visit(child, path + [node.title])

    for root_node in roots:
        visit(root_node, [])

    # ---- Enlace estructural por co-referencia (auto-descubierto) ----
    view_names = set(view_docs)
    additions: Dict[str, List[str]] = defaultdict(list)

    for doc in docs:
        if doc.metadata["doc_type"] in ("table_row", "section"):
            refs = sorted(set(_BACKTICK_REF_RE.findall(doc.page_content)) & view_names)
            if refs:
                doc.metadata["related_views"] = ", ".join(refs)
                if doc.metadata["doc_type"] == "table_row":
                    for v in refs:
                        additions[v].append(doc.page_content.splitlines()[-1])

    # El propio documento declara qué preguntas responde cada vista → se
    # materializan en el chunk de la vista (cierra el gap query↔chunk).
    for name, ref_lines in additions.items():
        view_docs[name].page_content += (
            "\n\n**Intenciones y preguntas de negocio asociadas** "
            "(mapeadas por el propio documento):\n"
            + "\n".join(f"- {line}" for line in ref_lines)
        )

    # Chunks puente columna → vistas (recuperación por métrica)
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

    logger.info(f"[Chunking] Chunks construidos: {chunk_stats(docs)}")
    return docs


def chunk_stats(docs: List[Document]) -> Dict[str, int]:
    """Conteo por doc_type — para logs del indexer y verificación."""
    stats: Dict[str, int] = defaultdict(int)
    for d in docs:
        stats[d.metadata.get("doc_type", "unknown")] += 1
    return dict(sorted(stats.items()))
