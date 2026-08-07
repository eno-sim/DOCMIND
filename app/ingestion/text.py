"""Extraction and chunking for text, Markdown, and PDF documents.
pypdf is the library used for PDF extraction.
Design decisions
----------------
* PDF pages are extracted independently so citations can include a page number.
* Chunking uses character windows with overlap.  It is deterministic, cheap to
  run locally, and gives us a useful baseline before evaluating semantic
  chunking.  The defaults target roughly 200 tokens (characters are only a
  practical approximation) with enough overlap to preserve context.
* Whitespace is normalized, but the original source file is never modified.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable


TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
PDF_SUFFIXES = {".pdf"}


@dataclass
class TextChunk:
    """A searchable section of an ingested source.

    ``metadata`` deliberately contains scalar values because ChromaDB and
    most sparse-index implementations require flat metadata.  Page numbers
    and timestamps are therefore represented as strings at this boundary.
    """

    text: str  # content of the chunk
    source: str  # path to the source document
    chunk_id: str  # unique identifier for this chunk, stable across re-ingestion
    metadata: dict[str, str] = field(default_factory=dict)


def _normalize_text(text: str) -> str:
    """Take messy extracted text and make it cleaner by:
    - standardizing line breaks
    - collapsing repeated spaces/tabs
    - reducing excessive blank lines
    - trimming leading and trailing whitespace
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _validate_path(path: str | Path) -> Path:
    """It checks whether the given path is a real file.

    If yes, it returns it as a Path object. If not, it raises an error.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Document does not exist: {file_path}")
    return file_path


def extract_text(path: str | Path) -> tuple[str, dict[str, str]]:
    """Extract a complete text document and its source metadata.

    Plain text and Markdown are decoded as UTF-8.  PDF extraction is lazy:
    importing :mod:`pypdf` only when needed keeps text-only usage lightweight.
    For PDFs, page markers are included in the returned text and the metadata
    records the page count; use :func:`extract_pages` when page-level chunks
    are required for citations.
    """
    file_path = _validate_path(path)
    suffix = file_path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        text = file_path.read_text(encoding="utf-8")
        return _normalize_text(text), {"source": str(file_path), "file_type": suffix[1:]}
    if suffix in PDF_SUFFIXES:
        pages = list(extract_pages(file_path))
        text = "\n\n".join(page_text for _, page_text in pages if page_text)
        return _normalize_text(text), {
            "source": str(file_path),
            "file_type": "pdf",
            "page_count": str(len(pages)),
        }
    raise ValueError(f"Unsupported document type: {suffix or '<none>'}")


def extract_pages(path: str | Path) -> Iterable[tuple[int, str]]:
    """Yield ``(one_based_page_number, text)`` pairs from a PDF."""
    file_path = _validate_path(path)
    if file_path.suffix.lower() != ".pdf":
        raise ValueError("extract_pages only accepts PDF files")

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install pypdf to ingest PDF files") from exc

    reader = PdfReader(file_path)
    for page_number, page in enumerate(reader.pages, start=1):
        yield page_number, _normalize_text(page.extract_text() or "")


def chunk_text(
    text: str,
    source: str,
    size: int = 800,
    overlap: int = 120,
    metadata: dict[str, str] | None = None,
) -> list[TextChunk]:
    """Split text into deterministic overlapping character-based chunks.

    The boundary prefers clean word boundaries both at the end and the beginning of chunks.
    """
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("size must be positive and overlap must be smaller than size")

    normalized = _normalize_text(text)
    base_metadata = dict(metadata or {})
    chunks: list[TextChunk] = []
    start = 0
    index = 0
    
    while start < len(normalized):
        hard_end = min(start + size, len(normalized))
        end = hard_end
        
        # 1. Ensure the end of the chunk prefers a clean word boundary
        if hard_end < len(normalized):
            boundary = normalized.rfind(" ", start, hard_end)
            if boundary > start:
                end = boundary
                
        content = normalized[start:end].strip()
        if content:
            chunks.append(TextChunk(
                text=content,
                source=source,
                chunk_id=f"{source}:chunk-{index}",
                metadata={**base_metadata, "source": source, "chunk_index": str(index)},
            ))
            index += 1
            
        if end >= len(normalized):
            break
            
        # 2. Calculate raw overlap start
        raw_start = max(end - overlap, start + 1)
        
        # 3. FIX: Ensure the beginning of the next chunk also starts on a clean word boundary!
        if raw_start > 0 and normalized[raw_start - 1] != " ":
            clean_start = normalized.find(" ", raw_start, end)
            if clean_start != -1:
                start = clean_start + 1
            else:
                start = raw_start
        else:
            start = raw_start
            
    return chunks


def ingest_text_document(path: str | Path, size: int = 800, overlap: int = 40) -> list[TextChunk]:
    """Extract and chunk a text/Markdown/PDF document with citation metadata."""
    file_path = _validate_path(path)
    if file_path.suffix.lower() == ".pdf":
        chunks: list[TextChunk] = []
        for page_number, page_text in extract_pages(file_path):
            chunks.extend(chunk_text(
                page_text,
                str(file_path),
                size=size,
                overlap=overlap,
                metadata={"file_type": "pdf", "page": str(page_number)},
            ))
        return chunks

    text, metadata = extract_text(file_path)
    return chunk_text(text, str(file_path), size=size, overlap=overlap, metadata=metadata)