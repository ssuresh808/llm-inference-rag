"""Document ingestion: load files, chunk them, and enforce a quality gate.

The pipeline is three composable steps:

1. :func:`load_documents` - read supported files from a directory.
2. :func:`chunk_documents` - split documents with a deterministic splitter.
3. :func:`apply_quality_gate` - clean whitespace and drop low-value chunks.

:func:`ingest_directory` runs all three. Each step is pure and side-effect free
(beyond logging), so it can be unit-tested without any network access.
"""

import logging
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

logger = logging.getLogger(__name__)

#: File suffixes loaded as plain UTF-8 text.
TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt"})
#: File suffixes loaded via the PDF parser.
PDF_SUFFIXES = frozenset({".pdf"})

#: Chunks shorter than this many characters (after cleaning) are rejected.
MIN_CHUNK_CHARS = 50
#: Default target chunk size in characters.
DEFAULT_CHUNK_SIZE = 1000
#: Default overlap between adjacent chunks, in characters.
DEFAULT_CHUNK_OVERLAP = 150

_WHITESPACE_RE = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    """Collapse all runs of whitespace to single spaces and strip the ends.

    Args:
        text: Raw text, possibly containing excessive or irregular whitespace.

    Returns:
        The text with internal whitespace runs collapsed to one space and
        leading/trailing whitespace removed.
    """
    return _WHITESPACE_RE.sub(" ", text).strip()


def _load_text_file(path: Path) -> Document:
    """Read a UTF-8 text/markdown file into a single ``Document``.

    Args:
        path: Path to the text or markdown file.

    Returns:
        A document whose ``metadata['source']`` is the file path.
    """
    return Document(
        page_content=path.read_text(encoding="utf-8"),
        metadata={"source": str(path)},
    )


def _load_pdf_file(path: Path) -> Document:
    """Extract text from every page of a PDF into a single ``Document``.

    Args:
        path: Path to the PDF file.

    Returns:
        A document with the concatenated page text and a ``source`` metadata
        entry.
    """
    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return Document(page_content=text, metadata={"source": str(path)})


def load_documents(directory: Path | str) -> list[Document]:
    """Load every supported file under ``directory`` (recursively).

    Markdown/text files are read as UTF-8; PDFs are parsed page by page.
    Unsupported file types are skipped with a debug log.

    Args:
        directory: Path to the corpus directory.

    Returns:
        The loaded documents, with source metadata populated by the loaders.

    Raises:
        NotADirectoryError: If ``directory`` does not exist or is not a
            directory.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    documents: list[Document] = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        suffix = path.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            documents.append(_load_text_file(path))
        elif suffix in PDF_SUFFIXES:
            documents.append(_load_pdf_file(path))
        else:
            logger.debug("Skipping unsupported file: %s", path)

    logger.info("Loaded %d document(s) from %s", len(documents), directory)
    return documents


def chunk_documents(
    documents: list[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into overlapping chunks with a deterministic splitter.

    Args:
        documents: Documents to split.
        chunk_size: Target maximum chunk size in characters.
        chunk_overlap: Overlap between adjacent chunks in characters.

    Returns:
        The resulting chunks as new ``Document`` objects (metadata preserved).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def chunk_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Chunk a raw string. Convenience wrapper around :func:`chunk_documents`.

    Args:
        text: Raw text to chunk.
        chunk_size: Target maximum chunk size in characters.
        chunk_overlap: Overlap between adjacent chunks in characters.

    Returns:
        The resulting chunks.
    """
    return chunk_documents(
        [Document(page_content=text)],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def apply_quality_gate(
    chunks: list[Document],
    *,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[Document]:
    """Clean whitespace and drop chunks shorter than ``min_chars``.

    The input chunks are not mutated; accepted chunks are returned as new
    ``Document`` objects with cleaned content and original metadata preserved.
    Processed/accepted/rejected counts are logged at INFO level.

    Args:
        chunks: Candidate chunks to filter.
        min_chars: Minimum length (after cleaning) for a chunk to be accepted.

    Returns:
        The accepted chunks, in their original order.
    """
    accepted: list[Document] = []
    rejected = 0
    for chunk in chunks:
        cleaned = _clean_text(chunk.page_content)
        if len(cleaned) < min_chars:
            rejected += 1
            continue
        accepted.append(Document(page_content=cleaned, metadata=chunk.metadata))

    logger.info(
        "Quality gate: %d processed, %d accepted, %d rejected (min_chars=%d)",
        len(chunks),
        len(accepted),
        rejected,
        min_chars,
    )
    return accepted


def ingest_directory(
    directory: Path | str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[Document]:
    """Run the full pipeline: load -> chunk -> quality gate.

    Args:
        directory: Path to the corpus directory.
        chunk_size: Target maximum chunk size in characters.
        chunk_overlap: Overlap between adjacent chunks in characters.
        min_chars: Minimum length for a chunk to survive the quality gate.

    Returns:
        Clean, gate-approved chunks ready for embedding/indexing.
    """
    documents = load_documents(directory)
    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return apply_quality_gate(chunks, min_chars=min_chars)
