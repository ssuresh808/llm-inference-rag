"""Tests for the ingestion pipeline (src/ingestion/loader.py).

All tests are offline: chunking and the quality gate operate on in-memory
strings, and loader tests use temporary files via the ``tmp_path`` fixture.
"""

import logging

import pytest
from langchain_core.documents import Document

from src.ingestion.loader import (
    apply_quality_gate,
    chunk_text,
    ingest_directory,
    load_documents,
)

# Comfortably above the 50-char minimum.
LONG_TEXT = "Continuous batching improves GPU utilization for LLM serving."


def test_quality_gate_rejects_short_chunks():
    chunks = [
        Document(page_content=LONG_TEXT),
        Document(page_content="too short"),
    ]

    kept = apply_quality_gate(chunks)

    assert len(kept) == 1
    assert kept[0].page_content == LONG_TEXT


def test_quality_gate_strips_excessive_whitespace():
    messy = "   spread   out   words   " * 4 + "\n\n\t  "

    [doc] = apply_quality_gate([Document(page_content=messy)])

    assert "  " not in doc.page_content  # no double spaces remain
    assert doc.page_content == doc.page_content.strip()  # ends trimmed


def test_quality_gate_preserves_metadata():
    chunk = Document(page_content=LONG_TEXT, metadata={"source": "vllm.md"})

    [kept] = apply_quality_gate([chunk])

    assert kept.metadata == {"source": "vllm.md"}


def test_quality_gate_does_not_mutate_input():
    original = Document(page_content=f"  {LONG_TEXT}  ")

    apply_quality_gate([original])

    assert original.page_content == f"  {LONG_TEXT}  "  # untouched


def test_quality_gate_logs_processed_and_rejected_counts(caplog):
    chunks = [Document(page_content=LONG_TEXT), Document(page_content="nope")]

    with caplog.at_level(logging.INFO):
        apply_quality_gate(chunks)

    assert "2 processed" in caplog.text
    assert "1 accepted" in caplog.text
    assert "1 rejected" in caplog.text


def test_chunking_splits_long_text_within_size():
    text = "The quick brown fox jumps over the lazy dog. " * 200

    chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 200 for c in chunks)


def test_load_documents_reads_markdown_and_skips_unknown(tmp_path):
    (tmp_path / "doc.md").write_text(
        "# vLLM\n\nPagedAttention manages the KV cache.", encoding="utf-8"
    )
    (tmp_path / "ignore.png").write_bytes(b"\x89PNG\r\n")

    docs = load_documents(tmp_path)

    assert len(docs) == 1
    assert "PagedAttention" in docs[0].page_content


def test_load_documents_raises_on_missing_directory(tmp_path):
    with pytest.raises(NotADirectoryError):
        load_documents(tmp_path / "does-not-exist")


def test_ingest_directory_end_to_end_filters_tiny_files(tmp_path):
    (tmp_path / "real.md").write_text(
        "Speculative decoding reduces latency. " * 30, encoding="utf-8"
    )
    (tmp_path / "tiny.md").write_text("hi", encoding="utf-8")  # rejected by the gate

    chunks = ingest_directory(tmp_path, chunk_size=200, chunk_overlap=20)

    assert len(chunks) >= 1
    assert all(len(c.page_content) >= 50 for c in chunks)
