"""Tests for HF dataset -> documents conversion (offline, no network).

Only the pure ``rows_to_documents`` transform is tested; the network-bound
``load_hf_corpus`` simply streams rows and delegates to it.
"""

from src.ingestion.hf_datasets import rows_to_documents

ROWS = [
    {"title": "Efficient Inference", "abstract": "We reduce GPU latency with quantization."},
    {"title": "Cooking Pasta", "abstract": "A guide to boiling water for dinner."},
    {"title": "KV Cache Tricks", "abstract": "Managing the kv cache for long context."},
    {"title": "", "abstract": ""},  # empty -> always skipped
]


def test_keyword_filter_keeps_on_domain_rows():
    titles = [d.metadata.get("title") for d in rows_to_documents(ROWS, dataset_id="test/ds")]
    assert "Efficient Inference" in titles
    assert "KV Cache Tricks" in titles
    assert "Cooking Pasta" not in titles  # no domain keyword -> filtered out


def test_metadata_and_content():
    docs = rows_to_documents(ROWS, dataset_id="test/ds")
    doc = docs[0]
    assert doc.metadata["source"].startswith("test/ds#")
    assert doc.metadata["title"] == "Efficient Inference"
    assert "quantization" in doc.page_content.lower()
    assert "Efficient Inference" in doc.page_content  # title concatenated in


def test_limit_caps_results():
    assert len(rows_to_documents(ROWS, dataset_id="test/ds", limit=1)) == 1


def test_no_keywords_keeps_all_non_empty():
    docs = rows_to_documents(ROWS, dataset_id="test/ds", keywords=None)
    assert len(docs) == 3  # the empty row is still skipped
