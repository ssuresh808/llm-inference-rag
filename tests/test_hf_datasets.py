"""Tests for HF dataset -> documents conversion (offline, no network).

Only the pure ``rows_to_documents`` transform is tested; the network-bound
``load_hf_corpus`` simply streams rows and delegates to it. The default filter
keeps a row only if it matches a model term AND an efficiency term.
"""

from src.ingestion.hf_datasets import rows_to_documents

ROWS = [
    # model term (transformer) + efficiency term (quantization) -> kept
    {
        "title": "Efficient Transformer Inference",
        "abstract": "Cutting transformer latency via quantization.",
    },
    # neither group -> rejected
    {"title": "Cooking Pasta", "abstract": "A guide to boiling water for dinner."},
    # model term only (language model), no efficiency term -> rejected
    {"title": "Scaling Laws", "abstract": "A study of large language model accuracy."},
    # model term (transformer) + efficiency term (kv cache) -> kept
    {
        "title": "KV Cache for LLMs",
        "abstract": "Managing the kv cache in a transformer decoder.",
    },
    # empty -> always skipped
    {"title": "", "abstract": ""},
]


def test_and_of_groups_filter_keeps_only_on_domain_rows():
    titles = [d.metadata.get("title") for d in rows_to_documents(ROWS, dataset_id="test/ds")]
    assert "Efficient Transformer Inference" in titles
    assert "KV Cache for LLMs" in titles
    assert "Cooking Pasta" not in titles  # matches neither group
    assert "Scaling Laws" not in titles  # model term but no efficiency term


def test_metadata_and_content():
    docs = rows_to_documents(ROWS, dataset_id="test/ds")
    doc = docs[0]
    assert doc.metadata["source"].startswith("test/ds#")
    assert doc.metadata["title"] == "Efficient Transformer Inference"
    assert "quantization" in doc.page_content.lower()
    assert "Efficient Transformer Inference" in doc.page_content  # title concatenated in


def test_limit_caps_results():
    assert len(rows_to_documents(ROWS, dataset_id="test/ds", limit=1)) == 1


def test_no_filter_keeps_all_non_empty():
    docs = rows_to_documents(ROWS, dataset_id="test/ds", keyword_groups=None)
    assert len(docs) == 4  # the empty row is still skipped
