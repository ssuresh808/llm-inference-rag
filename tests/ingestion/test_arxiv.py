"""Tests for on-domain arXiv ingestion (fully mocked, no network).

The HF stream is replaced by fake sample rows matching the
``librarian-bots/arxiv-metadata-snapshot`` schema
(id, title, abstract, categories, update_date, authors).
"""

from datetime import date

import datasets

from src.ingestion.arxiv import arxiv_rows_to_documents, load_arxiv_domain_abstracts

FAKE_ROWS = [
    {  # match: cs.CL + 2023 + seed vocab
        "id": "2301.00001",
        "title": "vLLM serving",
        "abstract": "PagedAttention manages the KV cache for high throughput.",
        "categories": "cs.CL cs.LG",
        "update_date": "2023-06-01",
        "authors": "Ada One",
    },
    {  # reject: date too old
        "id": "2210.99999",
        "title": "FlashAttention",
        "abstract": "Fast exact attention with continuous batching.",
        "categories": "cs.AI",
        "update_date": "2022-10-01",
        "authors": "Bob Two",
    },
    {  # reject: wrong category and no seed vocab
        "id": "2305.00002",
        "title": "Cooking with data",
        "abstract": "A guide to recipes.",
        "categories": "cs.CY",
        "update_date": "2023-05-01",
        "authors": "Cy Three",
    },
    {  # match: cs.AI + 2024 + 'quantization'
        "id": "2402.00003",
        "title": "Quantization of LLMs",
        "abstract": "Int8 quantization reduces inference cost.",
        "categories": "cs.AI",
        "update_date": "2024-02-01",
        "authors": "Dee Four",
    },
    {  # reject: category/date ok but no seed vocabulary
        "id": "2306.00004",
        "title": "Graph theory results",
        "abstract": "Unrelated combinatorics.",
        "categories": "cs.CL",
        "update_date": "2023-06-01",
        "authors": "Eve Five",
    },
]


def test_selects_only_on_domain_rows():
    docs = arxiv_rows_to_documents(FAKE_ROWS)
    titles = [d.metadata["title"] for d in docs]
    assert titles == ["vLLM serving", "Quantization of LLMs"]


def test_metadata_and_combined_text():
    docs = arxiv_rows_to_documents(FAKE_ROWS)
    doc = docs[0]
    assert doc.metadata["source"] == "arxiv:2301.00001"
    assert doc.metadata["authors"] == "Ada One"
    assert "vLLM serving" in doc.page_content  # title combined in
    assert "PagedAttention" in doc.page_content  # abstract combined in


def test_date_filter_is_type_safe_for_datetime():
    rows = [
        {
            "id": "x",
            "title": "vLLM",
            "abstract": "continuous batching",
            "categories": "cs.CL",
            "update_date": date(2023, 7, 1),  # datetime, not string
            "authors": "",
        }
    ]
    assert len(arxiv_rows_to_documents(rows)) == 1


def test_date_filter_rejects_old_datetime():
    rows = [
        {
            "id": "x",
            "title": "vLLM",
            "abstract": "continuous batching",
            "categories": "cs.CL",
            "update_date": date(2021, 1, 1),
            "authors": "",
        }
    ]
    assert arxiv_rows_to_documents(rows) == []


def test_max_docs_cap():
    assert len(arxiv_rows_to_documents(FAKE_ROWS, max_docs=1)) == 1


def test_max_scan_stops_early():
    # Only the 1st row is scanned, so the 2nd-position match is never reached.
    assert arxiv_rows_to_documents(FAKE_ROWS, max_scan=1)[0].metadata["title"] == "vLLM serving"
    # If we only scan the (rejected) old row at index 1 onward, nothing matches.
    assert arxiv_rows_to_documents(FAKE_ROWS[1:3], max_scan=2) == []


def test_load_arxiv_domain_abstracts_mocks_stream(monkeypatch):
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: list(FAKE_ROWS))
    docs = load_arxiv_domain_abstracts(max_docs=10)
    assert len(docs) == 2
