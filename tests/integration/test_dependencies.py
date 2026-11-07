"""Smoke test core dependencies without network calls or API token usage."""

import pytest


def test_chromadb_ephemeral_client():
    import chromadb

    client = chromadb.EphemeralClient()
    collection = client.create_collection("smoke_test")
    collection.add(
        documents=["RAG enables grounding LLMs with retrieved context."],
        ids=["doc_001"],
    )
    results = collection.query(query_texts=["grounding"], n_results=1)
    assert len(results["ids"][0]) == 1
    assert results["ids"][0][0] == "doc_001"


def test_pymupdf_available():
    import pymupdf

    # Verify pymupdf can create an empty in-memory document
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "RAG Engineering Academy Smoke Test")
    text = page.get_text()
    assert "Smoke Test" in text
    doc.close()


def test_pypdf_available():
    import pypdf

    assert hasattr(pypdf, "PdfReader")


def test_groq_sdk_available():
    import groq

    assert hasattr(groq, "Groq")
