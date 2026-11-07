"""Unit tests for LangChain RAG abstractions and First-Principles parity framework."""

import pytest
import numpy as np

from common.langchain_rag import (
    DeterministicLangChainEmbeddings,
    DeterministicLangChainChatModel,
    CitedAnswer,
    FirstPrinciplesRAGPipeline,
    LangChainLCELPipeline,
    benchmark_rag_architectures,
)
from langchain_core.messages import HumanMessage


# ---------------------------------------------------------------------------
# Test Fixtures & Minimal Corpus
# ---------------------------------------------------------------------------

SAMPLE_CORPUS = [
    "Vector embeddings map textual tokens into high-dimensional geometric coordinate spaces where semantic proximity correlates with cosine similarity.",
    "PyMuPDF provides high-performance document extraction, parsing font metrics and spatial bounding boxes with C-level efficiency.",
    "Reciprocal Rank Fusion merges ranked lists from dense vector search and sparse BM25 indices without requiring calibrated relevance score normalization.",
]


def test_deterministic_embeddings_adapter():
    """Verifies that DeterministicLangChainEmbeddings conforms to Embeddings protocol."""
    embeddings = DeterministicLangChainEmbeddings(dim=64)

    # Test single query embedding
    q_vec = embeddings.embed_query("Vector search mechanics")
    assert isinstance(q_vec, list)
    assert len(q_vec) == 64
    assert np.isclose(np.linalg.norm(q_vec), 1.0, atol=1e-3)

    # Test batch document embedding
    docs = ["Doc one text", "Doc two text"]
    doc_vecs = embeddings.embed_documents(docs)
    assert len(doc_vecs) == 2
    assert len(doc_vecs[0]) == 64
    assert len(doc_vecs[1]) == 64

    # Determinism assertion
    q_vec_again = embeddings.embed_query("Vector search mechanics")
    assert q_vec == q_vec_again


def test_deterministic_chat_model_grounded_answer():
    """Verifies DeterministicLangChainChatModel produces grounded answers with citations."""
    chat = DeterministicLangChainChatModel()
    prompt = (
        "=== CONTEXT PASSAGES ===\n"
        "[Doc 1] PyMuPDF extracts text bounding boxes with C-level speed.\n"
        "[Doc 2] BM25 relies on inverted index term frequency statistics.\n"
        "========================\n\n"
        "User Question: How does PyMuPDF extract text?\n\n"
        "Grounded Answer (with inline [Doc X] citations):"
    )
    result = chat.invoke([HumanMessage(content=prompt)])
    assert isinstance(result.content, str)
    assert "[Doc 1]" in result.content
    assert "PyMuPDF" in result.content


def test_deterministic_chat_model_refusal():
    """Verifies DeterministicLangChainChatModel refuses when context does not support query."""
    chat = DeterministicLangChainChatModel()
    prompt = (
        "=== CONTEXT PASSAGES ===\n"
        "[Doc 1] PyMuPDF extracts text bounding boxes with C-level speed.\n"
        "========================\n\n"
        "User Question: What is the GDP of Antarctica in 2026?\n\n"
        "Grounded Answer (with inline [Doc X] citations):"
    )
    result = chat.invoke([HumanMessage(content=prompt)])
    assert "cannot answer" in result.content.lower()


def test_first_principles_pipeline_lifecycle():
    """Verifies the pure Python baseline pipeline executes retrieval and generation."""
    pipeline = FirstPrinciplesRAGPipeline(dim=64)
    docs = [{"text": t, "metadata": {"doc_id": i}} for i, t in enumerate(SAMPLE_CORPUS)]
    pipeline.add_documents(docs)

    res = pipeline.run("How does PyMuPDF extract documents?", top_k=2)
    assert "query" in res
    assert "answer" in res
    assert "telemetry" in res
    assert len(res["retrieved_contexts"]) == 2
    assert any(c in res["answer"] for c in ["[Doc 1]", "[Doc 2]"])
    assert res["telemetry"]["total_ms"] > 0.0


def test_langchain_lcel_pipeline_lifecycle():
    """Verifies the LangChain LCEL pipeline indexing, invocation, and structured parsing."""
    pipeline = LangChainLCELPipeline(collection_name="test_unit_lcel", dim=64)
    split_docs = pipeline.index_texts(SAMPLE_CORPUS, chunk_size=200, chunk_overlap=20)
    assert len(split_docs) >= 3

    # Build and verify LCEL chain
    pipeline.build_chain(top_k=2)
    answer = pipeline.invoke("What does Reciprocal Rank Fusion do?")
    assert isinstance(answer, str)
    assert len(answer) > 10

    # Structured output verification
    structured: CitedAnswer = pipeline.invoke_structured("What does Reciprocal Rank Fusion do?")
    assert isinstance(structured, CitedAnswer)
    assert structured.question == "What does Reciprocal Rank Fusion do?"
    assert len(structured.answer) > 0
    assert not structured.is_refusal
    assert structured.confidence > 0.8

    # Batch invocation
    batch_results = pipeline.batch([
        "How does PyMuPDF work?",
        "What are vector embeddings?",
    ])
    assert len(batch_results) == 2
    assert all(isinstance(r, str) for r in batch_results)

    # Graph node extraction
    nodes = pipeline.get_graph_nodes()
    assert len(nodes) > 0
    assert any("Prompt" in n or "Chat" in n or "Parser" in n for n in nodes)


def test_comparative_benchmark_harness():
    """Verifies that side-by-side benchmark produces comparative telemetry and parity metrics."""
    test_queries = [
        "What is PyMuPDF used for?",
        "Explain Reciprocal Rank Fusion.",
        "What is quantum teleportation?",  # Refusal test
    ]
    summaries, aggregate = benchmark_rag_architectures(SAMPLE_CORPUS, test_queries, top_k=2)

    assert len(summaries) == 3
    assert aggregate["num_queries"] == 3
    assert aggregate["mean_first_principles_ms"] >= 0.0
    assert aggregate["mean_langchain_ms"] >= 0.0
    assert 0.0 <= aggregate["retrieval_parity_rate"] <= 1.0

    # The refusal test query should produce refusal in both pipelines
    refusal_summary = summaries[2]
    assert "cannot answer" in refusal_summary.first_principles_answer.lower()
    assert "cannot answer" in refusal_summary.langchain_answer.lower()
    assert refusal_summary.retrieval_parity is True
