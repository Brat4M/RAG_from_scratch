"""Unit tests for query transformation, expansion, rewriting, and HyDE strategies.

Validates:
1. DeterministicSemanticEmbedder vector normalization and similarity properties.
2. QueryRewriter offline abbreviation expansion and filler stripping.
3. StepBackPrompter offline concept abstraction.
4. SubQueryDecomposer atomic query splitting and multi-hop handling.
5. HyDEGenerator offline passage synthesis and vector similarity gap bridging.
6. Reciprocal Rank Fusion (RRF) mathematical scoring and multi-list ranking.
7. QueryTransformationEngine end-to-end execution across all strategies.
8. Graceful fallback when LLM API keys are missing.
"""

import numpy as np
import pytest

from common.query_transformations import (
    DeterministicSemanticEmbedder,
    QueryRewriter,
    StepBackPrompter,
    SubQueryDecomposer,
    HyDEGenerator,
    reciprocal_rank_fusion,
    QueryTransformationEngine,
)


@pytest.fixture
def sample_corpus():
    return {
        "doc_hnsw": (
            "HNSW (Hierarchical Navigable Small World) graphs construct multi-layer proximity networks "
            "for logarithmic approximate nearest neighbor search with high recall."
        ),
        "doc_ivf": (
            "Inverted File Index (IVF) partitions vector space into Voronoi cells using k-means centroids, "
            "reducing distance calculations during ANN querying."
        ),
        "doc_cuda_oom": (
            "PyTorch CUDA out-of-memory errors occur when GPU memory allocations exceed device capacity "
            "during tensor operations or large batch sizes."
        ),
        "doc_cuda_frag": (
            "CUDA memory fragmentation happens when dynamically sized tensors prevent contiguous page "
            "allocations on the GPU memory pool."
        ),
        "doc_chromadb": (
            "ChromaDB uses SQLite for relational metadata storage and DuckDB or Rust vector indices "
            "for fast persistent similarity search."
        ),
        "doc_hybrid": (
            "Hybrid search combines dense vector similarity with sparse BM25 lexical keyword matching "
            "using Reciprocal Rank Fusion."
        ),
    }


def test_deterministic_semantic_embedder():
    embedder = DeterministicSemanticEmbedder(dim=128)
    v1 = embedder.embed("PyTorch CUDA out of memory")
    v2 = embedder.embed("PyTorch CUDA out of memory")
    v3 = embedder.embed("HNSW graph approximate nearest neighbor")

    # 1. Normalization invariant: ||v|| == 1.0
    assert np.isclose(np.linalg.norm(v1), 1.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(v3), 1.0, atol=1e-5)

    # 2. Determinism: identical strings yield identical vectors
    assert np.isclose(float(np.dot(v1, v2)), 1.0, atol=1e-5)

    # 3. Discriminative power: different topics have lower similarity
    sim_unrelated = float(np.dot(v1, v3))
    assert sim_unrelated < 0.6, f"Expected lower similarity for unrelated topics, got {sim_unrelated}"

    # 4. Batch embedding shape
    batch = embedder.embed_batch(["text 1", "text 2", "text 3"])
    assert batch.shape == (3, 128)


def test_query_rewriter_offline():
    rewriter = QueryRewriter(groq_api_key=None)

    raw_query = "Can you tell me why I get oom on my gpu with big batch size?"
    rewritten = rewriter.rewrite_offline(raw_query)

    # Should strip conversational filler and expand 'oom'
    assert not rewritten.startswith("can you tell me")
    assert "out of memory cuda allocation failure" in rewritten

    # Test ANN expansion
    ann_query = "how does ann vector db work?"
    ann_rewritten = rewriter.rewrite_offline(ann_query)
    assert "approximate nearest neighbor" in ann_rewritten
    assert "vector database" in ann_rewritten


def test_step_back_prompter_offline():
    prompter = StepBackPrompter(groq_api_key=None)

    # Granular PyTorch CUDA question
    query_cuda = "Why did my conv2d layer trigger a CUDA assertion error at batch size 64?"
    step_back_cuda = prompter.step_back_offline(query_cuda)
    assert "GPU memory" in step_back_cuda or "PyTorch CUDA" in step_back_cuda

    # Granular HNSW indexing question
    query_hnsw = "Why is M=16 efConstruction=200 causing slow build times in my hnsw graph?"
    step_back_hnsw = prompter.step_back_offline(query_hnsw)
    assert "Approximate Nearest Neighbor" in step_back_hnsw


def test_sub_query_decomposer_offline():
    decomposer = SubQueryDecomposer(groq_api_key=None)

    # 1. Comparative query with "vs"
    query_vs = "HNSW vs IVF indexing in ChromaDB"
    sub_queries = decomposer.decompose_offline(query_vs)
    assert len(sub_queries) == 2
    assert "hnsw" in sub_queries[0].lower()
    assert "ivf" in sub_queries[1].lower()

    # 2. Comparative query with "compare X and Y"
    query_comp = "Compare dense vector similarity and sparse BM25 keyword search"
    sub_queries_comp = decomposer.decompose_offline(query_comp)
    assert len(sub_queries_comp) == 2
    assert "dense vector similarity" in sub_queries_comp[0].lower()
    assert "sparse bm25 keyword search" in sub_queries_comp[1].lower()

    # 3. Simple atomic query fallback
    query_simple = "What is Reciprocal Rank Fusion?"
    sub_simple = decomposer.decompose_offline(query_simple)
    assert len(sub_simple) == 1
    assert sub_simple[0] == query_simple


def test_reciprocal_rank_fusion_math():
    list_a = [("doc_1", 0.95), ("doc_2", 0.85), ("doc_3", 0.70)]
    list_b = [("doc_2", 0.90), ("doc_4", 0.80), ("doc_1", 0.65)]

    # Standard k = 60
    # doc_1 rank in A is 1 (score 1/61), rank in B is 3 (score 1/63) -> sum = 1/61 + 1/63
    # doc_2 rank in A is 2 (score 1/62), rank in B is 1 (score 1/61) -> sum = 1/62 + 1/61
    expected_doc1 = (1.0 / 61) + (1.0 / 63)
    expected_doc2 = (1.0 / 62) + (1.0 / 61)

    fused = reciprocal_rank_fusion([list_a, list_b], k=60, top_n=4)
    fused_dict = dict(fused)

    assert np.isclose(fused_dict["doc_1"], expected_doc1, atol=1e-6)
    assert np.isclose(fused_dict["doc_2"], expected_doc2, atol=1e-6)

    # doc_2 has higher fused score than doc_1 because 1/62 > 1/63
    assert fused[0][0] == "doc_2"
    assert fused[1][0] == "doc_1"


def test_hyde_hypothetical_document_and_similarity_boost(sample_corpus):
    embedder = DeterministicSemanticEmbedder(dim=128)
    hyde = HyDEGenerator(groq_api_key=None)

    colloquial_query = "Why is my GPU crashing when running big batches?"
    hypo_doc = hyde.generate_offline(colloquial_query)

    # Check generated passage content
    assert len(hypo_doc) > 50
    assert "CUDA" in hypo_doc or "GPU" in hypo_doc

    target_doc = sample_corpus["doc_cuda_oom"]
    unrelated_doc = sample_corpus["doc_hnsw"]

    vec_query = embedder.embed(colloquial_query)
    vec_hypo = embedder.embed(hypo_doc)
    vec_target = embedder.embed(target_doc)
    vec_unrelated = embedder.embed(unrelated_doc)

    sim_naive = float(np.dot(vec_query, vec_target))
    sim_hyde = float(np.dot(vec_hypo, vec_target))
    sim_unrelated = float(np.dot(vec_hypo, vec_unrelated))

    # HyDE passage must bridge the gap, producing significantly higher similarity to target
    assert sim_hyde > sim_naive, f"HyDE similarity {sim_hyde} should exceed naive {sim_naive}"
    assert sim_hyde > sim_unrelated, f"HyDE similarity to target {sim_hyde} should exceed unrelated {sim_unrelated}"


def test_query_transformation_engine_offline_execution(sample_corpus):
    engine = QueryTransformationEngine(corpus=sample_corpus, groq_api_key=None)

    # 1. Test Naive vs Rewritten
    raw_query = "oom error on gpu"
    rewritten_q, results_rw = engine.retrieve_rewritten(raw_query, top_k=2)
    assert "out of memory" in rewritten_q
    assert any("cuda" in r.doc_id for r in results_rw)

    # 2. Test Step-Back
    cuda_query = "Why is my batch size 128 crashing in PyTorch with CUDA assertion?"
    step_q, results_sb = engine.retrieve_step_back(cuda_query, top_k=2)
    assert len(results_sb) == 2
    assert any("cuda" in r.doc_id for r in results_sb)

    # 3. Test Sub-Query Decomposition
    comp_query = "HNSW vs IVF indexing in ChromaDB"
    sub_qs, results_dec = engine.retrieve_decomposed(comp_query, top_k=3)
    assert len(sub_qs) == 2
    retrieved_ids = [r.doc_id for r in results_dec]
    # Both HNSW and IVF documents should appear in top results!
    assert "doc_hnsw" in retrieved_ids
    assert "doc_ivf" in retrieved_ids

    # 4. Test HyDE
    hypo_passage, results_hyde = engine.retrieve_hyde(colloquial_query:="gpu memory runs out", top_k=2)
    assert len(results_hyde) == 2
    assert any("cuda" in r.doc_id for r in results_hyde)


def test_engine_llm_mode_graceful_fallback(sample_corpus):
    """Verifies that enabling use_llm=True with no API key does not crash and falls back cleanly."""
    engine = QueryTransformationEngine(corpus=sample_corpus, groq_api_key=None)

    rewritten_q, res_rw = engine.retrieve_rewritten("ann in vector db", use_llm=True)
    assert len(res_rw) > 0

    step_q, res_sb = engine.retrieve_step_back("cuda oom on rtx 4090", use_llm=True)
    assert len(res_sb) > 0

    sub_qs, res_dec = engine.retrieve_decomposed("HNSW vs IVF", use_llm=True)
    assert len(sub_qs) >= 2

    hypo_p, res_hyde = engine.retrieve_hyde("gpu memory fragmentation", use_llm=True)
    assert len(res_hyde) > 0
