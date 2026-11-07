"""Unit tests for Advanced Reranking and Cross-Encoder Architectures.

Validates:
1. BiEncoderSimulator embedding properties and candidate retrieval.
2. CrossEncoderSimulator attention matrix generation and negation resolution.
3. ScoreCalibrator sigmoid, min-max, and threshold truncation.
4. CascadedRetrievalPipeline end-to-end multi-stage flow.
5. RankingMetrics calculations (Precision@k, Recall@k, MRR, NDCG@k).
6. GroqLLMReranker graceful fallback in offline mode.
"""

import math
import numpy as np
import pytest

from common.reranking import (
    ScoredDocument,
    BiEncoderSimulator,
    CrossEncoderSimulator,
    ScoreCalibrator,
    CascadedRetrievalPipeline,
    RankingMetrics,
    GroqLLMReranker,
)


@pytest.fixture
def sample_corpus():
    return {
        "doc_chroma_with_docker": (
            "How to run ChromaDB with Docker containers and docker-compose on production Linux servers."
        ),
        "doc_chroma_without_docker": (
            "You can run ChromaDB directly as a Python library without Docker using pip install chromadb."
        ),
        "doc_hnsw_clustering": (
            "HNSW builds multi-layer skip-list graphs for approximate nearest neighbor search."
        ),
        "doc_cuda_out_of_memory": (
            "CUDA out of memory exceptions occur when GPU tensor allocations exceed physical VRAM."
        ),
    }


def test_bi_encoder_vector_normalization_and_scoring():
    bi_encoder = BiEncoderSimulator(dim=64)
    vec = bi_encoder.encode("HNSW vector search")
    assert vec.shape == (64,)
    assert math.isclose(float(np.linalg.norm(vec)), 1.0, rel_tol=1e-5)

    # Identical strings should have cosine ~ 1.0
    sim_self = bi_encoder.score("HNSW vector search", "HNSW vector search")
    assert sim_self > 0.99

    # Unrelated strings should have lower similarity
    sim_diff = bi_encoder.score("HNSW vector search", "banana strawberry smoothie recipe")
    assert sim_self > sim_diff


def test_bi_encoder_candidate_retrieval(sample_corpus):
    bi_encoder = BiEncoderSimulator()
    candidates = bi_encoder.retrieve_candidates(
        query="ChromaDB Docker setup", corpus=sample_corpus, top_k=2
    )
    assert len(candidates) == 2
    assert candidates[0].rank == 1
    assert candidates[1].rank == 2
    assert candidates[0].bi_encoder_score >= candidates[1].bi_encoder_score


def test_cross_encoder_attention_matrix():
    cross_encoder = CrossEncoderSimulator()
    q_tokens, d_tokens, attn = cross_encoder.compute_attention_matrix(
        query="ChromaDB without Docker",
        document="Run ChromaDB directly without Docker",
    )
    assert len(q_tokens) == 3
    assert len(d_tokens) == 5
    assert attn.shape == (3, 5)

    # Softmax row sums should equal 1.0
    row_sums = np.sum(attn, axis=1)
    for s in row_sums:
        assert math.isclose(float(s), 1.0, rel_tol=1e-5)


def test_cross_encoder_resolves_negation_distractor(sample_corpus):
    """Critical verification: Cross-encoder distinguishes 'without Docker' from 'with Docker'."""
    cross_encoder = CrossEncoderSimulator()
    query = "How to run ChromaDB without Docker"

    doc_with = sample_corpus["doc_chroma_with_docker"]
    doc_without = sample_corpus["doc_chroma_without_docker"]

    logit_with = cross_encoder.predict_logit(query, doc_with)
    logit_without = cross_encoder.predict_logit(query, doc_without)

    # The doc explicitly matching 'without Docker' MUST have higher logit
    assert logit_without > logit_with


def test_score_calibrator_sigmoid_and_temperature():
    # Sigmoid properties
    assert math.isclose(ScoreCalibrator.sigmoid(0.0), 0.5, abs_tol=1e-5)
    assert ScoreCalibrator.sigmoid(10.0) > 0.99
    assert ScoreCalibrator.sigmoid(-10.0) < 0.01

    # Temperature scaling
    # High temperature flattens towards 0.5
    soft = ScoreCalibrator.sigmoid(2.0, temperature=5.0)
    sharp = ScoreCalibrator.sigmoid(2.0, temperature=0.5)
    assert 0.5 < soft < sharp

    # Numerical bounds
    assert ScoreCalibrator.sigmoid(100.0) == 1.0
    assert ScoreCalibrator.sigmoid(-100.0) == 0.0


def test_score_calibrator_min_max():
    scores = [2.0, 4.0, 6.0, 10.0]
    norm = ScoreCalibrator.min_max_normalize(scores)
    assert math.isclose(norm[0], 0.0, abs_tol=1e-5)
    assert math.isclose(norm[-1], 1.0, abs_tol=1e-5)
    assert math.isclose(norm[2], 0.5, abs_tol=1e-5)

    # All identical scores
    flat = ScoreCalibrator.min_max_normalize([5.0, 5.0, 5.0])
    assert flat == [1.0, 1.0, 1.0]

    # Empty list
    assert ScoreCalibrator.min_max_normalize([]) == []


def test_score_calibrator_threshold_truncation():
    docs = [
        ScoredDocument(doc_id="d1", text="text1", calibrated_score=0.9),
        ScoredDocument(doc_id="d2", text="text2", calibrated_score=0.7),
        ScoredDocument(doc_id="d3", text="text3", calibrated_score=0.3),
        ScoredDocument(doc_id="d4", text="text4", calibrated_score=0.1),
    ]

    truncated = ScoreCalibrator.truncate_with_threshold(docs, top_k=2, relevance_threshold=0.5)
    assert len(truncated) == 2
    assert [d.doc_id for d in truncated] == ["d1", "d2"]
    assert truncated[0].rank == 1
    assert truncated[1].rank == 2

    # High threshold filtering
    strict = ScoreCalibrator.truncate_with_threshold(docs, top_k=5, relevance_threshold=0.8)
    assert len(strict) == 1
    assert strict[0].doc_id == "d1"


def test_cascaded_pipeline_execution(sample_corpus):
    pipeline = CascadedRetrievalPipeline()
    query = "How to run ChromaDB without Docker"
    candidates, final_results = pipeline.execute(
        query=query,
        corpus=sample_corpus,
        candidate_k=4,
        final_k=2,
        temperature=1.0,
        threshold=0.2,
    )

    assert len(candidates) == 4
    assert len(final_results) <= 2
    # The true answer without docker should be rank 1 in final reranked results
    assert final_results[0].doc_id == "doc_chroma_without_docker"


def test_ranking_metrics_calculations():
    retrieved = ["doc_A", "doc_B", "doc_C", "doc_D"]
    relevant = {"doc_B", "doc_D"}

    # Precision@k
    assert RankingMetrics.precision_at_k(retrieved, relevant, k=1) == 0.0
    assert RankingMetrics.precision_at_k(retrieved, relevant, k=2) == 0.5  # doc_B is hit
    assert RankingMetrics.precision_at_k(retrieved, relevant, k=4) == 0.5  # doc_B and doc_D

    # Recall@k
    assert RankingMetrics.recall_at_k(retrieved, relevant, k=1) == 0.0
    assert RankingMetrics.recall_at_k(retrieved, relevant, k=2) == 0.5
    assert RankingMetrics.recall_at_k(retrieved, relevant, k=4) == 1.0

    # Reciprocal Rank (first relevant doc_B is at index 1 -> rank 2)
    assert math.isclose(RankingMetrics.reciprocal_rank(retrieved, relevant), 0.5)

    # NDCG@k
    grades = {"doc_A": 0.0, "doc_B": 3.0, "doc_C": 0.0, "doc_D": 2.0}
    ndcg_2 = RankingMetrics.ndcg_at_k(retrieved, grades, k=2)
    assert 0.0 < ndcg_2 <= 1.0


def test_groq_llm_reranker_offline_fallback(sample_corpus):
    # Without API key, should use fallback simulator cleanly
    reranker = GroqLLMReranker(api_key=None)
    candidates = [
        ScoredDocument(doc_id=k, text=v, bi_encoder_score=0.5)
        for k, v in sample_corpus.items()
    ]
    results = reranker.rerank(
        query="How to run ChromaDB without Docker",
        candidates=candidates,
        top_k=2,
    )
    assert len(results) == 2
    assert results[0].doc_id == "doc_chroma_without_docker"
