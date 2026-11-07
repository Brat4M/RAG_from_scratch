"""Unit tests for the Automated RAG Evaluation Framework (common/evaluation.py).

Validates:
1. DeterministicSemanticEmbedder normalization and similarity properties.
2. PropositionExtractor atomic claim decomposition and refusal filtering.
3. OfflineClaimVerifier support determination and negation resolution.
4. Context Precision formula and rank-order sensitivity.
5. Context Recall ground-truth claim coverage.
6. Faithfulness hallucination detection and refusal preservation.
7. Answer Relevance query-response alignment.
8. RAG Triad composite harmonic mean calculation.
9. RAGSynthesizer prompt construction, citation parsing, validation, and offline synthesis.
10. RAGEvaluationHarness diagnostic detection of failure modes.
"""

import pytest
import numpy as np

from common.evaluation import (
    ClaimVerification,
    DeterministicSemanticEmbedder,
    EvaluationResult,
    EvaluationSample,
    OfflineClaimVerifier,
    PropositionExtractor,
    RAGEvaluationHarness,
    RAGSynthesizer,
    compute_answer_relevance,
    compute_context_precision,
    compute_context_recall,
    compute_faithfulness,
    compute_rag_triad_score,
)


# ---------------------------------------------------------------------------
# 1. Deterministic Semantic Embedder Tests
# ---------------------------------------------------------------------------

def test_embedder_normalization_and_similarity():
    embedder = DeterministicSemanticEmbedder(dim=128)
    vec = embedder.encode("ChromaDB vector database supports persistent local collections.")
    
    assert vec.shape == (128,)
    norm = np.linalg.norm(vec)
    assert np.isclose(norm, 1.0, atol=1e-5), f"Expected unit norm, got {norm}"

    # Self similarity should be ~1.0
    sim_self = embedder.similarity("Vector search in Python", "Vector search in Python")
    assert sim_self > 0.99

    # Related vs unrelated similarity
    sim_related = embedder.similarity(
        "HNSW graph algorithm for approximate nearest neighbors",
        "Hierarchical Navigable Small World indexing for ANN search",
    )
    sim_unrelated = embedder.similarity(
        "HNSW graph algorithm for approximate nearest neighbors",
        "Baking chocolate chip cookies with organic butter",
    )
    assert sim_related > sim_unrelated, f"Expected {sim_related} > {sim_unrelated}"


# ---------------------------------------------------------------------------
# 2. Proposition Extractor Tests
# ---------------------------------------------------------------------------

def test_proposition_extractor_atomic_claims():
    extractor = PropositionExtractor()
    compound_text = (
        "ChromaDB uses SQLite for metadata persistence, and furthermore it employs "
        "HNSW graphs for high-speed vector retrieval. [Doc 1]"
    )
    claims = extractor.extract_claims(compound_text)
    assert len(claims) >= 2, f"Expected multiple atomic claims, got {claims}"
    # Verify citations are stripped from claims
    for claim in claims:
        assert "[Doc 1]" not in claim


def test_proposition_extractor_refusal_filtering():
    extractor = PropositionExtractor()
    refusal_text = "Based on the provided context, I cannot answer this question as the information is not provided."
    claims = extractor.extract_claims(refusal_text)
    assert claims == [] or len(claims) == 0


# ---------------------------------------------------------------------------
# 3. Offline Claim Verifier Tests
# ---------------------------------------------------------------------------

def test_claim_verifier_supported_and_hallucinated():
    verifier = OfflineClaimVerifier()
    contexts = [
        "ChromaDB stores embedding vectors and provides cosine, L2, and inner product distances.",
        "PyMuPDF extracts raw text, tables, and vector graphics from PDF documents with AGPL license.",
    ]

    # Grounded claim
    v_supported = verifier.verify_claim("ChromaDB supports cosine and inner product distances", contexts)
    assert v_supported.is_supported is True
    assert 0 in v_supported.supporting_context_indices
    assert v_supported.confidence > 0.55

    # Hallucinated / Unsupported claim
    v_hallucinated = verifier.verify_claim("PyMuPDF automatically converts audio files to mp3 format", contexts)
    assert v_hallucinated.is_supported is False
    assert v_hallucinated.supporting_context_indices == []


def test_claim_verifier_negation_contradiction():
    verifier = OfflineClaimVerifier()
    contexts = [
        "ChromaDB does not require a Docker container to run locally in Python.",
    ]
    # Claim asserting the opposite (requiring Docker)
    v_contradiction = verifier.verify_claim("ChromaDB requires a Docker container to run", contexts)
    assert v_contradiction.is_supported is False


# ---------------------------------------------------------------------------
# 4. Context Precision Tests (Rank-Order Sensitivity)
# ---------------------------------------------------------------------------

def test_context_precision_rank_order_penalty():
    query = "How does ChromaDB persist vector embeddings locally?"
    ground_truth = "ChromaDB persists vector embeddings in a local SQLite file and parquet indices."

    rel_chunk = "ChromaDB persists collections locally using SQLite metadata tables and parquet index files."
    distractor_1 = "The French Revolution began in 1789 with the storming of the Bastille."
    distractor_2 = "Photosynthesis converts light energy into chemical energy in green plants."

    # Scenario A: Relevant chunk at Rank 1 (Top)
    contexts_a = [rel_chunk, distractor_1, distractor_2]
    prec_a, flags_a = compute_context_precision(query, contexts_a, ground_truth)

    # Scenario B: Relevant chunk at Rank 3 (Buried below distractors)
    contexts_b = [distractor_1, distractor_2, rel_chunk]
    prec_b, flags_b = compute_context_precision(query, contexts_b, ground_truth)

    # Rank 1 relevant should score 1.0 (Precision@1 = 1/1 = 1.0)
    # Rank 3 relevant should score 0.333 (Precision@3 = 1/3 = 0.333)
    assert prec_a > prec_b, f"Expected rank-sensitive precision: {prec_a} > {prec_b}"
    assert np.isclose(prec_a, 1.0, atol=1e-2)
    assert prec_b < 0.50


def test_context_precision_empty_or_zero_relevant():
    query = "Quantum computing error correction"
    contexts = ["Baking sourdough bread at home.", "Gardening tips for spring tomatoes."]
    prec, flags = compute_context_precision(query, contexts, ground_truth="")
    assert prec == 0.0
    assert not any(flags)


# ---------------------------------------------------------------------------
# 5. Context Recall Tests
# ---------------------------------------------------------------------------

def test_context_recall_full_and_partial():
    ground_truth = (
        "PyMuPDF parses PDF text and extract metadata, "
        "and ChromaDB indexes the resulting embeddings for nearest neighbor search."
    )

    full_contexts = [
        "PyMuPDF parses PDF text and extracts document metadata cleanly.",
        "ChromaDB indexes embedding vectors for nearest neighbor similarity search.",
    ]
    recall_full, verifs_full = compute_context_recall(ground_truth, full_contexts)
    assert recall_full >= 0.99, f"Expected near 1.0 recall, got {recall_full}"

    partial_contexts = [
        "PyMuPDF parses PDF text and extracts document metadata cleanly.",
        "Bananas are a rich dietary source of potassium and vitamin B6.",
    ]
    recall_partial, verifs_partial = compute_context_recall(ground_truth, partial_contexts)
    assert 0.0 < recall_partial < 1.0, f"Expected partial recall, got {recall_partial}"


# ---------------------------------------------------------------------------
# 6. Faithfulness Tests
# ---------------------------------------------------------------------------

def test_faithfulness_grounded_vs_hallucinated():
    contexts = [
        "HNSW constructs a multi-layer graph with geometric decay factor mL = 1 / ln(M).",
        "Top layers have long-range skip edges for logarithmic search complexity.",
    ]

    grounded_ans = "HNSW builds a multi-layer graph where top layers have long-range skip edges."
    faith_grounded, verifs_g = compute_faithfulness(grounded_ans, contexts)
    assert faith_grounded >= 0.99, f"Expected 1.0 faithfulness, got {faith_grounded}"

    hallucinated_ans = (
        "HNSW builds a multi-layer graph where top layers have skip edges, "
        "and it automatically trains a transformer neural network on GPU clusters."
    )
    faith_hallucinated, verifs_h = compute_faithfulness(hallucinated_ans, contexts)
    assert faith_hallucinated < 0.75, f"Expected lower faithfulness on hallucination, got {faith_hallucinated}"


def test_faithfulness_refusal_preservation():
    contexts = ["Python 3.12 was released in October 2023."]
    refusal_ans = "Based on the provided context, I cannot answer questions regarding Rust memory safety."
    faith, verifs = compute_faithfulness(refusal_ans, contexts)
    assert faith == 1.0
    assert "refusal" in verifs[0].reason.lower()


# ---------------------------------------------------------------------------
# 7. Answer Relevance Tests
# ---------------------------------------------------------------------------

def test_answer_relevance_on_topic_vs_drift():
    query = "What is the primary role of cross-encoder rerankers in RAG?"
    on_topic = "Cross-encoder rerankers perform joint cross-attention over query and document tokens to eliminate false positives."
    off_topic = "Mount Everest is the highest mountain peak above sea level in the Himalayas."

    rel_on = compute_answer_relevance(query, on_topic)
    rel_off = compute_answer_relevance(query, off_topic)

    assert rel_on > rel_off, f"Expected {rel_on} > {rel_off}"
    assert rel_on > 0.60
    assert rel_off < 0.35


# ---------------------------------------------------------------------------
# 8. RAG Triad Harmonic Score Tests
# ---------------------------------------------------------------------------

def test_rag_triad_score_harmonic_balance():
    # Balanced high scores
    score_balanced = compute_rag_triad_score(1.0, 1.0, 1.0)
    assert np.isclose(score_balanced, 1.0, atol=1e-3)

    # If one pillar is near zero (e.g. pure hallucination), harmonic mean must collapse
    score_collapsed = compute_rag_triad_score(1.0, 0.05, 1.0)
    assert score_collapsed < 0.20, f"Expected harmonic mean to collapse on failure, got {score_collapsed}"


# ---------------------------------------------------------------------------
# 9. RAG Synthesizer Tests
# ---------------------------------------------------------------------------

def test_synthesizer_prompt_and_citations():
    synth = RAGSynthesizer(groq_api_key="mock_key")
    contexts = [
        "ChromaDB is an open-source AI application database.",
        "BM25 is a term-frequency ranking algorithm for lexical retrieval.",
    ]
    prompt = synth.build_prompt("Compare ChromaDB and BM25", contexts)
    assert "=== CONTEXT PASSAGES ===" in prompt
    assert "[Doc 1] (Source ID: chunk_01)" in prompt
    assert "[Doc 2] (Source ID: chunk_02)" in prompt

    answer = "ChromaDB provides local vector storage [Doc 1], whereas BM25 ranks tokens lexically [Doc 2]."
    citations = synth.extract_citations(answer)
    assert citations == [1, 2]

    audit = synth.validate_citations(answer, num_contexts=2)
    assert audit["has_citations"] is True
    assert audit["all_citations_valid"] is True

    # Test out of bounds citation
    bad_answer = "ChromaDB is scalable [Doc 99]."
    bad_audit = synth.validate_citations(bad_answer, num_contexts=2)
    assert bad_audit["all_citations_valid"] is False
    assert 99 in bad_audit["invalid_citations"]


def test_synthesizer_offline_synthesis_and_refusal():
    synth = RAGSynthesizer(groq_api_key="mock_key")
    contexts = [
        "Hierarchical Navigable Small World (HNSW) uses logarithmic skip lists.",
    ]
    answer = synth.synthesize_offline("How does HNSW navigate?", contexts)
    assert "[Doc 1]" in answer
    assert "HNSW" in answer or "Navigable" in answer

    # Refusal test on totally irrelevant context
    irrelevant_contexts = ["Italian pizza recipes require double-zero flour."]
    refusal_answer = synth.synthesize_offline("Explain quantum superposition in physics", irrelevant_contexts)
    assert "cannot answer" in refusal_answer.lower()


# ---------------------------------------------------------------------------
# 10. Evaluation Harness & Diagnostic Diagnostics Tests
# ---------------------------------------------------------------------------

def test_evaluation_harness_sample_and_dataset():
    harness = RAGEvaluationHarness()

    sample = EvaluationSample(
        query="What is the advantage of Cross-Encoder reranking?",
        contexts=[
            "Cross-Encoders resolve bi-encoder representation bottlenecks by applying joint attention over all token pairs.",
            "Bi-encoders generate embeddings independently without cross-token interaction.",
        ],
        ground_truth="Cross-encoders perform full cross-attention over query and document tokens to resolve bi-encoder bottlenecks.",
        generated_answer="Cross-Encoders eliminate bi-encoder representation bottlenecks through joint attention across tokens [Doc 1].",
        sample_id="test_01",
    )

    res = harness.evaluate_sample(sample)
    assert isinstance(res, EvaluationResult)
    assert res.context_precision >= 0.80
    assert res.context_recall >= 0.80
    assert res.faithfulness >= 0.80
    assert res.answer_relevance >= 0.60
    assert res.diagnostics["pipeline_status"] == "PASS"

    dataset_summary = harness.evaluate_dataset([sample])
    assert dataset_summary["total_samples"] == 1
    assert "mean_context_precision" in dataset_summary
    assert "mean_rag_triad_score" in dataset_summary
