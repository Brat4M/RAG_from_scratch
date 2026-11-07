"""Unit tests for Track 15: Evaluation and Observability Engine.

Validates OpenTelemetry distributed tracing, PII sanitization firewall,
automated evaluation metrics (Precision, Recall, Faithfulness, Relevance, Citation Accuracy),
semantic drift detection (MMD, KS-test), latency and cost accounting, and human calibration.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from common.observability_evaluation import (
    DeterministicSemanticEmbedder,
    PIIRedactor,
    RAGTelemetryManager,
    ContextPrecisionEvaluator,
    ContextRecallEvaluator,
    FaithfulnessEvaluator,
    AnswerRelevanceEvaluator,
    CitationAccuracyEvaluator,
    RAGTriadScorer,
    SemanticDriftDetector,
    TelemetryMetricsTracker,
    HumanCalibrationAnalyzer,
    TracedRAGPipeline,
    compute_ks_2samp,
    compute_pearson_r,
    compute_spearman_rho,
    generate_observability_tracing_diagram,
    generate_evaluation_and_drift_diagram,
)


@pytest.fixture
def benchmark_fixture():
    fixture_path = Path("fixtures/evaluation_benchmarks/rag_curated_eval_benchmark.json")
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_deterministic_semantic_embedder():
    embedder = DeterministicSemanticEmbedder(dim=32)
    vec1 = embedder.embed_text("H100 GPU datacenter liquid cooling")
    vec2 = embedder.embed_text("H100 GPU datacenter liquid cooling")
    vec3 = embedder.embed_text("Downtown Italian pizza restaurant menu")

    assert vec1.shape == (32,)
    assert np.isclose(np.linalg.norm(vec1), 1.0, atol=1e-4)
    # Identical texts must have cosine similarity 1.0
    assert np.isclose(embedder.cosine_similarity(vec1, vec2), 1.0, atol=1e-4)
    # Dissimilar texts have lower similarity
    assert embedder.cosine_similarity(vec1, vec3) < 0.75

    batch = embedder.embed_batch(["alpha", "beta", "gamma"])
    assert batch.shape == (3, 32)


def test_pii_redactor():
    dirty_text = "Contact alice.dev@company.corp at 192.168.1.100 with card 4111-2222-3333-4444 and bearer: secret_token_xyz987"
    sanitized = PIIRedactor.sanitize(dirty_text)

    assert "[REDACTED_EMAIL]" in sanitized
    assert "[REDACTED_IP]" in sanitized
    assert "[REDACTED_CARD]" in sanitized
    assert "[REDACTED_SECRET]" in sanitized
    assert "alice.dev@company.corp" not in sanitized
    assert "192.168.1.100" not in sanitized

    dirty_dict = {
        "user_email": "ops@internal.net",
        "ip_list": ["10.0.0.1", "172.16.0.5"],
        "status": "active"
    }
    clean_dict = PIIRedactor.sanitize_attributes(dirty_dict)
    assert clean_dict["user_email"] == "[REDACTED_EMAIL]"
    assert clean_dict["ip_list"] == ["[REDACTED_IP]", "[REDACTED_IP]"]
    assert clean_dict["status"] == "active"


def test_opentelemetry_tracing_and_waterfall():
    telemetry = RAGTelemetryManager("test-tracer")
    telemetry.clear()

    with telemetry.start_span("parent_span", attributes={"user": "bob@test.com"}):
        with telemetry.start_span("child_retrieval", attributes={"k": 5}):
            pass
        with telemetry.start_span("child_generation", attributes={"tokens": 42}):
            pass

    spans = telemetry.get_finished_spans()
    assert len(spans) == 3

    # Check PII redaction on span attribute
    parent_span = next(s for s in spans if s.name == "parent_span")
    assert parent_span.attributes["user"] == "[REDACTED_EMAIL]"

    timeline = telemetry.build_waterfall_timeline()
    assert len(timeline) == 3
    assert timeline[0]["offset_ms"] >= 0.0


def test_context_precision_and_recall():
    contexts = [
        "The H100 datacenter pod utilizes direct-to-chip liquid cooling loops.",
        "Datacenter air conditioning backup systems run on secondary chillers.",
        "Irrelevant notes on cafeteria scheduling and holiday hours."
    ]
    ground_truth = "The H100 pod employs direct-to-chip liquid cooling loops."

    prec, flags = ContextPrecisionEvaluator.evaluate(contexts, ground_truth)
    assert flags[0] is True
    assert prec > 0.8

    rec, details = ContextRecallEvaluator.evaluate(contexts, ground_truth)
    assert rec >= 0.8
    assert len(details["covered_facts"]) >= 1


def test_faithfulness_and_hallucination_detection():
    contexts = [
        "Standard Sec-402 specifies TLS 1.3 encryption across all cluster network transmission boundaries."
    ]
    # Grounded answer
    grounded_ans = "Standard Sec-402 enforces TLS 1.3 across all cluster network transmission boundaries."
    faith_grounded, _ = FaithfulnessEvaluator.evaluate(grounded_ans, contexts)
    assert faith_grounded >= 0.9

    # Hallucinated answer
    hallucinated_ans = "Standard Sec-402 mandates quantum-resistant lattice AES-512 with RSA-8192 root key hierarchies."
    faith_hallucinated, details = FaithfulnessEvaluator.evaluate(hallucinated_ans, contexts)
    assert faith_hallucinated < 0.5
    assert details[0]["supported"] is False


def test_citation_accuracy():
    contexts = [
        "Doc 0: Enterprise Tier service level agreements guarantee 99.99% monthly uptime availability.",
        "Doc 1: Unexcused SLA breaches receive a 25% credit on monthly recurring billing."
    ]
    valid_cited_ans = "Enterprise Tier guarantees 99.99% uptime [Doc 0], while unexcused breaches receive a 25% credit [Doc 1]."
    acc_valid, _ = CitationAccuracyEvaluator.evaluate(valid_cited_ans, contexts)
    assert acc_valid == 1.0

    swapped_cited_ans = "Enterprise Tier guarantees 99.99% uptime [Doc 1], while unexcused breaches receive a 25% credit [Doc 0]."
    acc_swapped, details = CitationAccuracyEvaluator.evaluate(swapped_cited_ans, contexts)
    assert acc_swapped == 0.0
    assert len(details) == 2


def test_rag_triad_harmonic_mean():
    triad_score = RAGTriadScorer.compute(1.0, 1.0, 1.0)
    assert np.isclose(triad_score, 1.0, atol=1e-3)

    degraded_score = RAGTriadScorer.compute(0.0, 1.0, 1.0)
    # Harmonic mean severely penalizes zero in any dimension
    assert degraded_score < 0.01


def test_pure_numpy_statistical_helpers():
    # Pearson & Spearman
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    assert np.isclose(compute_pearson_r(x, y), 1.0, atol=1e-3)
    assert np.isclose(compute_spearman_rho(x, y), 1.0, atol=1e-3)

    # KS-test
    d1 = np.array([0.1, 0.2, 0.3, 0.4])
    d2 = np.array([0.1, 0.2, 0.3, 0.4])
    d_stat, p_val = compute_ks_2samp(d1, d2)
    assert np.isclose(d_stat, 0.0, atol=1e-3)


def test_semantic_drift_detector():
    detector = SemanticDriftDetector(drift_threshold=0.10)
    baseline_queries = [
        "What is the cooling mechanism for the GPU cluster?",
        "How do liquid coolant loops stabilize datacenter thermal load?",
        "Thermal shutdown threshold for high performance computing nodes.",
        "Cluster refrigeration and power distribution unit specifications."
    ]
    detector.fit_baseline(baseline_queries)

    # In-domain test batch (no drift)
    in_domain_batch = [
        "Datacenter thermal thresholds and liquid chiller loops.",
        "Operating temperature range for enterprise GPU pods."
    ]
    report_in = detector.evaluate_drift(in_domain_batch)
    assert report_in.severity in ("NORMAL", "WARNING")

    # Out-of-domain test batch (severe semantic drift)
    drift_batch = [
        "Best Italian pasta tasting menu downtown near convention center.",
        "Where can I buy gourmet espresso beans and pastries?",
        "Top romantic dining restaurants with outdoor patio seating."
    ]
    report_drift = detector.evaluate_drift(drift_batch)
    assert report_drift.drift_detected is True
    assert report_drift.severity in ("WARNING", "CRITICAL")
    assert report_drift.mmd_score > report_in.mmd_score


def test_telemetry_metrics_tracker():
    tracker = TelemetryMetricsTracker("groq-llama3-70b")
    m1 = tracker.record_execution(retrieval_ms=12.5, generation_ms=45.0, prompt_tokens=150, completion_tokens=50)
    m2 = tracker.record_execution(retrieval_ms=18.0, generation_ms=62.0, prompt_tokens=200, completion_tokens=80)

    assert m1.total_latency_ms == 57.5
    assert m1.estimated_cost_usd > 0.0

    summary = tracker.compute_summary_percentiles()
    assert summary["total_requests"] == 2
    assert summary["p50"] > 0.0
    assert summary["p99"] >= summary["p50"]


def test_human_calibration_analyzer():
    auto_scores = [0.95, 0.10, 0.70, 0.90, 0.20]
    human_scores = [1.00, 0.00, 0.67, 1.00, 0.10]

    calib = HumanCalibrationAnalyzer.compute_correlations(auto_scores, human_scores)
    assert calib["pearson_r"] > 0.95
    assert calib["spearman_rho"] >= 0.90
    assert calib["mae"] < 0.10


def test_traced_rag_pipeline_curated_benchmark(benchmark_fixture):
    pipeline = TracedRAGPipeline()

    cards = []
    for sample in benchmark_fixture:
        card, metrics = pipeline.execute_and_evaluate(sample)
        cards.append(card)
        assert metrics.total_tokens > 0

    assert len(cards) == 5

    # Sample 1: Perfect retrieval and groundedness
    s1 = cards[0]
    assert s1.faithfulness == 1.0
    assert s1.citation_accuracy == 1.0

    # Sample 2: Hallucination sample
    s2 = cards[1]
    assert s2.faithfulness < 0.5

    # Sample 4: Citation swapped
    s4 = cards[3]
    assert s4.citation_accuracy == 0.0

    # Verify telemetry captured spans
    finished_spans = pipeline.telemetry.get_finished_spans()
    assert len(finished_spans) >= 20


def test_diagram_generation_helpers(tmp_path):
    diag1 = tmp_path / "test_tracing.png"
    diag2 = tmp_path / "test_eval_drift.png"

    generate_observability_tracing_diagram(str(diag1))
    generate_evaluation_and_drift_diagram(str(diag2))

    assert diag1.exists()
    assert diag1.stat().st_size > 1000
    assert diag2.exists()
    assert diag2.stat().st_size > 1000
