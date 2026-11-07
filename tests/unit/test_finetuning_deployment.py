"""Unit tests for Track 16: Fine-Tuning versus RAG and Production Deployment.

Verifies:
1. Parametric vs. Non-Parametric trade-off evaluation & decision matrix.
2. Fine-tuning dataset validation (SFT instruction format & contrastive triplets).
3. Low-Rank Adaptation (LoRA) mathematics, B=0 initial state, parameter accounting, and weight merging.
4. Symmetric & asymmetric INT8 CPU quantization, zero-point calibration, and numerical fidelity.
5. Hybrid RAG + Fine-Tuning retrieval, structured synthesis, and refusal gates.
6. FastAPI production ASGI endpoints (/health, /v1/query, /v1/ingest, /v1/metrics).
7. Deployment readiness auditor and Docker containerization specifications.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
from starlette.testclient import TestClient

from common.finetuning_deployment import (
    ArchitectureRequirements,
    ArchitectureType,
    CPUQuantizer,
    DatasetValidator,
    DeploymentReadinessAuditor,
    HybridRAGFineTuningEngine,
    KnowledgeVolatility,
    LoRALinear,
    SFTInstructionSample,
    ContrastiveTripletSample,
    create_rag_production_app,
    evaluate_architecture_tradeoffs,
)


# ============================================================================
# 1. PARAMETRIC VS NON-PARAMETRIC DECISION MATRIX TESTS
# ============================================================================

def test_architecture_tradeoffs_rag_favored_when_high_volatility():
    """Verify RAG is recommended when knowledge is real-time and strict citations are needed."""
    req = ArchitectureRequirements(
        knowledge_volatility=KnowledgeVolatility.REAL_TIME,
        need_strict_citations=True,
        custom_style_or_syntax=False,
        strict_json_schema=False,
        access_control_required=True,
        upfront_budget_usd=100.0,
        training_samples_available=50,
    )
    rec = evaluate_architecture_tradeoffs(req)

    assert rec.primary_architecture == ArchitectureType.RAG_ONLY
    assert rec.confidence_score >= 0.80
    assert any("volatility" in r.lower() for r in rec.rationales)
    assert any("citation" in r.lower() for r in rec.rationales)
    assert rec.estimated_initial_cost_usd < 100.0


def test_architecture_tradeoffs_finetune_favored_when_static_and_custom_style():
    """Verify Fine-Tuning is recommended when knowledge is static, citations are not required, and custom style is needed."""
    req = ArchitectureRequirements(
        knowledge_volatility=KnowledgeVolatility.STATIC,
        need_strict_citations=False,
        custom_style_or_syntax=True,
        strict_json_schema=True,
        access_control_required=False,
        upfront_budget_usd=800.0,
        training_samples_available=1500,
        latency_budget_ms=80.0,
    )
    rec = evaluate_architecture_tradeoffs(req)

    assert rec.primary_architecture in (ArchitectureType.FINETUNE_ONLY, ArchitectureType.HYBRID_RAG_FT)
    assert "format_adherence" in rec.dimension_scores[ArchitectureType.FINETUNE_ONLY.value]
    assert rec.dimension_scores[ArchitectureType.FINETUNE_ONLY.value]["format_adherence"] >= 0.90


def test_architecture_tradeoffs_hybrid_favored_when_citations_and_strict_schema():
    """Verify Hybrid RAG + FT is favored when both dynamic facts/citations and strict format adherence are required."""
    req = ArchitectureRequirements(
        knowledge_volatility=KnowledgeVolatility.DAILY_WEEKLY,
        need_strict_citations=True,
        custom_style_or_syntax=True,
        strict_json_schema=True,
        access_control_required=True,
        upfront_budget_usd=600.0,
        training_samples_available=400,
    )
    rec = evaluate_architecture_tradeoffs(req)

    assert rec.primary_architecture == ArchitectureType.HYBRID_RAG_FT
    assert rec.confidence_score >= 0.85
    assert "ChromaDB" in rec.recommended_serving_stack["retriever"]
    assert "LoRA" in rec.recommended_serving_stack["model_adapter"]


# ============================================================================
# 2. FINE-TUNING DATASET VALIDATION TESTS
# ============================================================================

def test_dataset_validator_sft_valid_and_leakage_detection():
    """Verify SFT validator checks token distribution, duplicates, and target leakage."""
    valid_samples = [
        {"instruction": "Summarize policy A", "input": "Text of policy A.", "output": "Summary A."},
        {"instruction": "Extract latency SLA", "input": "SLA is 200ms.", "output": "200ms latency."},
    ]
    report = DatasetValidator.validate_sft_dataset(valid_samples)
    assert report.is_valid is True
    assert report.valid_count == 2
    assert report.invalid_count == 0
    assert len(report.detected_leakages) == 0

    # Test leakage and duplicate detection
    bad_samples = [
        {"instruction": "Query 1", "input": "Same content", "output": "Same content"}, # leakage: in == out
        {"instruction": "Query 1", "input": "Different", "output": "Something else"},   # duplicate instruction
        {"instruction": "Query 2", "input": "", "output": ""},                          # empty output
    ]
    bad_report = DatasetValidator.validate_sft_dataset(bad_samples)
    assert bad_report.is_valid is False
    assert bad_report.invalid_count >= 1
    assert any("leakage" in l.lower() for l in bad_report.detected_leakages)
    assert any("duplicate" in l.lower() for l in bad_report.detected_leakages)


def test_dataset_validator_contrastive_triplets():
    """Verify contrastive triplet validation detects false negatives (positive == negative)."""
    clean_triplets = [
        {
            "query": "What is the timeout?",
            "positive": "The request timeout is 30 seconds.",
            "negative": "Database replication occurs every hour.",
        }
    ]
    rep = DatasetValidator.validate_contrastive_triplets(clean_triplets)
    assert rep.is_valid is True
    assert rep.valid_count == 1

    flawed_triplets = [
        {
            "query": "What is the timeout?",
            "positive": "The request timeout is 30 seconds.",
            "negative": "The request timeout is 30 seconds.", # identical!
        }
    ]
    rep_flawed = DatasetValidator.validate_contrastive_triplets(flawed_triplets)
    assert any("identical" in l.lower() for l in rep_flawed.detected_leakages)


# ============================================================================
# 3. LORA (LOW-RANK ADAPTATION) MATHEMATICAL TESTS
# ============================================================================

def test_lora_linear_initialization_zero_initial_delta():
    """Verify B is initialized to zero, ensuring Delta W = 0 initially (base model unchanged)."""
    layer = LoRALinear(in_dim=64, out_dim=32, rank=4, alpha=8.0, seed=42)

    delta_w = layer.compute_delta_w()
    assert np.allclose(delta_w, 0.0)
    assert np.allclose(layer.lora_b, 0.0)

    # Initial forward pass equals pure base linear pass
    x = np.random.randn(10, 64).astype(np.float32)
    y_lora = layer.forward(x)
    y_base = x @ layer.w_base.T
    assert np.allclose(y_lora, y_base, atol=1e-6)


def test_lora_linear_parameter_reduction_ratio():
    """Verify parameter count efficiency calculation."""
    in_dim, out_dim, rank = 1024, 1024, 8
    layer = LoRALinear(in_dim=in_dim, out_dim=out_dim, rank=rank)

    # Base parameters: 1024 * 1024 = 1,048,576
    assert layer.base_parameter_count == 1024 * 1024
    # LoRA parameters: 8 * 1024 + 1024 * 8 = 16,384
    assert layer.lora_parameter_count == 2 * rank * in_dim
    # Reduction ratio: ~1.56% of parameters
    expected_ratio = 16384 / 1048576
    assert abs(layer.parameter_reduction_ratio - expected_ratio) < 1e-5


def test_lora_linear_merge_and_unmerge_invariance():
    """Verify weight fusion (merge) produces identical forward pass as unmerged adapter path."""
    layer = LoRALinear(in_dim=16, out_dim=8, rank=2, alpha=4.0, seed=123)
    # Simulate non-zero adapter weights
    layer.lora_b = np.random.randn(8, 2).astype(np.float32)

    x = np.random.randn(5, 16).astype(np.float32)

    y_unmerged = layer.forward(x)

    # Merge weights
    layer.merge_weights()
    assert layer.is_merged is True
    y_merged = layer.forward(x)

    assert np.allclose(y_unmerged, y_merged, atol=1e-5)

    # Unmerge weights
    layer.unmerge_weights()
    assert layer.is_merged is False
    y_restored = layer.forward(x)
    assert np.allclose(y_unmerged, y_restored, atol=1e-5)


# ============================================================================
# 4. CPU QUANTIZATION & NUMERICAL FIDELITY TESTS
# ============================================================================

def test_cpu_quantizer_symmetric_int8():
    """Verify symmetric INT8 quantization maps [-max, max] to [-127, 127] with high cosine fidelity."""
    rng = np.random.default_rng(42)
    matrix = rng.normal(0.0, 1.0, size=(64, 128)).astype(np.float32)

    q_tensor = CPUQuantizer.quantize_symmetric_int8(matrix)

    assert q_tensor.mode == "symmetric"
    assert q_tensor.zero_point == 0
    assert q_tensor.q_data.dtype == np.int8
    assert np.max(q_tensor.q_data) <= 127
    assert np.min(q_tensor.q_data) >= -127

    reconstructed = CPUQuantizer.dequantize(q_tensor)
    metrics = CPUQuantizer.compute_fidelity_metrics(matrix, reconstructed)

    assert metrics["cosine_fidelity"] > 0.999
    assert metrics["rmse"] < 0.05
    assert metrics["memory_compression_ratio"] == 4.0  # FP32 (4 bytes) -> INT8 (1 byte)


def test_cpu_quantizer_asymmetric_int8():
    """Verify asymmetric INT8 quantization handles non-centered distributions cleanly."""
    # Strictly positive distribution (e.g. ReLU activations or probability distributions)
    activations = np.linspace(0.1, 10.0, 256).astype(np.float32)

    q_tensor = CPUQuantizer.quantize_asymmetric_int8(activations)

    assert q_tensor.mode == "asymmetric"
    assert q_tensor.q_data.dtype == np.uint8
    assert np.max(q_tensor.q_data) <= 255
    assert np.min(q_tensor.q_data) >= 0

    reconstructed = CPUQuantizer.dequantize(q_tensor)
    metrics = CPUQuantizer.compute_fidelity_metrics(activations, reconstructed)

    assert metrics["cosine_fidelity"] > 0.9999
    assert metrics["rmse"] < 0.05


# ============================================================================
# 5. HYBRID RAG + FINE-TUNING PIPELINE TESTS
# ============================================================================

def test_hybrid_rag_finetuning_engine_retrieval_and_synthesis():
    """Verify hybrid engine retrieves relevant chunk and generates citation-grounded response."""
    corpus = [
        {
            "id": "c1",
            "text": "All internal API secrets must rotate every 30 days under enterprise security baseline.",
            "metadata": {"source": "SecOps_Doc_Sec_3"},
        },
        {
            "id": "c2",
            "text": "The cafeteria menu features vegan chili on Wednesdays.",
            "metadata": {"source": "Cafeteria_Schedule"},
        },
    ]
    engine = HybridRAGFineTuningEngine(corpus_chunks=corpus)

    chunks = engine.retrieve("How often must internal API secrets rotate?", top_k=1)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "c1"

    synth = engine.synthesize("How often must internal API secrets rotate?", chunks)
    assert synth.is_refusal is False
    assert "SecOps_Doc_Sec_3" in synth.citations
    assert "rotate every 30 days" in synth.answer


def test_hybrid_rag_finetuning_engine_refusal_threshold():
    """Verify synthesis triggers structured refusal when no confident context is found."""
    engine = HybridRAGFineTuningEngine(corpus_chunks=[])
    chunks = engine.retrieve("What is the quantum telemetry bandwidth?", top_k=2)

    synth = engine.synthesize("What is the quantum telemetry bandwidth?", chunks)
    assert synth.is_refusal is True
    assert "REFUSAL" in synth.answer
    assert len(synth.citations) == 0


# ============================================================================
# 6. FASTAPI PRODUCTION SERVING TESTS
# ============================================================================

def test_fastapi_production_app_endpoints():
    """Verify production ASGI endpoints (/health, /v1/query, /v1/ingest, /v1/metrics)."""
    app = create_rag_production_app(service_version="2.4.0")
    client = TestClient(app)

    # 1. Health check (Liveness / Readiness)
    res_health = client.get("/health")
    assert res_health.status_code == 200
    data_h = res_health.json()
    assert data_h["status"] == "healthy"
    assert data_h["version"] == "2.4.0"
    assert "index_size_chunks" in data_h

    # 2. Query endpoint
    query_payload = {"query": "What is the retention period under SOC2?", "top_k": 2}
    res_query = client.post("/v1/query", json=query_payload)
    assert res_query.status_code == 200
    data_q = res_query.json()
    assert "retention" in data_q["answer"].lower() or "soc2" in data_q["answer"].lower()
    assert len(data_q["citations"]) > 0
    assert "X-Process-Time-Ms" in res_query.headers

    # 3. Document Ingest endpoint
    ingest_payload = {
        "documents": [
            {
                "id": "doc_test_100",
                "text": "Zero-trust network policy requires mTLS between all microservices.",
                "metadata": {"source": "ZeroTrust_Manual"},
            }
        ]
    }
    res_ingest = client.post("/v1/ingest", json=ingest_payload)
    assert res_ingest.status_code == 201
    assert res_ingest.json()["ingested_count"] == 1

    # 4. Metrics endpoint
    res_metrics = client.get("/v1/metrics")
    assert res_metrics.status_code == 200
    data_m = res_metrics.json()
    assert data_m["requests_total"] >= 2
    assert data_m["queries_total"] >= 1
    assert "latency_p50_ms" in data_m


# ============================================================================
# 7. DEPLOYMENT READINESS AUDITOR TESTS
# ============================================================================

def test_deployment_readiness_auditor_flags_insecure_and_unquantized_configs():
    """Verify auditor catches root user (UID 0), missing probes, and unquantized runtimes."""
    bad_config = {
        "security": {"non_root_user_uid": 0, "acl_enforcement": False},
        "observability": {"health_endpoint": None, "metrics_endpoint": None},
        "inference": {"quantization_mode": "FP32_UNCOMPRESSED"},
        "server": {"timeout_seconds": 120.0},
        "retrieval": {"vector_store_path": ":memory:"},
    }
    report = DeploymentReadinessAuditor.audit_configuration(bad_config)

    assert report.is_ready_for_production is False
    assert report.compliance_score < 50.0
    assert len(report.recommendations) >= 4
    failed_check_ids = [c.check_id for c in report.results if not c.passed]
    assert "SEC_001" in failed_check_ids  # Non-root user
    assert "OPS_001" in failed_check_ids  # Missing /health probe
    assert "PERF_002" in failed_check_ids # Unquantized runtime
    assert "DATA_001" in failed_check_ids # Ephemeral memory vector store


def test_deployment_readiness_auditor_clean_production_config():
    """Verify a hardened enterprise configuration passes 100% of audit checks."""
    fixture_path = Path("fixtures/deployment/production_rag_config.json")
    assert fixture_path.exists()

    with open(fixture_path, "r", encoding="utf-8") as f:
        clean_config = json.load(f)

    report = DeploymentReadinessAuditor.audit_configuration(clean_config)
    assert report.is_ready_for_production is True
    assert report.compliance_score == 100.0
    assert report.failed_checks == 0
    assert len(report.recommendations) == 0


def test_dockerfile_and_docker_compose_generation():
    """Verify generation of production multi-stage Dockerfile and Docker Compose specs."""
    df = DeploymentReadinessAuditor.generate_dockerfile()
    assert "FROM python:3.12-slim-bookworm AS builder" in df
    assert "FROM python:3.12-slim-bookworm AS runner" in df
    assert "useradd -u 10001" in df
    assert "USER raguser:raggroup" in df
    assert "HEALTHCHECK" in df

    compose = DeploymentReadinessAuditor.generate_docker_compose()
    assert "chroma_data:" in compose
    assert "healthcheck:" in compose
    assert "no-new-privileges:true" in compose
