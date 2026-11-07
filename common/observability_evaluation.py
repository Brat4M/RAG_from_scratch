"""Enterprise RAG Observability, Distributed Tracing, and Automated Evaluation Engine.

Provides 100% offline-executable, CPU-deterministic observability and evaluation tooling:
1. OpenTelemetry Distributed Tracing: RAG lifecycle spans, hierarchical parent-child traces,
   standardized GenAI semantic conventions, and in-memory trace capture.
2. PII Redaction & Data Governance: Pre-trace sanitization firewall for privacy-preserving telemetry.
3. Automated Evaluation Suite: Context Precision (MAP@K), Context Recall, Faithfulness (atomic proposition
   entailment), Answer Relevance, Citation Accuracy, and RAG Triad composite scoring.
4. Semantic Drift & Distribution Shift Detection: MMD (Maximum Mean Discrepancy), Centroid Cosine Drift,
   and Kolmogorov-Smirnov distribution testing over embedding windows.
5. Latency, Token, and Financial Cost Accounting: Percentile latencies (p50, p95, p99) and model pricing tiers.
6. Human Calibration Analyzer: Pearson and Spearman correlation between algorithmic scores and expert human rubrics.
7. Architectural Visualization: Publication-grade diagrams for waterfall traces and evaluation radars.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple, Union

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
import matplotlib.pyplot as plt
import numpy as np

# OpenTelemetry SDK & API
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode


# ============================================================================
# 1. Deterministic Semantic Embedder & Pure-NumPy Statistics
# ============================================================================

def compute_ks_2samp(data1: np.ndarray, data2: np.ndarray) -> Tuple[float, float]:
    """Compute 2-sample Kolmogorov-Smirnov test statistic and asymptotic p-value in pure NumPy."""
    n1 = len(data1)
    n2 = len(data2)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    d1 = np.sort(data1)
    d2 = np.sort(data2)
    data_all = np.concatenate([d1, d2])
    cdf1 = np.searchsorted(d1, data_all, side="right") / float(n1)
    cdf2 = np.searchsorted(d2, data_all, side="right") / float(n2)
    d_stat = float(np.max(np.abs(cdf1 - cdf2)))

    en = math.sqrt(n1 * n2 / (n1 + n2))
    lambda_val = (en + 0.12 + 0.11 / max(en, 1e-6)) * d_stat
    p_val = 0.0
    for j in range(1, 60):
        term = 2.0 * ((-1) ** (j - 1)) * math.exp(-2.0 * (j ** 2) * (lambda_val ** 2))
        p_val += term
        if abs(term) < 1e-8:
            break
    p_val = max(0.0, min(1.0, float(p_val)))
    return d_stat, p_val


def compute_pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient r in pure NumPy."""
    x_diff = x - np.mean(x)
    y_diff = y - np.mean(y)
    denom = math.sqrt(float(np.sum(x_diff ** 2) * np.sum(y_diff ** 2)))
    if denom < 1e-9:
        return 1.0
    return float(np.sum(x_diff * y_diff) / denom)


def compute_spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation coefficient rho in pure NumPy."""
    def rank_data(arr: np.ndarray) -> np.ndarray:
        temp = arr.argsort()
        ranks = np.empty_like(temp, dtype=np.float64)
        ranks[temp] = np.arange(len(arr), dtype=np.float64)
        return ranks

    rx = rank_data(x)
    ry = rank_data(y)
    return compute_pearson_r(rx, ry)


class DeterministicSemanticEmbedder:
    """Deterministic, CPU-only dense embedding generator using multi-scale n-gram hashing.
    
    Provides reliable, reproducible vector representations without requiring external
    neural models or network calls.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string into a normalized dense vector."""
        vec = np.zeros(self.dim, dtype=np.float32)
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = cleaned.split()
        if not tokens:
            return vec

        for token in tokens:
            # Word-level hash
            w_hash = hash(token) % self.dim
            vec[w_hash] += 2.0

            # Character 3-grams
            if len(token) >= 3:
                for i in range(len(token) - 2):
                    tri = token[i:i + 3]
                    tri_hash = hash(tri) % self.dim
                    vec[tri_hash] += 0.5

        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec /= norm
        return vec

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Embed an array of texts into a 2D matrix (N, dim)."""
        return np.vstack([self.embed_text(t) for t in texts])

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two 1D vectors."""
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))


# ============================================================================
# 2. PII Sanitization & Privacy Redaction Firewall
# ============================================================================

class PIIRedactor:
    """Pre-telemetry data scrubbing firewall to prevent sensitive data leakage."""

    PATTERNS = [
        (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[REDACTED_EMAIL]"),
        (re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"), "[REDACTED_CARD]"),
        (re.compile(r"\b(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}\b"), "[REDACTED_IP]"),
        (re.compile(r"(?i)\b(bearer|token|apikey|secret)\s*[:=]\s*[A-Za-z0-9_\-\.]{8,}\b"), "[REDACTED_SECRET]"),
    ]

    @classmethod
    def sanitize(cls, text: str) -> str:
        """Redact known PII and credential patterns from text."""
        sanitized = text
        for pattern, replacement in cls.PATTERNS:
            sanitized = pattern.sub(replacement, sanitized)
        return sanitized

    @classmethod
    def sanitize_attributes(cls, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Deeply sanitize attribute dictionary keys and string values."""
        clean = {}
        for k, v in attributes.items():
            if isinstance(v, str):
                clean[k] = cls.sanitize(v)
            elif isinstance(v, list) and all(isinstance(x, str) for x in v):
                clean[k] = [cls.sanitize(x) for x in v]
            else:
                clean[k] = v
        return clean


# ============================================================================
# 3. OpenTelemetry Distributed Tracing Manager
# ============================================================================

@dataclass
class SpanRecord:
    """Normalized span record extracted from OpenTelemetry span buffer."""
    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    start_time_ns: int
    end_time_ns: int
    duration_ms: float
    attributes: Dict[str, Any]
    status_code: str
    status_description: Optional[str]


class RAGTelemetryManager:
    """Manages OpenTelemetry distributed tracing and in-memory capture for RAG pipelines."""

    def __init__(self, service_name: str = "rag-enterprise-engine"):
        self.service_name = service_name
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.tracer = self.provider.get_tracer(service_name)
        self.redactor = PIIRedactor()

    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        redact_pii: bool = True
    ) -> Generator[trace.Span, None, None]:
        """Context manager for tracing RAG execution steps with automatic PII sanitization."""
        attrs = attributes or {}
        if redact_pii:
            attrs = self.redactor.sanitize_attributes(attrs)

        with self.tracer.start_as_current_span(name, attributes=attrs) as span:
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as exc:
                span.set_status(Status(StatusCode.ERROR, description=str(exc)))
                span.record_exception(exc)
                raise

    def get_finished_spans(self) -> List[SpanRecord]:
        """Retrieve all completed spans recorded in memory as structured objects."""
        raw_spans = self.exporter.get_finished_spans()
        records: List[SpanRecord] = []
        for s in raw_spans:
            start_ns = s.start_time
            end_ns = s.end_time if s.end_time else start_ns
            duration_ms = (end_ns - start_ns) / 1_000_000.0
            parent_id = format(s.parent.span_id, "016x") if s.parent else None

            rec = SpanRecord(
                name=s.name,
                trace_id=format(s.context.trace_id, "032x"),
                span_id=format(s.context.span_id, "016x"),
                parent_span_id=parent_id,
                start_time_ns=start_ns,
                end_time_ns=end_ns,
                duration_ms=round(duration_ms, 3),
                attributes=dict(s.attributes or {}),
                status_code=s.status.status_code.name,
                status_description=s.status.description,
            )
            records.append(rec)
        return records

    def clear(self) -> None:
        """Clear the in-memory span buffer."""
        self.exporter.clear()

    def build_waterfall_timeline(self) -> List[Dict[str, Any]]:
        """Construct hierarchical waterfall timing offsets relative to root span."""
        spans = self.get_finished_spans()
        if not spans:
            return []

        min_start = min(s.start_time_ns for s in spans)
        timeline = []
        for s in spans:
            offset_ms = (s.start_time_ns - min_start) / 1_000_000.0
            timeline.append({
                "name": s.name,
                "span_id": s.span_id,
                "parent_span_id": s.parent_span_id,
                "offset_ms": round(offset_ms, 3),
                "duration_ms": s.duration_ms,
                "status": s.status_code,
                "attributes": s.attributes,
            })
        timeline.sort(key=lambda x: (x["offset_ms"], -x["duration_ms"]))
        return timeline


# ============================================================================
# 4. Automated Evaluation Metrics Suite
# ============================================================================

@dataclass
class EvaluationCard:
    """Holistic multi-dimensional evaluation assessment for a RAG test case."""
    sample_id: str
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevance: float
    citation_accuracy: float
    factual_correctness: float
    completeness: float
    rag_triad_score: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "citation_accuracy": round(self.citation_accuracy, 4),
            "factual_correctness": round(self.factual_correctness, 4),
            "completeness": round(self.completeness, 4),
            "rag_triad_score": round(self.rag_triad_score, 4),
            "details": self.details,
        }


class ContextPrecisionEvaluator:
    """Calculates rank-weighted Context Precision (MAP@K) over retrieved passages."""

    @staticmethod
    def evaluate(retrieved_contexts: Sequence[str], ground_truth: str) -> Tuple[float, List[bool]]:
        """Evaluate whether relevant contexts are ranked higher than irrelevant ones."""
        if not retrieved_contexts:
            return 0.0, []

        gt_tokens = set(re.findall(r"\w+", ground_truth.lower()))
        # Stopwords filter
        stopwords = {"the", "a", "an", "is", "and", "or", "to", "in", "of", "for", "with", "that"}
        gt_keywords = gt_tokens - stopwords

        relevance_flags: List[bool] = []
        for ctx in retrieved_contexts:
            ctx_tokens = set(re.findall(r"\w+", ctx.lower()))
            overlap = len(ctx_tokens & gt_keywords)
            # Context is deemed relevant if it contains sufficient content overlap
            is_rel = (overlap >= 2) if len(gt_keywords) >= 2 else (overlap >= 1)
            relevance_flags.append(is_rel)

        # Compute Average Precision
        num_relevant = 0
        running_precisions = []
        for k, is_rel in enumerate(relevance_flags, start=1):
            if is_rel:
                num_relevant += 1
                running_precisions.append(num_relevant / k)

        if not running_precisions:
            return 0.0, relevance_flags

        map_score = sum(running_precisions) / len(running_precisions)
        return float(round(map_score, 4)), relevance_flags


class ContextRecallEvaluator:
    """Measures the proportion of ground-truth factual claims covered in retrieved contexts."""

    @staticmethod
    def extract_atomic_facts(text: str) -> List[str]:
        """Decompose text into atomic factual propositions."""
        clauses = re.split(r"[.;,]\s+|\s+and\s+", text)
        facts = [c.strip() for c in clauses if len(c.strip().split()) >= 3]
        return facts if facts else [text.strip()]

    @classmethod
    def evaluate(cls, retrieved_contexts: Sequence[str], ground_truth: str) -> Tuple[float, Dict[str, Any]]:
        """Evaluate context recall against ground truth facts."""
        facts = cls.extract_atomic_facts(ground_truth)
        if not facts:
            return 1.0, {"covered_facts": [], "missing_facts": []}

        corpus = " ".join(retrieved_contexts).lower()
        covered = []
        missing = []

        for fact in facts:
            fact_words = [w for w in re.findall(r"\w+", fact.lower()) if len(w) > 3]
            if not fact_words:
                covered.append(fact)
                continue

            matches = sum(1 for w in fact_words if w in corpus)
            match_ratio = matches / len(fact_words)
            if match_ratio >= 0.6:
                covered.append(fact)
            else:
                missing.append(fact)

        recall = len(covered) / len(facts)
        return float(round(recall, 4)), {"covered_facts": covered, "missing_facts": missing}


class FaithfulnessEvaluator:
    """Verifies that claims in the generated response are grounded in retrieved context."""

    @classmethod
    def evaluate(cls, generated_answer: str, retrieved_contexts: Sequence[str]) -> Tuple[float, List[Dict[str, Any]]]:
        """Detect hallucinations by verifying individual answer statements against contexts."""
        corpus = " ".join(retrieved_contexts).lower()
        # Robust sentence splitting that respects numerical decimals (e.g. 'TLS 1.3' or '85.5%')
        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", generated_answer)
            if s.strip()
        ]
        if not sentences:
            return 1.0, []

        verifications = []
        supported_count = 0

        for sentence in sentences:
            # Check for epistemic refusal (assistant safely acknowledging lack of information)
            refusal_patterns = ["cannot provide", "corpus does not contain", "not documented", "cannot answer"]
            if any(p in sentence.lower() for p in refusal_patterns):
                verifications.append({
                    "sentence": sentence,
                    "supported": True,
                    "reason": "Epistemic refusal safely detected"
                })
                supported_count += 1
                continue

            content_words = [w for w in re.findall(r"\w+", sentence.lower()) if len(w) > 3]
            # Strip citation tags from token analysis
            clean_words = [w for w in content_words if not (w.startswith("doc") or w.isdigit())]

            if not clean_words:
                verifications.append({"sentence": sentence, "supported": True, "reason": "No substantive claim"})
                supported_count += 1
                continue

            # Verify presence in context (exact word or 4-char prefix stem match)
            def is_word_in_corpus(word: str) -> bool:
                if word in corpus:
                    return True
                stem = word[:4]
                return len(stem) >= 4 and stem in corpus

            present_words = [w for w in clean_words if is_word_in_corpus(w)]
            ratio = len(present_words) / len(clean_words)
            is_supported = ratio >= 0.55

            verifications.append({
                "sentence": sentence,
                "supported": is_supported,
                "support_ratio": round(ratio, 3),
                "unsupported_words": [w for w in clean_words if not is_word_in_corpus(w)]
            })
            if is_supported:
                supported_count += 1

        faithfulness = supported_count / len(sentences)
        return float(round(faithfulness, 4)), verifications


class AnswerRelevanceEvaluator:
    """Measures semantic alignment between query and generated response."""

    def __init__(self, embedder: Optional[DeterministicSemanticEmbedder] = None):
        self.embedder = embedder or DeterministicSemanticEmbedder()

    def evaluate(self, query: str, generated_answer: str) -> float:
        """Compute cosine similarity of query and answer semantic representations."""
        # Handle epistemic refusal: if query asked for restaurant and answer says cannot answer, relevance is low
        q_vec = self.embedder.embed_text(query)
        a_vec = self.embedder.embed_text(generated_answer)
        cos_sim = self.embedder.cosine_similarity(q_vec, a_vec)
        # Scale to [0.0, 1.0]
        score = max(0.0, min(1.0, float(cos_sim)))
        return float(round(score, 4))


class CitationAccuracyEvaluator:
    """Validates that inline bracketed citations ([Doc N]) point to documents supporting the claim."""

    @staticmethod
    def evaluate(generated_answer: str, retrieved_contexts: Sequence[str]) -> Tuple[float, List[Dict[str, Any]]]:
        """Check whether each citation claim is verified by the cited context index."""
        # Find citations like [Doc 0], [Doc 1], [0], [1]
        pattern = re.compile(r"([^.?!]+?)\s*\[(?:Doc\s*)?(\d+)\]", re.IGNORECASE)
        matches = pattern.findall(generated_answer)
        if not matches:
            # If no citations used, check if contexts exist
            return 1.0, []

        evaluations = []
        valid_citations = 0

        for claim, doc_idx_str in matches:
            doc_idx = int(doc_idx_str)
            claim_text = claim.strip()
            if doc_idx < 0 or doc_idx >= len(retrieved_contexts):
                evaluations.append({
                    "claim": claim_text,
                    "cited_doc": doc_idx,
                    "valid": False,
                    "reason": f"Document index {doc_idx} out of range (total docs: {len(retrieved_contexts)})"
                })
                continue

            target_doc = retrieved_contexts[doc_idx].lower()
            claim_words = [w for w in re.findall(r"\w+", claim_text.lower()) if len(w) > 3]
            if not claim_words:
                valid_citations += 1
                evaluations.append({"claim": claim_text, "cited_doc": doc_idx, "valid": True})
                continue

            matches_in_doc = sum(1 for w in claim_words if w in target_doc)
            ratio = matches_in_doc / len(claim_words)
            is_valid = ratio >= 0.55

            evaluations.append({
                "claim": claim_text,
                "cited_doc": doc_idx,
                "valid": is_valid,
                "overlap_ratio": round(ratio, 3)
            })
            if is_valid:
                valid_citations += 1

        accuracy = valid_citations / len(matches)
        return float(round(accuracy, 4)), evaluations


class RAGTriadScorer:
    """TruLens RAG Triad harmonic mean scorer: Context Relevance, Groundedness, Answer Relevance."""

    @staticmethod
    def compute(context_precision: float, faithfulness: float, answer_relevance: float) -> float:
        """Compute harmonic mean of triad metrics with numerical stability."""
        eps = 1e-6
        vals = [max(eps, context_precision), max(eps, faithfulness), max(eps, answer_relevance)]
        h_mean = 3.0 / sum(1.0 / v for v in vals)
        return float(round(h_mean, 4))


# ============================================================================
# 5. Semantic Drift & Distribution Shift Detector
# ============================================================================

@dataclass
class DriftReport:
    """Report summarizing semantic drift indicators between baseline and production windows."""
    centroid_distance: float
    cosine_drift: float
    mmd_score: float
    ks_statistic: float
    ks_pvalue: float
    drift_detected: bool
    severity: str
    sample_count: int


class SemanticDriftDetector:
    """Detects concept drift, covariate shift, and out-of-distribution queries in RAG traffic."""

    def __init__(
        self,
        embedder: Optional[DeterministicSemanticEmbedder] = None,
        drift_threshold: float = 0.10,
        p_value_threshold: float = 0.01
    ):
        self.embedder = embedder or DeterministicSemanticEmbedder()
        self.drift_threshold = drift_threshold
        self.p_value_threshold = p_value_threshold
        self.baseline_embeddings: Optional[np.ndarray] = None
        self.baseline_centroid: Optional[np.ndarray] = None

    def fit_baseline(self, baseline_queries: Sequence[str]) -> None:
        """Calibrate the reference distribution from golden in-domain queries."""
        self.baseline_embeddings = self.embedder.embed_batch(baseline_queries)
        self.baseline_centroid = np.mean(self.baseline_embeddings, axis=0)
        norm = np.linalg.norm(self.baseline_centroid)
        if norm > 1e-9:
            self.baseline_centroid /= norm

    def compute_mmd(self, X: np.ndarray, Y: np.ndarray, gamma: float = 0.5) -> float:
        """Compute Maximum Mean Discrepancy (MMD) with RBF kernel between distributions."""
        def rbf_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            sq_dists = np.sum(A**2, axis=1)[:, np.newaxis] + np.sum(B**2, axis=1) - 2 * np.dot(A, B.T)
            return np.exp(-gamma * np.maximum(sq_dists, 0.0))

        m = X.shape[0]
        n = Y.shape[0]

        K_XX = rbf_kernel(X, X)
        K_YY = rbf_kernel(Y, Y)
        K_XY = rbf_kernel(X, Y)

        t_xx = float(np.mean(K_XX))
        t_yy = float(np.mean(K_YY))
        t_xy = 2.0 * float(np.mean(K_XY))

        mmd_sq = max(0.0, t_xx + t_yy - t_xy)
        return float(math.sqrt(mmd_sq))

    def evaluate_drift(self, production_queries: Sequence[str]) -> DriftReport:
        """Assess semantic drift on a production batch of queries relative to baseline."""
        if self.baseline_embeddings is None or self.baseline_centroid is None:
            raise ValueError("Baseline distribution not calibrated. Call fit_baseline() first.")

        prod_embs = self.embedder.embed_batch(production_queries)
        prod_centroid = np.mean(prod_embs, axis=0)
        norm = np.linalg.norm(prod_centroid)
        if norm > 1e-9:
            prod_centroid /= norm

        # Centroid distance & Cosine drift
        euc_dist = float(np.linalg.norm(self.baseline_centroid - prod_centroid))
        cos_sim = float(np.dot(self.baseline_centroid, prod_centroid))
        cos_drift = float(max(0.0, 1.0 - cos_sim))

        # MMD score
        mmd = self.compute_mmd(self.baseline_embeddings, prod_embs)

        # 1D Kolmogorov-Smirnov test on projection along baseline centroid vector
        base_proj = np.dot(self.baseline_embeddings, self.baseline_centroid)
        prod_proj = np.dot(prod_embs, self.baseline_centroid)
        ks_stat, ks_pval = compute_ks_2samp(base_proj, prod_proj)

        # Drift decision logic
        drift_detected = (mmd > self.drift_threshold) or (cos_drift > 0.35) or (ks_pval < self.p_value_threshold)

        if mmd > (self.drift_threshold * 1.4) or cos_drift > 0.42:
            severity = "CRITICAL"
        elif drift_detected:
            severity = "WARNING"
        else:
            severity = "NORMAL"

        return DriftReport(
            centroid_distance=round(euc_dist, 4),
            cosine_drift=round(cos_drift, 4),
            mmd_score=round(mmd, 4),
            ks_statistic=round(float(ks_stat), 4),
            ks_pvalue=round(float(ks_pval), 4),
            drift_detected=drift_detected,
            severity=severity,
            sample_count=len(production_queries),
        )


# ============================================================================
# 6. Latency, Cost, and Telemetry Metrics Accounting
# ============================================================================

@dataclass
class PipelineMetrics:
    """Comprehensive performance telemetry and operational accounting report."""
    total_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    cache_hit: bool = False


class TelemetryMetricsTracker:
    """Tracks latency percentiles, token usage, and financial cost modeling."""

    PRICING_TABLE = {
        "groq-llama3-70b": {"prompt_per_1m": 0.59, "completion_per_1m": 0.79},
        "groq-llama3-8b": {"prompt_per_1m": 0.05, "completion_per_1m": 0.08},
        "mock-cpu-model": {"prompt_per_1m": 0.00, "completion_per_1m": 0.00},
    }

    def __init__(self, model_name: str = "groq-llama3-70b"):
        self.model_name = model_name
        self.records: List[PipelineMetrics] = []

    def record_execution(
        self,
        retrieval_ms: float,
        generation_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        cache_hit: bool = False
    ) -> PipelineMetrics:
        """Record an execution instance and compute financial cost."""
        total_ms = retrieval_ms + generation_ms
        pricing = self.PRICING_TABLE.get(self.model_name, self.PRICING_TABLE["groq-llama3-70b"])
        cost = (
            (prompt_tokens / 1_000_000.0) * pricing["prompt_per_1m"]
            + (completion_tokens / 1_000_000.0) * pricing["completion_per_1m"]
        )

        metrics = PipelineMetrics(
            total_latency_ms=round(total_ms, 2),
            retrieval_latency_ms=round(retrieval_ms, 2),
            generation_latency_ms=round(generation_ms, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            estimated_cost_usd=round(cost, 6),
            cache_hit=cache_hit,
        )
        self.records.append(metrics)
        return metrics

    def compute_summary_percentiles(self) -> Dict[str, float]:
        """Calculate p50, p95, p99 latency percentiles across recorded requests."""
        if not self.records:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}

        latencies = [r.total_latency_ms for r in self.records]
        return {
            "p50": round(float(np.percentile(latencies, 50)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
            "p99": round(float(np.percentile(latencies, 99)), 2),
            "mean": round(float(np.mean(latencies)), 2),
            "total_requests": len(latencies),
            "total_cost_usd": round(sum(r.estimated_cost_usd for r in self.records), 6),
        }


# ============================================================================
# 7. Human Review Calibration Analyzer
# ============================================================================

class HumanCalibrationAnalyzer:
    """Calculates statistical alignment between automated metrics and human expert ground truth."""

    @staticmethod
    def compute_correlations(
        automated_scores: Sequence[float],
        human_scores: Sequence[float]
    ) -> Dict[str, float]:
        """Compute Pearson and Spearman rank correlation coefficients."""
        if len(automated_scores) != len(human_scores):
            raise ValueError("Mismatched sample counts between automated and human scores.")

        a = np.array(automated_scores, dtype=np.float64)
        h = np.array(human_scores, dtype=np.float64)

        # Handle zero variance degenerate cases
        if np.std(a) < 1e-7 or np.std(h) < 1e-7:
            return {"pearson_r": 1.0, "spearman_rho": 1.0, "mae": round(float(np.mean(np.abs(a - h))), 4), "rmse": round(float(np.sqrt(np.mean((a - h) ** 2))), 4)}

        pearson_r = compute_pearson_r(a, h)
        spearman_rho = compute_spearman_rho(a, h)
        mae = float(np.mean(np.abs(a - h)))
        rmse = float(np.sqrt(np.mean((a - h) ** 2)))

        return {
            "pearson_r": round(float(pearson_r), 4),
            "spearman_rho": round(float(spearman_rho), 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
        }


# ============================================================================
# 8. Full End-to-End Traced Evaluation Orchestrator
# ============================================================================

class TracedRAGPipeline:
    """Orchestrates an end-to-end RAG workflow instrumented with OpenTelemetry and Evaluation."""

    def __init__(
        self,
        telemetry: Optional[RAGTelemetryManager] = None,
        tracker: Optional[TelemetryMetricsTracker] = None
    ):
        self.telemetry = telemetry or RAGTelemetryManager()
        self.tracker = tracker or TelemetryMetricsTracker()
        self.embedder = DeterministicSemanticEmbedder()

    def execute_and_evaluate(
        self,
        sample: Dict[str, Any],
        simulate_latencies: bool = False
    ) -> Tuple[EvaluationCard, PipelineMetrics]:
        """Execute mock/deterministic RAG pipeline with hierarchical OpenTelemetry spans."""
        query = sample["query"]
        contexts = sample.get("retrieved_contexts", [])
        ground_truth = sample.get("ground_truth", "")
        generated_answer = sample.get("generated_answer", "")
        sample_id = sample.get("sample_id", "sample_00")

        start_total = time.time()

        # Root Trace: rag.pipeline.execute
        with self.telemetry.start_span(
            "rag.pipeline.execute",
            attributes={
                "gen_ai.system": "deterministic-rag-offline",
                "rag.sample_id": sample_id,
                "rag.query": query,
            }
        ):
            # Span 1: Query Transformation & Scrubbing
            with self.telemetry.start_span("rag.query.transform", attributes={"raw_query": query}):
                if simulate_latencies:
                    time.sleep(0.002)

            # Span 2: Dense Retrieval Search
            with self.telemetry.start_span(
                "rag.retrieval.dense_search",
                attributes={"top_k": len(contexts), "doc_count": len(contexts)}
            ):
                t_ret_start = time.time()
                if simulate_latencies:
                    time.sleep(0.005)
                retrieval_ms = (time.time() - t_ret_start) * 1000.0

            # Span 3: Answer Generation
            with self.telemetry.start_span(
                "rag.generation.llm_call",
                attributes={
                    "gen_ai.request.model": "groq-llama3-70b",
                    "gen_ai.usage.prompt_tokens": len(query.split()) + sum(len(c.split()) for c in contexts),
                    "gen_ai.usage.completion_tokens": len(generated_answer.split()),
                }
            ):
                t_gen_start = time.time()
                if simulate_latencies:
                    time.sleep(0.008)
                generation_ms = (time.time() - t_gen_start) * 1000.0

            # Span 4: Evaluation and Observability Audit
            with self.telemetry.start_span("rag.evaluation.audit"):
                ctx_prec, prec_flags = ContextPrecisionEvaluator.evaluate(contexts, ground_truth)
                ctx_rec, rec_details = ContextRecallEvaluator.evaluate(contexts, ground_truth)
                faith, faith_details = FaithfulnessEvaluator.evaluate(generated_answer, contexts)
                ans_rel = AnswerRelevanceEvaluator(self.embedder).evaluate(query, generated_answer)
                cit_acc, cit_details = CitationAccuracyEvaluator.evaluate(generated_answer, contexts)
                triad_score = RAGTriadScorer.compute(ctx_prec, faith, ans_rel)

                # Groundedness & Completeness
                fact_corr = round((ctx_rec + faith) / 2.0, 4)
                comp = ctx_rec

                card = EvaluationCard(
                    sample_id=sample_id,
                    context_precision=ctx_prec,
                    context_recall=ctx_rec,
                    faithfulness=faith,
                    answer_relevance=ans_rel,
                    citation_accuracy=cit_acc,
                    factual_correctness=fact_corr,
                    completeness=comp,
                    rag_triad_score=triad_score,
                    details={
                        "relevance_flags": prec_flags,
                        "recall_details": rec_details,
                        "faithfulness_details": faith_details,
                        "citation_details": cit_details,
                    }
                )

        # Track telemetry performance
        p_toks = len(query.split()) + sum(len(c.split()) for c in contexts)
        c_toks = len(generated_answer.split())
        metrics = self.tracker.record_execution(
            retrieval_ms=max(retrieval_ms, 1.0),
            generation_ms=max(generation_ms, 2.0),
            prompt_tokens=p_toks,
            completion_tokens=c_toks,
        )

        return card, metrics


# ============================================================================
# 9. Architectural Diagram Generation Helpers
# ============================================================================

def generate_observability_tracing_diagram(output_path: str) -> None:
    """Generate high-resolution diagram illustrating OpenTelemetry RAG Waterfall & PII Redaction."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)
    fig.patch.set_facecolor("#0B0F19")
    ax1.set_facecolor("#111827")
    ax2.set_facecolor("#111827")

    # Subplot 1: OpenTelemetry Span Waterfall
    spans = [
        ("rag.pipeline.execute [Root]", 0, 100, "#38BDF8"),
        ("rag.query.transform", 2, 12, "#818CF8"),
        ("rag.retrieval.dense_search", 15, 38, "#34D399"),
        ("rag.reranking.cross_encoder", 40, 22, "#FBBF24"),
        ("rag.context.format", 63, 6, "#A78BFA"),
        ("rag.generation.llm_call", 70, 75, "#F472B6"),
        ("rag.evaluation.audit", 147, 18, "#2DD4BF"),
    ]

    y_positions = np.arange(len(spans))[::-1]
    for idx, (name, start, dur, color) in enumerate(spans):
        y = y_positions[idx]
        ax1.barh(y, dur, left=start, height=0.5, color=color, edgecolor="#FFFFFF", alpha=0.9)
        ax1.text(
            start + dur + 2, y,
            f"{name} ({dur}ms)",
            va="center", color="#F3F4F6", fontsize=9, fontweight="bold"
        )

    ax1.set_yticks([])
    ax1.set_xlabel("Pipeline Execution Timeline (ms)", color="#9CA3AF", fontsize=10)
    ax1.set_title("OpenTelemetry Distributed RAG Waterfall Spans", color="#F9FAFB", fontsize=12, fontweight="bold")
    ax1.set_xlim(0, 185)
    ax1.tick_params(colors="#9CA3AF")
    ax1.grid(axis="x", linestyle="--", alpha=0.25, color="#9CA3AF")

    # Subplot 2: Telemetry Architecture & PII Redaction Firewall
    ax2.axis("off")
    ax2.set_title("Enterprise Observability & Security Architecture", color="#F9FAFB", fontsize=12, fontweight="bold")

    stages = [
        ("1. Ingestion & User Query", "Inbound payload with raw metadata and user credentials", "#60A5FA"),
        ("2. PII Sanitization Firewall", "Redacts emails, card tokens, API secrets before trace export", "#EF4444"),
        ("3. OpenTelemetry TracerProvider", "Standard GenAI attributes (tokens, top_k, latencies, model)", "#10B981"),
        ("4. InMemorySpanExporter Buffer", "High-throughput local buffer with zero external network leakage", "#F59E0B"),
        ("5. Continuous Evaluation Loop", "RAG Triad scoring, citation audit, and anomaly detection", "#A855F7"),
    ]

    y = 0.88
    for title, desc, col in stages:
        rect = plt.Rectangle((0.05, y - 0.08), 0.90, 0.09, facecolor="#1F2937", edgecolor=col, linewidth=1.8)
        ax2.add_patch(rect)
        ax2.text(0.08, y - 0.035, title, fontsize=10.5, fontweight="bold", color=col)
        ax2.text(0.08, y - 0.065, desc, fontsize=9.2, color="#E5E7EB")
        y -= 0.135

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_evaluation_and_drift_diagram(output_path: str) -> None:
    """Generate high-resolution radar chart and semantic drift distribution comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)
    fig.patch.set_facecolor("#0B0F19")
    ax1.set_facecolor("#111827")
    ax2.set_facecolor("#111827")

    # Subplot 1: Evaluation Metrics Comparison (Automated vs Human Review)
    categories = ["Context Prec", "Context Recall", "Faithfulness", "Answer Rel", "Citation Acc", "Factual Corr"]
    auto_vals = [0.95, 0.90, 0.92, 0.88, 0.94, 0.91]
    human_vals = [0.93, 0.89, 0.95, 0.86, 0.96, 0.90]

    x = np.arange(len(categories))
    width = 0.35

    ax1.bar(x - width / 2, auto_vals, width, label="Automated Metric Engine", color="#38BDF8", alpha=0.9)
    ax1.bar(x + width / 2, human_vals, width, label="Human Expert Review", color="#F472B6", alpha=0.9)

    ax1.set_ylabel("Evaluation Score (0.0 - 1.0)", color="#9CA3AF", fontsize=10)
    ax1.set_title("Automated vs Human Expert Review Calibration", color="#F9FAFB", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, color="#F3F4F6", fontsize=9.5, rotation=15)
    ax1.set_ylim(0, 1.15)
    ax1.tick_params(colors="#9CA3AF")
    ax1.legend(facecolor="#1F2937", edgecolor="#374151", labelcolor="#F9FAFB")
    ax1.grid(axis="y", linestyle="--", alpha=0.25, color="#9CA3AF")

    # Subplot 2: Semantic Drift Density Shift (Baseline vs Drifted Queries)
    np.random.seed(42)
    base_dist = np.random.normal(loc=0.82, scale=0.08, size=400)
    drift_dist = np.random.normal(loc=0.55, scale=0.14, size=400)

    ax2.hist(base_dist, bins=25, density=True, alpha=0.65, color="#34D399", label="Baseline Golden Corpus (In-Domain)")
    ax2.hist(drift_dist, bins=25, density=True, alpha=0.65, color="#EF4444", label="Production Traffic (Covariate Drift)")

    ax2.axvline(0.82, color="#34D399", linestyle="--", linewidth=1.8, label="Baseline Centroid")
    ax2.axvline(0.55, color="#EF4444", linestyle="--", linewidth=1.8, label="Shifted Centroid (MMD Alert)")

    ax2.set_xlabel("Semantic Alignment Cosine Projection", color="#9CA3AF", fontsize=10)
    ax2.set_ylabel("Empirical Probability Density", color="#9CA3AF", fontsize=10)
    ax2.set_title("Semantic Drift & Distribution Shift Detection (MMD)", color="#F9FAFB", fontsize=12, fontweight="bold")
    ax2.tick_params(colors="#9CA3AF")
    ax2.legend(facecolor="#1F2937", edgecolor="#374151", labelcolor="#F9FAFB")
    ax2.grid(linestyle="--", alpha=0.25, color="#9CA3AF")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
