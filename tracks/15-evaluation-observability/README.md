# Track 15 — Evaluation & Observability: OpenTelemetry Tracing, Automated RAG Metrics, Semantic Drift & Continuous Telemetry

Production-grade observability and evaluation architectures for enterprise RAG platforms: OpenTelemetry distributed lifecycle spans, PII sanitization firewalls, automated evaluation metrics (Context Precision, Recall, Faithfulness, Relevance, Citation Accuracy), TruLens RAG Triad composite scoring, semantic drift detection via Maximum Mean Discrepancy (MMD), and human review calibration.

---

## Overview

Unlike deterministic software systems that fail with explicit exceptions and error codes, Retrieval-Augmented Generation (RAG) platforms are prone to silent degradation:
- **Retrieval Drift**: Retrieving irrelevant documents or suffering from covariate shift as user vocabulary evolves.
- **Hallucinations**: Generative LLMs fabricating plausible-sounding but unsupported claims.
- **Citation Misattribution**: Quoting factual data while citing incorrect reference documents.
- **Latency & Token Spikes**: Hidden performance bottlenecks in reranking or LLM generation inflating infrastructure bills.

Track 15 establishes an end-to-end observability and evaluation framework that operates 100% offline on CPU by default, while supporting seamless export to enterprise collectors (OpenTelemetry, Arize Phoenix, LangSmith, and Groq).

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                         Inbound User Query & Request                        │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                     PII Sanitization & Redaction Firewall                   │
 │           (Scrubs Emails, Cards, IP Addresses, and API Bearer Tokens)       │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                  OpenTelemetry Distributed Tracing Manager                  │
 │  Root Trace: rag.pipeline.execute                                           │
 │   ├─ Child: rag.query.transform                                             │
 │   ├─ Child: rag.retrieval.dense_search (Captures top_k, doc_count)          │
 │   ├─ Child: rag.retrieval.rerank (Captures candidate pool, model)           │
 │   └─ Child: rag.generation.llm_call (Captures prompt/completion tokens)     │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                    First-Principles Automated Evaluation                    │
 │  • Context Precision: Rank-weighted MAP@K of retrieved chunks               │
 │  • Context Recall: Ground-truth proposition coverage in corpus              │
 │  • Faithfulness: Atomic proposition verification & hallucination penalty    │
 │  • Answer Relevance: Query-answer semantic cosine alignment                 │
 │  • Citation Accuracy: Exact inline [Doc N] citation grounding check         │
 │  • TruLens RAG Triad: Harmonic composite penalizing single-pillar failures  │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                   Semantic Drift & Distribution Shift Detector              │
 │  • Centroid Cosine Drift (Shift in average query vector direction)          │
 │  • Maximum Mean Discrepancy (MMD with RBF / Cosine kernel in RKHS)          │
 │  • 2-Sample Kolmogorov-Smirnov Cumulative Distribution Test                 │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                    Operational Telemetry & Cost Accounting                  │
 │   (p50, p95, p99 Latency Percentiles + Groq Llama 3 Pricing Tier Modeling)  │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## Notebook Catalog

| Notebook | Focus Areas | Key Implementations |
|---|---|---|
| [`01_evaluation_and_observability.ipynb`](./01_evaluation_and_observability.ipynb) | OpenTelemetry spans, PII redaction, automated evaluation metrics, TruLens triad, MMD semantic drift, latency/cost accounting, and human calibration | `RAGTelemetryManager`, `PIIRedactor`, `ContextPrecisionEvaluator`, `ContextRecallEvaluator`, `FaithfulnessEvaluator`, `AnswerRelevanceEvaluator`, `CitationAccuracyEvaluator`, `RAGTriadScorer`, `SemanticDriftDetector`, `TelemetryMetricsTracker`, `HumanCalibrationAnalyzer` |

---

## Architectural Diagrams

- **Distributed Tracing & PII Redaction**: [`assets/diagrams/track15_rag_observability_tracing.png`](../../assets/diagrams/track15_rag_observability_tracing.png)
- **Evaluation Radar & Semantic Drift**: [`assets/diagrams/track15_rag_evaluation_triad_and_drift.png`](../../assets/diagrams/track15_rag_evaluation_triad_and_drift.png)

---

## Test Verification

Run the dedicated Track 15 test suite:

```bash
.venv312/bin/pytest tests/unit/test_observability_evaluation.py -v
```

Run the entire repository test suite (188+ tests):

```bash
.venv312/bin/pytest tests/unit/ -q
```
