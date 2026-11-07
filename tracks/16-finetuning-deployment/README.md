# Track 16 — Fine-Tuning versus RAG and Production Deployment

The capstone track of the **RAG Engineering Academy**. Evaluates the fundamental architectural trade-offs between Parametric Knowledge (LoRA/PEFT Fine-Tuning) and Non-Parametric Knowledge (RAG), and establishes hardened enterprise production serving standards.

---

## Pedagogical Objectives

1. **Parametric vs. Non-Parametric Decision Framework**:
   - Multi-dimensional trade-off matrix scoring applications across Knowledge Volatility, Hallucination Risk, Citation Strictness, Format Adherence, Access Control (ACL), Latency Budgets, and Upfront Engineering Cost.
   - Algorithmic recommendation engine selecting RAG-Only, Fine-Tuning-Only, or Hybrid architectures with customized serving stacks.

2. **Fine-Tuning Dataset Engineering & Leakage Detection**:
   - Supervised Fine-Tuning (SFT) instruction schemas (`instruction`, `input`, `output`) for extraction and refusal policies.
   - Contrastive triplet formatting (`query`, `positive`, `negative`) for embedding bi-encoder optimization.
   - Automated leakage detection catching target duplication, false negative triplets, and cross-split contamination.

3. **First-Principles Low-Rank Adaptation (LoRA)**:
   - Mathematical decomposition: $\Delta W = \frac{\alpha}{r} (B \cdot A)$ with $B=0$ initialization guaranteeing $\Delta W = 0$ on day zero.
   - Parameter efficiency accounting ($< 3.5\%$ trainable weights).
   - Zero-overhead inference weight merging ($W_{merged} = W_0 + \Delta W$) and dynamic adapter swapping.

4. **CPU Quantization & Numerical Fidelity**:
   - Symmetric and asymmetric INT8 quantization formulas mapping continuous FP32 tensors to 8-bit integer buffers.
   - Zero-point ($Z$) and scale ($S$) calibration.
   - Cosine directional fidelity ($> 0.9999$), RMSE error bounds, and $75\%$ RAM compression on standard CPU runtimes.

5. **Hybrid RAG + Fine-Tuning Synergy**:
   - Coupling non-parametric ChromaDB vector search with fine-tuned structured synthesis.
   - Grounded source citations (`[Citation: Source]`) and calibrated refusal gates against hallucinations on out-of-domain queries.

6. **Production Serving Contract (FastAPI & ASGI)**:
   - Microservice architecture with Kubernetes Liveness/Readiness probes (`/health`).
   - Query endpoint (`/v1/query`) with structured citation schemas and `X-Process-Time-Ms` streaming latency tracking.
   - Ingestion endpoint (`/v1/ingest`) for real-time index updates without service reboots.
   - Operational telemetry metrics (`/v1/metrics`) tracking percentiles ($p_{50}, p_{95}, p_{99}$), request volumes, and refusal rates.

7. **Containerization & Deployment Readiness Audit**:
   - Automated 8-pillar enterprise compliance audit (non-root execution, secret isolation, timeout bounds, persistent volumes, telemetry, ACL filtering).
   - Hardened multi-stage Dockerfile (`python:3.12-slim-bookworm`) with unprivileged service account (`UID 10001`).
   - Production Docker Compose topology with resource limits and health checks.

---

## Notebooks

- [`01_finetuning_vs_rag_and_deployment.ipynb`](01_finetuning_vs_rag_and_deployment.ipynb): Comprehensive end-to-end interactive tutorial, mathematical derivations, live ASGI endpoint testing, and visual architectural topology diagrams.

---

## Architectural Assets

- `assets/diagrams/track16_finetuning_vs_rag_decision_matrix.png`: Decision matrix quadrant and multi-dimensional comparison radar.
- `assets/diagrams/track16_production_deployment_architecture.png`: Enterprise serving topology from Client Ingress to Quantized LoRA heads, ChromaDB, and OpenTelemetry.
- `fixtures/finetuning/sample_sft_dataset.jsonl`: Curated instruction tuning dataset fixture.
- `fixtures/finetuning/sample_contrastive_triplets.jsonl`: Contrastive embedding fine-tuning triplet fixture.
- `fixtures/deployment/production_rag_config.json`: Hardened enterprise deployment configuration specification.
