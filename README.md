# RAG Engineering Academy

A rigorous, visual, first-principles curriculum for learning Retrieval-Augmented Generation (RAG), from fundamental mathematical intuitions through production-grade enterprise architectures.

---

## 🎯 Philosophy

1. **First Principles First:** Understand and implement core mechanics (vector spaces, retrieval math, chunking strategies) in pure Python before using high-level framework abstractions.
2. **Visual Learning:** Every major concept is illustrated with deterministic diagrams, charts, and architectural workflows.
3. **Reproducibility & Cost Discipline:** Default execution runs locally without external API dependencies or heavy asset downloads. Real LLM calls (via Groq API) and remote services are explicitly isolated in optional cells.
4. **Production Readiness:** Comprehensive coverage spanning evaluations, guardrails, security, tenant isolation, and graph/multimodal patterns.

---

## 📚 Curriculum Overview (17 Tracks)

The curriculum is structured across 4 progressive stages:

### Stage 1 — Foundations
- **Track 00 — Foundations and Environment:** RAG motivation, hallucinations, context windows, tokenization, high-level architecture.
- **Track 01 — Ingestion and Chunking:** PyMuPDF extraction, fixed/recursive/semantic chunking, structural and late chunking.
- **Track 02 — Embeddings and Vector Search:** Vector spaces, similarity metrics (cosine/dot/Euclidean), NumPy search, ChromaDB local persistence.
- **Track 03 — Retrieval Engineering:** BM25, dense retrieval, hybrid search, Reciprocal Rank Fusion (RRF), cross-encoder reranking, HyDE.
- **Track 04 — RAG from Scratch:** End-to-end pipeline built in pure Python without third-party agent/RAG frameworks.

### Stage 2 — Frameworks & Intelligent Systems
- **Track 05 — LangChain RAG:** LCEL, loaders, splitters, retrievers, composition patterns.
- **Track 06 — LangGraph RAG:** Cyclic graphs, state management, corrective RAG (CRAG), self-RAG, adaptive RAG.
- **Track 07 — Agentic RAG and DeepAgent:** Tool calling, query planning, iterative retrieval loops, budget and stop guardrails.
- **Track 08 — Memory Engineering:** Short-term vs. working memory, episodic retrieval, consolidation, forgetting policies.
- **Track 09 — GraphRAG and Knowledge Graphs:** Entity-relation extraction, Cypher/graph representation, hybrid graph/vector retrieval.
- **Track 10 — Multimodal RAG:** Layout analysis, images, audio, video retrieval, multimodal evidence fusion.
- **Track 11 — Google Open Knowledge Format (OKF):** Verified OKF schemas, knowledge representation, trust, and provenance.
- **Track 12 — TypeSafe Jev Decision Engine:** Deterministic decision gates, typed outcomes, confidence evaluation.
- **Track 13 — Decision Intelligence with RAG:** Combining retrieval with evidence-backed decision engines and auditability.

### Stage 3 — Enterprise, Evaluation & Deployment
- **Track 14 — Enterprise RAG:** Asynchronous ingestion, tenant isolation, document-level ACLs, caching, prompt injection defense.
- **Track 15 — Evaluation and Observability:** Retrieval precision/recall, factual groundedness, faithfulness, latency, token tracing.
- **Track 16 — Fine-Tuning versus RAG and Deployment:** Decision matrices, FastAPI serving, containerization, local release verification.

---

## 🚀 Quick Start

### 1. Environment Setup
The project uses Python 3.12 (specifically tested with `.venv312` on macOS arm64).

```bash
# Clone the repository
git clone https://github.com/Brat4M/RAG_from_scratch.git
cd RAG_from_scratch

# Activate the virtual environment
source .venv312/bin/activate

# Install base dependencies
pip install -r requirements/base.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

If you plan to run optional external model cells (e.g., using Groq):
Add your `GROQ_API_KEY` inside `.env`. Default notebook execution runs locally and does **not** require an API key.

### 3. Launch JupyterLab
```bash
jupyter lab
```

Explore the notebooks starting in `tracks/00-foundations/`.

---

## 🧪 Testing & Validation

Run the test suite:
```bash
pytest
```

---

## 📄 License & Attribution
- Repository code and educational materials are licensed under the [MIT License](LICENSE).
- PDF extraction uses [PyMuPDF](https://github.com/pymupdf/PyMuPDF) (licensed under AGPL-3.0 for open-source use) alongside [pypdf](https://github.com/py-pdf/pypdf) (Apache 2.0). See [docs/setup-guide.md](docs/setup-guide.md) for details.
