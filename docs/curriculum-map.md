# Curriculum Map — RAG Engineering Academy

This document outlines the detailed pedagogical map of all 17 tracks across the 4 stages.

---

## Stage 1 — Foundations (Tracks 00–04)

### Track 00 — Foundations and Environment
- **Core Concepts:** What is RAG and why is it needed? Parametric vs. non-parametric knowledge, hallucination mechanisms, context limits, tokenization mechanics, transformer attention intuition, end-to-end pipeline anatomy.
- **Visuals:** Anatomy of a RAG query; Context window constraints.
- **Hands-on:** Tokenizer inspection, measuring context overhead, environment smoke test.

### Track 01 — Ingestion and Chunking
- **Core Concepts:** Document ingestion lifecycle; PDF parsing with PyMuPDF; clean text extraction vs. structure preservation; chunking trade-offs (fixed-size, recursive character, sentence, semantic); parent-child hierarchies, sentence-window retrieval, table/heading extraction, late chunking.
- **Visuals:** Chunk boundary comparison (sliding window vs. semantic); Document tree parsing.
- **Hands-on:** Comparative benchmark between PyMuPDF and pypdf; Implementing recursive chunking from first principles.

### Track 02 — Embeddings and Vector Search
- **Core Concepts:** Vector spaces and high-dimensional representations; geometric intuition of similarity (Cosine similarity, Dot Product, Euclidean distance $L_2$); NumPy implementation of brute-force search; Approximate Nearest Neighbors (ANN) intuition with Hierarchical Navigable Small World (HNSW) graphs; ChromaDB collection schemas, metadata filtering, and local persistence.
- **Visuals:** Vector space projection; HNSW layer traversal.
- **Hands-on:** Building an in-memory vector index with NumPy; ChromaDB persistence and metadata query experiments.

### Track 03 — Retrieval Engineering
- **Core Concepts:** Lexical search (BM25, inverted index, term frequency / inverse document frequency) vs. dense semantic retrieval; Hybrid retrieval; Reciprocal Rank Fusion (RRF); Cross-encoder reranking vs. bi-encoders; Hypothetical Document Embeddings (HyDE); Maximum Marginal Relevance (MMR); Contextual retrieval and compression; Query expansion and routing.
- **Visuals:** Bi-encoder vs. Cross-encoder architecture; RRF rank blending.
- **Hands-on:** Implementing RRF in pure Python; Cross-encoder reranker experiments.

### Track 04 — RAG from Scratch
- **Core Concepts:** Pure Python assembly of an end-to-end RAG system without third-party agent/orchestration frameworks (no LangChain, no LlamaIndex); Custom prompt assembly with citation tags; Source document attribution; Grounding validation; Error analysis on failure modes.
- **Visuals:** Pipeline flow diagram with explicit state transitions.
- **Hands-on:** Complete pipeline execution from local raw text to grounded cited generation.

---

## Stage 2 — Frameworks & Intelligent Systems (Tracks 05–13)

### Track 05 — LangChain RAG
- Abstractions, LCEL (LangChain Expression Language), Document Loaders, Text Splitters, VectorStores, Retrievers, and Runnables.

### Track 06 — LangGraph RAG
- State graphs, nodes, edges, conditional branching, checkpointing, Corrective RAG (CRAG), Self-RAG, Adaptive RAG, human-in-the-loop gates.

### Track 07 — Agentic RAG and DeepAgent
- Tool calling, multi-step query planning, sub-query decomposition, controlled iterative loops, token budgets, stop conditions.

### Track 08 — Memory Engineering
- Short-term conversation state vs. episodic knowledge retrieval, entity memory, consolidation, forgetting policies, user isolation.

### Track 09 — GraphRAG and Knowledge Graphs
- Entities and relationships, Cypher queries, Neo4j, community detection, GraphRAG global vs. local search, hybrid graph-vector retrieval.

### Track 10 — Multimodal RAG
- Image embeddings, document layouts, charts/tables, ColPali concepts, audio and video temporal segmentation and alignment.

### Track 11 — Google Open Knowledge Format (OKF)
- Specification exploration, canonical schema representation, provenance tracking, trust, interoperability in knowledge workflows.

### Track 12 — TypeSafe Jev Decision Engine
- Strongly typed decision gates, Choice, Score, Noul semantics, deterministic confidence routing.

### Track 13 — Decision Intelligence with RAG
- Evidence retrieval combined with formal decision frameworks, provenance auditing, escalation triggers, and human oversight.

---

## Stage 3 — Enterprise, Evaluation & Deployment (Tracks 14–16)

### Track 14 — Enterprise RAG
- Asynchronous ingestion pipelines, scaling and sharding, multi-tenancy, document-level Access Control Lists (ACLs), caching, prompt-injection defense.

### Track 15 — Evaluation and Observability
- Context precision, context recall, faithfulness, groundedness, answer relevance, latency tracking, OpenTelemetry tracing, Phoenix / LangSmith integration.

### Track 16 — Fine-Tuning versus RAG and Deployment
- Trade-off matrix (RAG vs. fine-tuning vs. hybrid), embedding fine-tuning concepts, FastAPI service wrapping, containerization, and release verification.
