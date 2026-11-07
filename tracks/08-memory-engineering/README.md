# Track 08 — Memory Engineering

Mastering stateful architectures, multi-tier memory management, semantic episodic retrieval, entity knowledge graphs, and memory consolidation for production RAG systems and autonomous agents.

---

## Overview

In standard stateless RAG, each turn is treated in total isolation. In production conversational systems and agent workflows, maintaining state across interaction turns and sessions is essential for personalized, coherent, and non-redundant generation.

This track explores the design, mathematics, and operational trade-offs of multi-tier memory engineering:
- **Short-Term Context Management**: `TokenBufferMemory` (with pinned system prompts) and `SummaryBufferMemory` (sliding recent turns with progressive summary accumulation).
- **Working Memory Scratchpads**: Segregating active execution goals, constraints, and runtime state from conversational dialogue history.
- **Automated Privacy Guardrails**: `PrivacyFilter` redacting emails, phone numbers, SSNs, credit cards, and API keys before persistence.
- **Long-Term Semantic Episodic Vectors**: Modeling the **Ebbinghaus Forgetting Curve** with spaced-repetition reinforcement to balance cosine similarity, intrinsic importance, and recency decay.
- **Entity Knowledge Graphs**: Structured entity-attribute-value triples with automatic conflict detection, tombstones, and superseded revision tracking.
- **Consolidation & Pruning**: Cosine-similarity deduplication and decay-based memory eviction.
- **Diagnostics**: Multi-tenant isolation integrity, contradiction rate, and context window efficiency.

---

## Notebooks

| Notebook | Focus | Key Topics |
| :--- | :--- | :--- |
| [`01_memory_engineering.ipynb`](./01_memory_engineering.ipynb) | Multi-Tier Memory Engineering | Token buffers, summary accumulators, Ebbinghaus decay, entity graphs, PII redaction, consolidation |

---

## Architectural Diagrams

- [`track08_multi_tier_memory_architecture.png`](../../assets/diagrams/track08_multi_tier_memory_architecture.png): Multi-tier dataflow across working scratchpad, short-term buffers, episodic vectors, entity graphs, and context synthesis.
- [`track08_ebbinghaus_decay_curve.png`](../../assets/diagrams/track08_ebbinghaus_decay_curve.png): Mathematical visualization of memory retention decay and spaced repetition flattening.

---

## Verification & Testing

Run all Track 08 unit tests:
```bash
.venv312/bin/pytest tests/unit/test_memory_engineering.py
```
