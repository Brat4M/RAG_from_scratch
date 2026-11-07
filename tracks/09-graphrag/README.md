# Track 09 — GraphRAG and Knowledge Graphs

Mastering entity-relation extraction, property graphs, Louvain community clustering, and hierarchical Map-Reduce retrieval for production RAG systems.

---

## Overview

In standard Retrieval-Augmented Generation (**Vector RAG**), text passages are isolated in dense vector spaces. While effective for localized semantic matching, Vector RAG struggles with:
1. **Multi-Hop Relational Traversal**: Inability to follow connected causal and structural chains (`A -> B -> C`) spanning multiple documents.
2. **Global Corpus Sensemaking**: Inability to answer overarching exploratory questions (*"What are the central themes across all documents?"*) because no single chunk contains the complete answer.

**GraphRAG** resolves these limitations by synthesizing a structured **Property Knowledge Graph**, clustering entities into hierarchical **Communities** via the Louvain modularity algorithm, and providing two search paradigms:
- **Local Search (Entity-Centric)**: Explores $k$-hop ego-networks around focal entities, extracting incident relational triples and source chunk provenance.
- **Global Search (Hierarchical Map-Reduce)**: Synthesizes high-level answers across community partitions, ranking candidate points and generating an executive summary.
- **Hybrid Search**: Fuses dense vector similarity with structured knowledge graph paths.

---

## Notebooks

| Notebook | Focus | Key Topics |
| :--- | :--- | :--- |
| [`01_graphrag_knowledge_graphs.ipynb`](./01_graphrag_knowledge_graphs.ipynb) | GraphRAG & Knowledge Graphs | Entity-relation extraction, property graphs, Louvain modularity ($Q$), community reports, Local/Global/Hybrid search, Neo4j Cypher export |

---

## Architectural Diagrams

- [`track09_graphrag_architecture.png`](../../assets/diagrams/track09_graphrag_architecture.png): Detailed taxonomy comparing Naive Vector RAG, Local GraphRAG (Ego-networks), and Global GraphRAG (Map-Reduce).
- [`track09_community_graph.png`](../../assets/diagrams/track09_community_graph.png): Visual NetworkX spring-layout rendering of the multi-domain knowledge graph with Louvain community color-coding.

---

## Verification & Testing

Run all Track 09 unit tests:
```bash
.venv312/bin/pytest tests/unit/test_graph_rag.py
```
