# RAG & Information Retrieval Glossary

A reference glossary of terms, algorithms, and metrics used throughout the academy.

---

### Approximate Nearest Neighbors (ANN)
Algorithmic approaches (such as HNSW, ScaNN, or IVF) designed to find vector points nearest to a query vector in sub-linear time, trading guaranteed mathematical exactness for orders-of-magnitude speedups.

### Bi-Encoder
An architecture where the query and document are embedded independently into dense vectors using two separate forward passes (or the same Siamese network). Extremely fast for retrieval because document embeddings can be precomputed offline.

### BM25 (Best Matching 25)
A non-linear probabilistic bag-of-words retrieval function that ranks documents based on the matching query terms, incorporating term frequency saturation and document length normalization.

### ChromaDB
An open-source, lightweight embedding database designed for local developer workflows and production vector persistence with metadata filtering support.

### Chunking
The process of dividing long documents into smaller segments or passages suitable for embedding models and context windows. Strategies include character-based, recursive, sentence-boundary, semantic, and hierarchical parent-child chunking.

### Cross-Encoder
A transformer model where the query and document are concatenated and passed together through all attention layers. Computes cross-attention across all tokens in both texts, yielding higher relevance accuracy than bi-encoders at higher computational cost.

### Faithfulness
An evaluation metric measuring whether the generated answer is strictly supported by and grounded in the retrieved context, without introducing hallucinations.

### Hierarchical Navigable Small World (HNSW)
A multi-layer graph data structure for Approximate Nearest Neighbor search that achieves logarithmic search scaling $\mathcal{O}(\log N)$ by building skip-list-like graph layers.

### Hypothetical Document Embeddings (HyDE)
A query transformation technique where an LLM first generates a hypothetical answer to a question, and that generated passage is embedded to search for real documents matching its semantic profile.

### Reciprocal Rank Fusion (RRF)
An rank-aggregation method that combines rankings from multiple distinct retrieval algorithms (e.g., dense vector search and sparse BM25) without requiring score normalization:
$$RRF\_Score(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

### Semantic Chunking
A chunking strategy that determines chunk boundaries dynamically by calculating cosine distance between adjacent sentence embeddings and breaking when similarity drops below a threshold.
