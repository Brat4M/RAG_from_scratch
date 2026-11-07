# Track 02 — Embeddings and Vector Search

Dense embeddings, metric geometry, NumPy retrieval, vector database indexing, and ChromaDB persistence.

---

## 🗺️ Notebook Curriculum

| Notebook | Title | Status | Core Topics |
|---|---|---|---|
| [`01_dense_embeddings_and_distance_metrics.ipynb`](./01_dense_embeddings_and_distance_metrics.ipynb) | **Dense Embeddings, Vector Spaces, and Distance Metrics** | ✅ **Validated** | Vector space geometry ($\mathbb{R}^D$), Dot Product vs. Cosine vs. Euclidean ($L_2$), $L_2$ Normalization Invariant Theorem, Vectorized Batch Retrieval, Curse of Dimensionality & Distance Concentration, Dimensional Collapse & Anisotropy (SVD & Mean Centering). |
| [`02_vector_indexing_and_ann.ipynb`](./02_vector_indexing_and_ann.ipynb) | **Vector Indexing & Approximate Nearest Neighbor (ANN) Search (HNSW & IVF)** | ✅ **Validated** | Exact $k$-NN vs. ANN, Recall@$k$ metric, Inverted File Index (IVF) with Voronoi partitioning and boundary miss recovery, Hierarchical Navigable Small World (HNSW) graphs, multi-layer skip graphs, greedy routing and beam search, empirical Pareto frontier benchmarks. |
| `03_local_vector_databases_chromadb.ipynb` | **Production Local Vector Storage with ChromaDB** | ⏳ *Planned (02.3)* | Local persistent vector storage, collections, schema metadata filtering, payload management, and query optimization. |

---

## 🧪 Verification
Execute the test suite to validate this track:
```bash
.venv312/bin/pytest tests/unit/test_vector_metrics.py
.venv312/bin/pytest tests/unit/test_ann_indexing.py
.venv312/bin/pytest tests/notebook_validation/test_notebook_structure.py
```
