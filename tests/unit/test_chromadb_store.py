"""Unit tests for production local vector storage using ChromaDB.

Validates:
1. Ephemeral vs. Persistent client lifecycle.
2. Complete Document CRUD operations (add, get, update, upsert, delete).
3. Compound metadata filtering ($and, $or, $in, $gte, $eq).
4. Distance metric spaces ('cosine', 'l2', 'ip') and distance semantics.
5. Custom EmbeddingFunction implementation and integration.
6. Direct pre-computed vector ingestion.
"""

import shutil
import tempfile
import zlib
import numpy as np
import pytest
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


class DeterministicHashEmbeddingFunction(EmbeddingFunction[Documents]):
    """Lightweight, deterministic, zero-dependency token hash projection for offline CPU testing."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            vec = np.zeros(self.dim, dtype=np.float32)
            for word in text.lower().replace(".", "").replace(",", "").split():
                idx = zlib.crc32(word.encode("utf-8")) % self.dim
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec.tolist())
        return embeddings


def test_ephemeral_client_crud_lifecycle():
    """Validates complete CRUD cycle on an in-memory ephemeral client."""
    client = chromadb.Client()
    col = client.create_collection("test_ephemeral_crud")
    assert col.count() == 0

    # 1. CREATE (add)
    col.add(
        ids=["doc_1", "doc_2"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        documents=["Document one content", "Document two content"],
        metadatas=[{"topic": "nlp", "version": 1}, {"topic": "cv", "version": 1}],
    )
    assert col.count() == 2

    # In ChromaDB 1.x (Rust engine), adding an existing ID does not overwrite the existing record
    col.add(
        ids=["doc_1"],
        embeddings=[[1.0, 0.0, 0.0]],
        documents=["Attempted duplicate overwrite"],
        metadatas=[{"topic": "nlp", "version": 999}],
    )
    # Verify original record was preserved (idempotent / non-destructive add)
    orig_check = col.get(ids=["doc_1"])
    assert orig_check["documents"] == ["Document one content"]
    assert orig_check["metadatas"][0]["version"] == 1

    # 2. READ (get)
    record = col.get(ids=["doc_1"], include=["documents", "metadatas"])
    assert record["ids"] == ["doc_1"]
    assert record["documents"] == ["Document one content"]
    assert record["metadatas"] == [{"topic": "nlp", "version": 1}]

    # 3. UPDATE
    col.update(
        ids=["doc_1"],
        embeddings=[[0.9, 0.1, 0.0]],
        documents=["Updated document one"],
        metadatas=[{"topic": "nlp", "version": 2}],
    )
    updated = col.get(ids=["doc_1"], include=["documents", "metadatas"])
    assert updated["documents"] == ["Updated document one"]
    assert updated["metadatas"] == [{"topic": "nlp", "version": 2}]

    # 4. UPSERT (modify existing, insert new)
    col.upsert(
        ids=["doc_1", "doc_3"],
        embeddings=[[0.95, 0.05, 0.0], [0.0, 0.0, 1.0]],
        documents=["Upserted document one", "Document three content"],
        metadatas=[{"topic": "nlp", "version": 3}, {"topic": "audio", "version": 1}],
    )
    assert col.count() == 3
    upserted = col.get(ids=["doc_1", "doc_3"], include=["metadatas"])
    assert upserted["metadatas"][0]["version"] == 3

    # 5. DELETE
    col.delete(ids=["doc_2"])
    assert col.count() == 2
    remaining = col.get()["ids"]
    assert "doc_2" not in remaining
    assert "doc_1" in remaining and "doc_3" in remaining


def test_persistent_client_lifecycle():
    """Validates on-disk persistence and reloading across separate client instances."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Client instance 1: create collection and write documents
        client1 = chromadb.PersistentClient(path=temp_dir)
        col1 = client1.create_collection("persisted_kb", metadata={"hnsw:space": "cosine"})
        col1.add(
            ids=["p_001", "p_002"],
            embeddings=[[0.6, 0.8], [0.8, 0.6]],
            documents=["Knowledge base alpha", "Knowledge base beta"],
            metadatas=[{"tenant": "enterprise_a"}, {"tenant": "enterprise_b"}],
        )
        assert col1.count() == 2
        del col1
        del client1

        # Client instance 2: reopen same directory and verify intact data
        client2 = chromadb.PersistentClient(path=temp_dir)
        collections = [c.name for c in client2.list_collections()]
        assert "persisted_kb" in collections

        col2 = client2.get_collection("persisted_kb")
        assert col2.count() == 2
        retrieved = col2.get(ids=["p_001"], include=["documents", "metadatas"])
        assert retrieved["documents"] == ["Knowledge base alpha"]
        assert retrieved["metadatas"] == [{"tenant": "enterprise_a"}]

        # Query persistence
        res = col2.query(query_embeddings=[[0.6, 0.8]], n_results=1)
        assert res["ids"][0][0] == "p_001"
        assert res["distances"][0][0] == pytest.approx(0.0, abs=1e-5)
    finally:
        shutil.rmtree(temp_dir)


def test_metadata_filtering_operators():
    """Validates compound filtering using $and, $or, $in, $gte, and $eq."""
    client = chromadb.Client()
    col = client.create_collection("filtered_collection")
    col.add(
        ids=["doc_a", "doc_b", "doc_c", "doc_d"],
        embeddings=[[1.0, 0.0], [0.8, 0.2], [0.2, 0.8], [0.0, 1.0]],
        documents=["Doc A", "Doc B", "Doc C", "Doc D"],
        metadatas=[
            {"source": "wiki", "year": 2024, "public": True, "score": 9.5},
            {"source": "manual", "year": 2023, "public": True, "score": 8.0},
            {"source": "arxiv", "year": 2022, "public": False, "score": 7.0},
            {"source": "manual", "year": 2024, "public": False, "score": 6.5},
        ],
    )

    # 1. $and with $in and $gte
    res_and = col.query(
        query_embeddings=[[1.0, 0.0]],
        n_results=10,
        where={
            "$and": [
                {"source": {"$in": ["wiki", "manual"]}},
                {"year": {"$gte": 2023}},
            ]
        },
    )
    found_ids = set(res_and["ids"][0])
    assert found_ids == {"doc_a", "doc_b", "doc_d"}

    # 2. $or filter
    res_or = col.query(
        query_embeddings=[[1.0, 0.0]],
        n_results=10,
        where={
            "$or": [
                {"public": {"$eq": True}},
                {"score": {"$lt": 7.0}},
            ]
        },
    )
    assert set(res_or["ids"][0]) == {"doc_a", "doc_b", "doc_d"}

    # 3. Document text search filtering ($contains)
    res_doc = col.query(
        query_embeddings=[[1.0, 0.0]],
        n_results=10,
        where_document={"$contains": "Doc C"},
    )
    assert res_doc["ids"][0] == ["doc_c"]


def test_distance_metrics_spaces():
    """Validates cosine, l2, and ip metric behavior and mathematical properties."""
    # Orthogonal vectors in 2D
    v_x = [1.0, 0.0]
    v_y = [0.0, 1.0]

    # 1. Cosine: distance = 1 - cosine_similarity
    client = chromadb.Client()
    col_cosine = client.create_collection("col_cos", metadata={"hnsw:space": "cosine"})
    col_cosine.add(ids=["vx", "vy"], embeddings=[v_x, v_y])

    res_cos = col_cosine.query(query_embeddings=[v_x], n_results=2)
    # Distance to itself should be 0.0; to orthogonal vector should be 1.0
    assert res_cos["ids"][0] == ["vx", "vy"]
    assert res_cos["distances"][0][0] == pytest.approx(0.0, abs=1e-5)
    assert res_cos["distances"][0][1] == pytest.approx(1.0, abs=1e-5)

    # 2. L2: squared Euclidean distance ||u - v||^2
    col_l2 = client.create_collection("col_l2", metadata={"hnsw:space": "l2"})
    col_l2.add(ids=["vx", "vy"], embeddings=[v_x, v_y])

    res_l2 = col_l2.query(query_embeddings=[v_x], n_results=2)
    # ||vx - vx||^2 = 0; ||vx - vy||^2 = (1-0)^2 + (0-1)^2 = 2.0
    assert res_l2["ids"][0] == ["vx", "vy"]
    assert res_l2["distances"][0][0] == pytest.approx(0.0, abs=1e-5)
    assert res_l2["distances"][0][1] == pytest.approx(2.0, abs=1e-5)

    # 3. IP: 1.0 - inner_product
    col_ip = client.create_collection("col_ip", metadata={"hnsw:space": "ip"})
    col_ip.add(ids=["vx", "vy"], embeddings=[v_x, v_y])

    res_ip = col_ip.query(query_embeddings=[v_x], n_results=2)
    # 1 - (vx . vx) = 1 - 1 = 0; 1 - (vx . vy) = 1 - 0 = 1.0
    assert res_ip["ids"][0] == ["vx", "vy"]
    assert res_ip["distances"][0][0] == pytest.approx(0.0, abs=1e-5)
    assert res_ip["distances"][0][1] == pytest.approx(1.0, abs=1e-5)


def test_custom_embedding_function_integration():
    """Validates registration and transparent query-time embedding using custom EmbeddingFunction."""
    ef = DeterministicHashEmbeddingFunction(dim=8)
    client = chromadb.Client()
    col = client.create_collection("custom_ef_col", embedding_function=ef)

    col.add(
        ids=["art_ai", "art_space", "art_cooking"],
        documents=[
            "Deep learning neural networks and transformers",
            "Orbital rockets and interplanetary space exploration",
            "Culinary recipes baking bread and preparing meals",
        ],
        metadatas=[{"cat": "tech"}, {"cat": "science"}, {"cat": "lifestyle"}],
    )

    # Query with plain text - ChromaDB uses ef transparently
    res = col.query(query_texts=["neural networks deep learning"], n_results=1)
    assert res["ids"][0][0] == "art_ai"
    assert res["documents"][0][0] == "Deep learning neural networks and transformers"


def test_precomputed_embeddings_ingestion():
    """Validates that pre-computed NumPy vector embeddings bypass text embedding functions."""
    client = chromadb.Client()
    col = client.create_collection("precomputed_col")

    # Generate 4-dimensional pre-computed embeddings
    raw_vecs = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)

    col.add(
        ids=["vec_0", "vec_1", "vec_2"],
        embeddings=raw_vecs.tolist(),
        documents=["Concept zero", "Concept one", "Concept two"],
    )

    query_vec = [0.9, 0.1, 0.0, 0.0]
    res = col.query(query_embeddings=[query_vec], n_results=2)
    assert res["ids"][0][0] == "vec_0"
