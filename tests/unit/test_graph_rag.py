"""Unit tests for Track 09 — GraphRAG and Knowledge Graphs.

Tests:
1. Entity & Relationship data models, ID normalization, and metadata.
2. EntityRelationExtractor pattern matching and ontology mapping.
3. KnowledgeGraphStore CRUD, k-hop ego-network retrieval, shortest paths, PageRank, and Cypher export.
4. CommunityDetector Louvain modularity clustering and reproducibility.
5. CommunitySummarizer community reports and Map-Reduce meta-summarization.
6. GraphRAGEngine: Local Search, Global Search, and Hybrid Search execution.
7. GraphEvaluator metrics (density, modularity, coverage) and answer groundedness.
8. GraphVisualizer diagram generation.
"""

from __future__ import annotations

import os
import pytest

from common.graph_rag import (
    Entity,
    Relationship,
    Community,
    GraphSearchResult,
    EntityRelationExtractor,
    KnowledgeGraphStore,
    CommunityDetector,
    CommunitySummarizer,
    GraphRAGEngine,
    GraphEvaluator,
    GraphVisualizer,
)


@pytest.fixture
def sample_documents():
    """Deterministic, minimal test documents."""
    return [
        {
            "id": "chunk_01",
            "text": "The Transformer architecture relies on Attention. Self-Attention implements Multi-Head Attention.",
        },
        {
            "id": "chunk_02",
            "text": "Encoder consists of Self-Attention. Decoder consists of Self-Attention.",
        },
        {
            "id": "chunk_03",
            "text": "ChromaDB stores Embeddings. ChromaDB utilizes HNSW for approximate nearest neighbors.",
        },
        {
            "id": "chunk_04",
            "text": "LlamaIndex indexes Embeddings and integrates ChromaDB.",
        },
        {
            "id": "chunk_05",
            "text": "GraphRAG partitions Knowledge Graph using Louvain algorithm. GraphRAG uses Neo4j and Cypher.",
        },
    ]


def test_entity_and_relationship_models():
    """Tests basic validation and properties of Entity and Relationship models."""
    ent = Entity(
        id="transformer",
        name="Transformer",
        entity_type="ARCHITECTURE",
        description="Sequence transduction model.",
        source_chunks=["c1"],
    )
    assert ent.id == "transformer"
    assert ent.name == "Transformer"
    assert "c1" in ent.source_chunks

    rel = Relationship(
        source="transformer",
        target="attention",
        relation_type="USES",
        description="Transformer uses Attention mechanism.",
        weight=2.5,
        source_chunks=["c1", "c2"],
    )
    assert rel.source == "transformer"
    assert rel.target == "attention"
    assert rel.weight == 2.5
    assert len(rel.source_chunks) == 2


def test_entity_relation_extractor_offline():
    """Tests deterministic extraction of entities and relational triples."""
    extractor = EntityRelationExtractor()
    text = "The Transformer architecture relies on Attention. ChromaDB stores Embeddings."
    entities, rels = extractor.extract_from_text(text, chunk_id="chunk_test")

    ent_ids = {e.id for e in entities}
    assert "transformer" in ent_ids
    assert "attention" in ent_ids
    assert "chromadb" in ent_ids

    # Verify relation predicates
    rel_types = {r.relation_type for r in rels}
    assert "USES" in rel_types or "INDEXES" in rel_types


def test_knowledge_graph_store_crud():
    """Tests node/edge addition, property merging, and source chunk provenance."""
    kg = KnowledgeGraphStore()
    e1 = Entity(id="a", name="Node A", description="First node", source_chunks=["c1"])
    e2 = Entity(id="b", name="Node B", description="Second node", source_chunks=["c2"])
    kg.add_entity(e1)
    kg.add_entity(e2)

    # Re-adding entity with new chunk merges provenance
    e1_update = Entity(id="a", name="Node A", description="Updated description", source_chunks=["c3"])
    kg.add_entity(e1_update)

    stored_a = kg.get_entity("a")
    assert stored_a is not None
    assert set(stored_a.source_chunks) == {"c1", "c3"}

    # Add relationship
    rel = Relationship(source="a", target="b", relation_type="CONNECTS_TO", weight=1.0, source_chunks=["c1"])
    kg.add_relationship(rel)

    # Accumulate weight
    rel2 = Relationship(source="a", target="b", relation_type="CONNECTS_TO", weight=2.0, source_chunks=["c4"])
    kg.add_relationship(rel2)

    stored_rels = kg.all_relationships()
    assert len(stored_rels) == 1
    assert stored_rels[0].weight == 3.0
    assert set(stored_rels[0].source_chunks) == {"c1", "c4"}


def test_knowledge_graph_traversal_and_paths():
    """Tests k-hop ego-neighborhood retrieval and shortest path traversal."""
    kg = KnowledgeGraphStore()
    kg.add_relationship(Relationship(source="a", target="b", relation_type="USES"))
    kg.add_relationship(Relationship(source="b", target="c", relation_type="USES"))
    kg.add_relationship(Relationship(source="c", target="d", relation_type="USES"))
    kg.add_relationship(Relationship(source="x", target="y", relation_type="USES"))

    # 1-hop from b
    ents, rels = kg.get_k_hop_neighborhood("b", k=1)
    ent_ids = {e.id for e in ents}
    assert ent_ids == {"a", "b", "c"}

    # Shortest path
    path = kg.find_shortest_path("a", "d")
    assert path == ["a", "b", "c", "d"]

    # Disconnected components have no path
    assert kg.find_shortest_path("a", "x") is None


def test_pagerank_and_cypher_export():
    """Tests PageRank centrality computation and Neo4j Cypher generation."""
    kg = KnowledgeGraphStore()
    kg.add_relationship(Relationship(source="transformer", target="attention", relation_type="USES"))
    kg.add_relationship(Relationship(source="bert", target="transformer", relation_type="IMPLEMENTS"))

    centrality = kg.compute_centrality()
    assert "transformer" in centrality
    assert centrality["transformer"] > 0

    cypher = kg.to_cypher_statements()
    assert len(cypher) > 0
    assert any("MERGE (e:" in stmt for stmt in cypher)
    assert any("MATCH (s " in stmt for stmt in cypher)


def test_community_detector_louvain():
    """Tests Louvain community detection partitioning and modularity score Q."""
    kg = KnowledgeGraphStore()
    # Cluster 1: Attention & Transformers
    kg.add_relationship(Relationship(source="transformer", target="attention", relation_type="USES", weight=3.0))
    kg.add_relationship(Relationship(source="transformer", target="encoder", relation_type="PART_OF", weight=2.0))
    kg.add_relationship(Relationship(source="encoder", target="attention", relation_type="USES", weight=2.0))

    # Cluster 2: Vector Search & ChromaDB
    kg.add_relationship(Relationship(source="chromadb", target="embeddings", relation_type="INDEXES", weight=3.0))
    kg.add_relationship(Relationship(source="chromadb", target="hnsw", relation_type="USES", weight=2.0))
    kg.add_relationship(Relationship(source="hnsw", target="embeddings", relation_type="INDEXES", weight=2.0))

    # Weak bridge edge between clusters
    kg.add_relationship(Relationship(source="transformer", target="embeddings", relation_type="USES", weight=0.5))

    detector = CommunityDetector(resolution=1.0, random_state=42)
    clusters, q = detector.detect_communities(kg)

    assert len(clusters) == 2
    assert q > 0.0  # Strong positive modularity


def test_community_summarizer_map_reduce():
    """Tests generation of community reports and hierarchical meta-summarization."""
    kg = KnowledgeGraphStore()
    kg.add_entity(Entity(id="transformer", name="Transformer", description="Architecture"))
    kg.add_entity(Entity(id="attention", name="Attention", description="Mechanism"))
    kg.add_relationship(Relationship(
        source="transformer", target="attention", relation_type="USES", description="Transformer uses Attention"
    ))

    summarizer = CommunitySummarizer()
    comm_report = summarizer.generate_community_report(
        community_id=0, member_ids=["transformer", "attention"], kg=kg, level=0
    )

    assert comm_report.id == 0
    assert "Transformer" in comm_report.title or "Attention" in comm_report.title
    assert len(comm_report.findings) > 0
    assert comm_report.weight >= 1.0

    # Test reduce phase
    meta_summary = summarizer.reduce_meta_summary([comm_report], query="What is Attention?")
    assert "Hierarchical Knowledge Synthesis" in meta_summary
    assert "Transformer" in meta_summary


def test_graphrag_engine_end_to_end(sample_documents):
    """Tests end-to-end indexing, Local Search, Global Search, and Hybrid Search."""
    kg = KnowledgeGraphStore()
    engine = GraphRAGEngine(kg=kg)

    index_stats = engine.index_documents(sample_documents)
    assert index_stats["entities_extracted"] > 0
    assert index_stats["relationships_extracted"] > 0
    assert index_stats["communities_detected"] >= 2

    # 1. Test Local Search (entity focused)
    local_res = engine.local_search(query="How does the Transformer utilize Attention?", top_k_entities=2)
    assert isinstance(local_res, GraphSearchResult)
    assert local_res.search_mode == "local"
    assert "transformer" in local_res.grounded_entities or "attention" in local_res.grounded_entities
    assert len(local_res.source_chunks) > 0
    assert len(local_res.grounded_relationships) > 0

    # 2. Test Global Search (macro sensemaking)
    global_res = engine.global_search(query="What are the overarching themes and architectural components?")
    assert isinstance(global_res, GraphSearchResult)
    assert global_res.search_mode == "global"
    assert len(global_res.grounded_communities) > 0
    assert len(global_res.intermediate_claims) > 0
    assert "Executive Thematic Summary" in global_res.answer

    # 3. Test Hybrid Search (graph + vector)
    vector_chunks = [
        {"id": "chunk_01", "text": "The Transformer uses attention mechanisms."},
        {"id": "chunk_03", "text": "ChromaDB stores embeddings with HNSW."},
    ]
    hybrid_res = engine.hybrid_search(
        query="Explain Transformer architecture and vector storage.",
        vector_chunks=vector_chunks,
    )
    assert isinstance(hybrid_res, GraphSearchResult)
    assert hybrid_res.search_mode == "hybrid"
    assert len(hybrid_res.source_chunks) > 0
    assert "Structured Graph Evidence" in hybrid_res.answer
    assert "Unstructured Vector Context" in hybrid_res.answer


def test_graph_evaluator_metrics_and_groundedness(sample_documents):
    """Tests graph topology metrics and answer groundedness evaluation."""
    kg = KnowledgeGraphStore()
    engine = GraphRAGEngine(kg=kg)
    engine.index_documents(sample_documents)

    metrics = GraphEvaluator.evaluate_graph(kg, engine.communities)
    assert metrics["node_count"] > 0
    assert metrics["edge_count"] > 0
    assert metrics["graph_density"] >= 0.0
    assert metrics["community_count"] >= 2

    # Groundedness test
    grounded_eval = GraphEvaluator.evaluate_answer_groundedness(
        "The Transformer implements Attention and ChromaDB manages Embeddings.", kg
    )
    assert grounded_eval["is_grounded"] is True
    assert "transformer" in grounded_eval["mentioned_entities"]
    assert "chromadb" in grounded_eval["mentioned_entities"]


def test_graph_visualizer_diagrams(sample_documents, tmp_path):
    """Tests rendering and saving of knowledge graph plots and architecture diagrams."""
    kg = KnowledgeGraphStore()
    engine = GraphRAGEngine(kg=kg)
    engine.index_documents(sample_documents)

    kg_plot_path = str(tmp_path / "test_kg_plot.png")
    arch_plot_path = str(tmp_path / "test_arch_plot.png")

    out1 = GraphVisualizer.plot_knowledge_graph(kg, kg_plot_path, communities=engine.communities)
    out2 = GraphVisualizer.plot_graphrag_architecture(arch_plot_path)

    assert os.path.exists(out1)
    assert os.path.getsize(out1) > 1000
    assert os.path.exists(out2)
    assert os.path.getsize(out2) > 1000
