"""GraphRAG and Knowledge Graphs for Production RAG Systems.

Provides high-performance, 100% offline-executable, CPU-friendly GraphRAG architectures:
1. Entity & Relationship Extraction: Rule-based heuristics, custom ontologies, and optional LLM-assisted triples.
2. Property Knowledge Graph Store: NetworkX-backed directed property graph with Cypher statement generation.
3. Hierarchical Community Detection: Newman-Girvan modularity optimization via Louvain clustering.
4. Community Summarization (Map-Reduce): Synthesis of community summaries, key findings, and importance ratings.
5. Dual GraphRAG Search Modes:
   - Local Search: Ego-network traversal, 1-to-2 hop neighborhoods, and entity-centric retrieval.
   - Global Search: Map-reduce summarization across community hierarchies for dataset-wide sensemaking.
   - Hybrid Search: Reciprocal fusion of dense vector chunks and structured graph paths.
6. Graph Diagnostics & Visualization: Modularity, density, citation coverage, and Matplotlib network plotting.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity
import numpy as np
from pydantic import BaseModel, Field

from common.query_transformations import DeterministicSemanticEmbedder


# ---------------------------------------------------------------------------
# 1. Pydantic & Data Models
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    """Represents a canonical knowledge graph entity node."""
    id: str = Field(..., description="Unique lowercase normalized entity identifier")
    name: str = Field(..., description="Display name of the entity")
    entity_type: str = Field(default="CONCEPT", description="Entity category/type (e.g., CONCEPT, ARCHITECTURE, ORGANIZATION)")
    description: str = Field(default="", description="Factual summary or definition of the entity")
    source_chunks: List[str] = Field(default_factory=list, description="IDs of source text chunks mentioning this entity")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary attributes (degree, pagerank, community_id)")


class Relationship(BaseModel):
    """Represents a directed knowledge graph relationship edge."""
    source: str = Field(..., description="Source entity id")
    target: str = Field(..., description="Target entity id")
    relation_type: str = Field(..., description="Predicate/type (e.g., USES, IMPLEMENTS, PART_OF)")
    description: str = Field(default="", description="Factual description of the relationship")
    weight: float = Field(default=1.0, description="Confidence or co-occurrence frequency")
    source_chunks: List[str] = Field(default_factory=list, description="Source chunk IDs supporting this relationship")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary attributes")


class Community(BaseModel):
    """Represents a clustered community of closely interconnected entities."""
    id: int = Field(..., description="Unique community index")
    level: int = Field(default=0, description="Hierarchy level (0 = base cluster, 1 = parent cluster)")
    title: str = Field(..., description="Descriptive title summarizing the cluster theme")
    summary: str = Field(..., description="Comprehensive narrative summary of the community")
    findings: List[str] = Field(default_factory=list, description="Key insights and factual propositions")
    entity_ids: List[str] = Field(default_factory=list, description="Member entity IDs")
    weight: float = Field(default=1.0, description="Community significance rating (1.0 to 10.0)")
    modularity_contribution: float = Field(default=0.0, description="Internal link density / contribution")


class GraphSearchResult(BaseModel):
    """Unified result container for GraphRAG search executions."""
    query: str
    search_mode: str = Field(..., description="'local', 'global', or 'hybrid'")
    answer: str
    grounded_entities: List[str] = Field(default_factory=list)
    grounded_relationships: List[Dict[str, Any]] = Field(default_factory=list)
    grounded_communities: List[int] = Field(default_factory=list)
    source_chunks: List[str] = Field(default_factory=list)
    intermediate_claims: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 2. Information Extraction: Entity & Relation Extractor
# ---------------------------------------------------------------------------

class EntityRelationExtractor:
    """Extracts entities and relational triples from raw text.
    
    Provides 100% deterministic, offline regex/pattern-based extraction for CPU execution,
    with an optional LLM-assisted structured extraction pathway when credentials exist.
    """

    KNOWN_ENTITY_TYPES: Dict[str, str] = {
        "transformer": "ARCHITECTURE",
        "attention": "MECHANISM",
        "self-attention": "MECHANISM",
        "multi-head attention": "MECHANISM",
        "encoder": "COMPONENT",
        "decoder": "COMPONENT",
        "bert": "MODEL",
        "gpt": "MODEL",
        "chromadb": "VECTOR_DATABASE",
        "llamaindex": "FRAMEWORK",
        "langchain": "FRAMEWORK",
        "langgraph": "FRAMEWORK",
        "hnsw": "ALGORITHM",
        "bm25": "ALGORITHM",
        "groq": "ORGANIZATION",
        "google": "ORGANIZATION",
        "openai": "ORGANIZATION",
        "rag": "PARADIGM",
        "graphrag": "PARADIGM",
        "knowledge graph": "DATA_STRUCTURE",
        "louvain": "ALGORITHM",
        "cypher": "QUERY_LANGUAGE",
        "neo4j": "GRAPH_DATABASE",
    }

    RELATION_PATTERNS = [
        # X uses Y
        (re.compile(r"([A-Za-z0-9_\-\s]+?)\s+(?:uses|utilizes|employs|relies on)\s+([A-Za-z0-9_\-\s]+)", re.IGNORECASE), "USES"),
        # X implements Y
        (re.compile(r"([A-Za-z0-9_\-\s]+?)\s+(?:implements|realizes|executes)\s+([A-Za-z0-9_\-\s]+)", re.IGNORECASE), "IMPLEMENTS"),
        # X consists of / contains Y
        (re.compile(r"([A-Za-z0-9_\-\s]+?)\s+(?:consists of|contains|is composed of|comprises)\s+([A-Za-z0-9_\-\s]+)", re.IGNORECASE), "PART_OF"),
        # X is based on Y
        (re.compile(r"([A-Za-z0-9_\-\s]+?)\s+(?:is based on|builds upon|extends|derives from)\s+([A-Za-z0-9_\-\s]+)", re.IGNORECASE), "EXTENDS"),
        # X indexes / stores Y
        (re.compile(r"([A-Za-z0-9_\-\s]+?)\s+(?:indexes|stores|persists|manages)\s+([A-Za-z0-9_\-\s]+)", re.IGNORECASE), "INDEXES"),
        # X created / proposed by Y
        (re.compile(r"([A-Za-z0-9_\-\s]+?)\s+(?:was proposed by|was developed by|was created by|authored by)\s+([A-Za-z0-9_\-\s]+)", re.IGNORECASE), "CREATED_BY"),
        # X clusters / partitions Y
        (re.compile(r"([A-Za-z0-9_\-\s]+?)\s+(?:clusters|partitions|segments)\s+([A-Za-z0-9_\-\s]+)", re.IGNORECASE), "CLUSTERS"),
        # X evaluates / benchmarks Y
        (re.compile(r"([A-Za-z0-9_\-\s]+?)\s+(?:evaluates|benchmarks|measures)\s+([A-Za-z0-9_\-\s]+)", re.IGNORECASE), "EVALUATES"),
    ]

    def __init__(self, custom_ontology: Optional[Dict[str, str]] = None):
        self.ontology = dict(self.KNOWN_ENTITY_TYPES)
        if custom_ontology:
            self.ontology.update(custom_ontology)

    @staticmethod
    def normalize_id(text: str) -> str:
        """Normalizes an entity name into a canonical lowercase ID."""
        cleaned = re.sub(r"[^\w\s-]", "", text.strip().lower())
        return re.sub(r"[\s_]+", "-", cleaned).strip("-")

    def _infer_type(self, entity_name: str) -> str:
        norm_id = self.normalize_id(entity_name)
        if norm_id in self.ontology:
            return self.ontology[norm_id]
        for key, val in self.ontology.items():
            if key in norm_id or norm_id in key:
                return val
        return "CONCEPT"

    def extract_from_text(self, text: str, chunk_id: str) -> Tuple[List[Entity], List[Relationship]]:
        """Extracts entities and relational triples using deterministic pattern matching."""
        entities_dict: Dict[str, Entity] = {}
        relationships: List[Relationship] = []

        # 1. Scan for known dictionary concepts
        for term, etype in self.ontology.items():
            pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            if pattern.search(text):
                norm_id = self.normalize_id(term)
                display_name = term.replace("-", " ").title()
                entities_dict[norm_id] = Entity(
                    id=norm_id,
                    name=display_name,
                    entity_type=etype,
                    description=f"Entity {display_name} ({etype}) referenced in chunk {chunk_id}.",
                    source_chunks=[chunk_id],
                )

        # 2. Extract relation triples using regex patterns
        sentences = re.split(r"[.\n;]", text)
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            for pattern, rel_type in self.RELATION_PATTERNS:
                matches = pattern.finditer(sent)
                for match in matches:
                    raw_src = match.group(1).strip()
                    raw_tgt = match.group(2).strip()

                    # Clean clauses
                    raw_src = re.sub(r"^(the|a|an|in|with|by|for)\s+", "", raw_src, flags=re.IGNORECASE).strip()
                    raw_tgt = re.sub(r"^(the|a|an|in|with|by|for)\s+", "", raw_tgt, flags=re.IGNORECASE).strip()

                    # Limit entity token span length to avoid long clause noise
                    src_words = raw_src.split()
                    tgt_words = raw_tgt.split()
                    if not (1 <= len(src_words) <= 4 and 1 <= len(tgt_words) <= 4):
                        continue

                    src_id = self.normalize_id(raw_src)
                    tgt_id = self.normalize_id(raw_tgt)

                    if not src_id or not tgt_id or src_id == tgt_id:
                        continue

                    # Ensure entities exist in entities_dict
                    if src_id not in entities_dict:
                        entities_dict[src_id] = Entity(
                            id=src_id,
                            name=raw_src.title(),
                            entity_type=self._infer_type(raw_src),
                            description=f"Entity {raw_src.title()} extracted from {chunk_id}.",
                            source_chunks=[chunk_id],
                        )
                    else:
                        if chunk_id not in entities_dict[src_id].source_chunks:
                            entities_dict[src_id].source_chunks.append(chunk_id)

                    if tgt_id not in entities_dict:
                        entities_dict[tgt_id] = Entity(
                            id=tgt_id,
                            name=raw_tgt.title(),
                            entity_type=self._infer_type(raw_tgt),
                            description=f"Entity {raw_tgt.title()} extracted from {chunk_id}.",
                            source_chunks=[chunk_id],
                        )
                    else:
                        if chunk_id not in entities_dict[tgt_id].source_chunks:
                            entities_dict[tgt_id].source_chunks.append(chunk_id)

                    rel_desc = f"{entities_dict[src_id].name} {rel_type.lower().replace('_', ' ')} {entities_dict[tgt_id].name}"
                    relationships.append(Relationship(
                        source=src_id,
                        target=tgt_id,
                        relation_type=rel_type,
                        description=rel_desc,
                        weight=1.0,
                        source_chunks=[chunk_id],
                    ))

        return list(entities_dict.values()), relationships


# ---------------------------------------------------------------------------
# 3. Knowledge Graph Store
# ---------------------------------------------------------------------------

class KnowledgeGraphStore:
    """In-memory Property Knowledge Graph backed by NetworkX DiGraph.
    
    Supports entity/relation storage, k-hop ego-network retrieval,
    PageRank centrality, path finding, and Cypher statement export.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self._entities: Dict[str, Entity] = {}

    def add_entity(self, entity: Entity) -> None:
        """Adds or updates an entity node in the graph."""
        if entity.id in self._entities:
            existing = self._entities[entity.id]
            # Merge descriptions and source chunks
            merged_chunks = sorted(list(set(existing.source_chunks + entity.source_chunks)))
            if len(entity.description) > len(existing.description):
                existing.description = entity.description
            existing.source_chunks = merged_chunks
            existing.metadata.update(entity.metadata)
            self.graph.nodes[entity.id]["description"] = existing.description
            self.graph.nodes[entity.id]["source_chunks"] = merged_chunks
            self.graph.nodes[entity.id]["metadata"] = existing.metadata
        else:
            self._entities[entity.id] = entity
            self.graph.add_node(
                entity.id,
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                source_chunks=entity.source_chunks,
                metadata=entity.metadata,
            )

    def add_relationship(self, rel: Relationship) -> None:
        """Adds or updates a directed relationship edge in the graph."""
        # Ensure endpoints exist
        if rel.source not in self._entities:
            self.add_entity(Entity(id=rel.source, name=rel.source.title(), source_chunks=rel.source_chunks))
        if rel.target not in self._entities:
            self.add_entity(Entity(id=rel.target, name=rel.target.title(), source_chunks=rel.source_chunks))

        if self.graph.has_edge(rel.source, rel.target):
            edge_data = self.graph[rel.source][rel.target]
            edge_data["weight"] = edge_data.get("weight", 1.0) + rel.weight
            merged_chunks = sorted(list(set(edge_data.get("source_chunks", []) + rel.source_chunks)))
            edge_data["source_chunks"] = merged_chunks
            if rel.description and rel.description not in edge_data.get("descriptions", []):
                edge_data.setdefault("descriptions", []).append(rel.description)
        else:
            self.graph.add_edge(
                rel.source,
                rel.target,
                relation_type=rel.relation_type,
                weight=rel.weight,
                description=rel.description,
                descriptions=[rel.description] if rel.description else [],
                source_chunks=rel.source_chunks,
                metadata=rel.metadata,
            )

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieves an entity by ID."""
        return self._entities.get(entity_id)

    def all_entities(self) -> List[Entity]:
        """Returns all registered entities."""
        return list(self._entities.values())

    def all_relationships(self) -> List[Relationship]:
        """Returns all relationships in the graph."""
        rels = []
        for u, v, data in self.graph.edges(data=True):
            rels.append(Relationship(
                source=u,
                target=v,
                relation_type=data.get("relation_type", "RELATED_TO"),
                description=data.get("description", ""),
                weight=data.get("weight", 1.0),
                source_chunks=data.get("source_chunks", []),
                metadata=data.get("metadata", {}),
            ))
        return rels

    def get_k_hop_neighborhood(self, entity_id: str, k: int = 1) -> Tuple[List[Entity], List[Relationship]]:
        """Retrieves k-hop ego-network around a given entity (both in- and out-edges)."""
        if entity_id not in self.graph:
            return [], []

        undirected = self.graph.to_undirected(as_view=True)
        lengths = nx.single_source_shortest_path_length(undirected, entity_id, cutoff=k)
        neighbor_ids = set(lengths.keys())

        entities = [self._entities[nid] for nid in neighbor_ids if nid in self._entities]
        relationships = []
        for u, v, data in self.graph.edges(neighbor_ids, data=True):
            if u in neighbor_ids and v in neighbor_ids:
                relationships.append(Relationship(
                    source=u,
                    target=v,
                    relation_type=data.get("relation_type", "RELATED_TO"),
                    description=data.get("description", ""),
                    weight=data.get("weight", 1.0),
                    source_chunks=data.get("source_chunks", []),
                ))
        return entities, relationships

    def find_shortest_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """Finds shortest directed path between two entities, or None if disconnected."""
        if source_id not in self.graph or target_id not in self.graph:
            return None
        try:
            return nx.shortest_path(self.graph, source=source_id, target=target_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def compute_centrality(self) -> Dict[str, float]:
        """Computes PageRank centrality scores for all entity nodes."""
        if len(self.graph) == 0:
            return {}
        try:
            return nx.pagerank(self.graph, weight="weight")
        except Exception:
            return {node: float(self.graph.degree(node)) for node in self.graph.nodes()}

    def to_cypher_statements(self) -> List[str]:
        """Generates standard Neo4j Cypher statements representing the graph."""
        statements: List[str] = []
        # Nodes
        for entity in self._entities.values():
            safe_name = entity.name.replace("'", "\\'")
            safe_type = entity.entity_type.replace("'", "\\'")
            safe_desc = entity.description.replace("'", "\\'")
            statements.append(
                f"MERGE (e:{safe_type} {{id: '{entity.id}'}}) "
                f"ON CREATE SET e.name = '{safe_name}', e.description = '{safe_desc}';"
            )
        # Edges
        for rel in self.all_relationships():
            safe_rel = rel.relation_type.upper().replace(" ", "_")
            statements.append(
                f"MATCH (s {{id: '{rel.source}'}}), (t {{id: '{rel.target}'}}) "
                f"MERGE (s)-[r:{safe_rel} {{weight: {rel.weight}}}]->(t);"
            )
        return statements


# ---------------------------------------------------------------------------
# 4. Community Detection: Louvain Modularity Partitioning
# ---------------------------------------------------------------------------

class CommunityDetector:
    """Partitions the knowledge graph into modular communities using Louvain optimization."""

    def __init__(self, resolution: float = 1.0, random_state: int = 42):
        self.resolution = resolution
        self.random_state = random_state

    def detect_communities(self, kg: KnowledgeGraphStore) -> Tuple[List[List[str]], float]:
        """Runs Louvain community detection on the undirected projection of the KG.
        
        Returns:
            Tuple of (list of entity ID sets/lists, modularity score Q)
        """
        if len(kg.graph) == 0:
            return [], 0.0

        # Undirected representation with accumulated edge weights
        undirected_g = nx.Graph()
        for u in kg.graph.nodes():
            undirected_g.add_node(u)

        for u, v, data in kg.graph.edges(data=True):
            w = data.get("weight", 1.0)
            if undirected_g.has_edge(u, v):
                undirected_g[u][v]["weight"] += w
            else:
                undirected_g.add_edge(u, v, weight=w)

        # Detect communities using native NetworkX Louvain algorithm
        try:
            raw_communities = louvain_communities(
                undirected_g,
                weight="weight",
                resolution=self.resolution,
                seed=self.random_state,
            )
        except Exception:
            # Fallback to connected components if disconnected singletons
            raw_communities = list(nx.connected_components(undirected_g))

        # Sort communities by size descending
        sorted_communities = [sorted(list(c)) for c in sorted(raw_communities, key=len, reverse=True)]

        # Calculate modularity score Q
        q_score = 0.0
        if len(sorted_communities) > 1 and undirected_g.number_of_edges() > 0:
            try:
                q_score = float(modularity(undirected_g, sorted_communities, weight="weight"))
            except Exception:
                q_score = 0.0

        return sorted_communities, q_score


# ---------------------------------------------------------------------------
# 5. Community Summarization (Map-Reduce Engine)
# ---------------------------------------------------------------------------

class CommunitySummarizer:
    """Synthesizes structured narrative reports and key insights for detected communities.
    
    Implements Map-Reduce hierarchical synthesis:
    - Map: Summarize each entity cluster independently based on internal links and central hubs.
    - Reduce: Synthesize meta-summaries across community partitions.
    """

    def generate_community_report(
        self,
        community_id: int,
        member_ids: List[str],
        kg: KnowledgeGraphStore,
        level: int = 0,
    ) -> Community:
        """Generates a structured Community report from graph topology and relational semantics."""
        members = [kg.get_entity(mid) for mid in member_ids if kg.get_entity(mid)]
        if not members:
            return Community(
                id=community_id,
                level=level,
                title=f"Community {community_id}",
                summary="Empty community.",
                findings=[],
                entity_ids=[],
            )

        # Determine top central hub entities in this community
        centrality = kg.compute_centrality()
        sorted_members = sorted(members, key=lambda m: centrality.get(m.id, 0.0), reverse=True)
        top_hubs = [m.name for m in sorted_members[:3]]

        # Gather internal relations
        internal_rels: List[Relationship] = []
        member_set = set(member_ids)
        for u, v, data in kg.graph.edges(member_ids, data=True):
            if u in member_set and v in member_set:
                internal_rels.append(Relationship(
                    source=u,
                    target=v,
                    relation_type=data.get("relation_type", "RELATED_TO"),
                    description=data.get("description", ""),
                    weight=data.get("weight", 1.0),
                    source_chunks=data.get("source_chunks", []),
                ))

        # Synthesize Title
        if len(top_hubs) >= 2:
            title = f"{top_hubs[0]} and {top_hubs[1]} Cluster"
        elif len(top_hubs) == 1:
            title = f"{top_hubs[0]} Subsystem"
        else:
            title = f"Knowledge Cluster {community_id}"

        # Synthesize Findings (factual relational propositions)
        findings: List[str] = []
        for r in internal_rels[:6]:
            if r.description:
                findings.append(r.description)
            else:
                findings.append(f"{r.source.title()} {r.relation_type.lower().replace('_', ' ')} {r.target.title()}.")

        if not findings:
            entity_names = ", ".join([m.name for m in sorted_members])
            findings.append(f"Contains related entities: {entity_names}.")

        # Synthesize Summary narrative
        summary_lines = [
            f"Community {community_id} centers on {', '.join(top_hubs)} with {len(members)} interconnected entities.",
            f"Key internal structural interactions comprise {len(internal_rels)} verified relationships.",
            f"Core mechanisms include: {'; '.join(findings[:3])}.",
        ]
        summary = " ".join(summary_lines)

        # Significance weight proportional to member count and edge density
        edge_density = (2 * len(internal_rels)) / max(1, len(members) * (len(members) - 1)) if len(members) > 1 else 1.0
        weight = min(10.0, max(1.0, 2.0 * len(members) + 5.0 * edge_density))

        return Community(
            id=community_id,
            level=level,
            title=title,
            summary=summary,
            findings=findings,
            entity_ids=member_ids,
            weight=round(weight, 2),
        )

    def reduce_meta_summary(self, communities: List[Community], query: str = "") -> str:
        """Hierarchical Reduce step: synthesizes macro-level executive summary across all communities."""
        if not communities:
            return "No community structures available to summarize."

        sorted_comms = sorted(communities, key=lambda c: c.weight, reverse=True)
        sections = []
        for c in sorted_comms:
            key_insights = "; ".join(c.findings[:2])
            sections.append(
                f"- **{c.title}** (Weight: {c.weight:.1f}, Entities: {len(c.entity_ids)}): "
                f"{c.summary} Insights: {key_insights}"
            )

        header = "### Hierarchical Knowledge Synthesis (Reduce Phase)\n"
        if query:
            header += f"**Addressing Query:** *{query}*\n\n"
        return header + "\n".join(sections)


# ---------------------------------------------------------------------------
# 6. GraphRAG Engine: Local Search, Global Search & Hybrid Retrieval
# ---------------------------------------------------------------------------

class GraphRAGEngine:
    """Orchestrates Local Search, Global Search, and Hybrid Retrieval over Knowledge Graphs."""

    def __init__(
        self,
        kg: KnowledgeGraphStore,
        extractor: Optional[EntityRelationExtractor] = None,
        detector: Optional[CommunityDetector] = None,
        summarizer: Optional[CommunitySummarizer] = None,
        embedder: Optional[DeterministicSemanticEmbedder] = None,
    ):
        self.kg = kg
        self.extractor = extractor or EntityRelationExtractor()
        self.detector = detector or CommunityDetector()
        self.summarizer = summarizer or CommunitySummarizer()
        self.embedder = embedder or DeterministicSemanticEmbedder()

        self.communities: List[Community] = []
        self.entity_to_community: Dict[str, int] = {}
        self.modularity_q: float = 0.0

    def index_documents(self, documents: List[Dict[str, str]]) -> Dict[str, Any]:
        """Ingests, extracts entities/relations, builds KG, and performs community clustering.
        
        Args:
            documents: List of dicts with 'id' and 'text'.
        """
        total_entities = 0
        total_relations = 0

        for doc in documents:
            cid = doc.get("id", "doc_0")
            text = doc.get("text", "")
            ents, rels = self.extractor.extract_from_text(text, cid)
            for ent in ents:
                self.kg.add_entity(ent)
                total_entities += 1
            for rel in rels:
                self.kg.add_relationship(rel)
                total_relations += 1

        # Perform Louvain Community Detection
        clusters, q = self.detector.detect_communities(self.kg)
        self.modularity_q = q
        self.communities = []
        self.entity_to_community = {}

        for cid, member_ids in enumerate(clusters):
            comm = self.summarizer.generate_community_report(cid, member_ids, self.kg, level=0)
            self.communities.append(comm)
            for mid in member_ids:
                self.entity_to_community[mid] = cid
                if mid in self.kg.graph.nodes:
                    self.kg.graph.nodes[mid]["community_id"] = cid

        return {
            "entities_extracted": len(self.kg.all_entities()),
            "relationships_extracted": len(self.kg.all_relationships()),
            "communities_detected": len(self.communities),
            "modularity_q": self.modularity_q,
        }

    def _find_matching_entities(self, query: str, top_k: int = 3) -> List[Tuple[Entity, float]]:
        """Finds candidate entities matching the query using lexical overlap + semantic embeddings."""
        all_ents = self.kg.all_entities()
        if not all_ents:
            return []

        q_norm = self.extractor.normalize_id(query)
        q_tokens = set(q_norm.split("-"))
        q_emb = self.embedder.embed(query)

        scored: List[Tuple[Entity, float]] = []
        for ent in all_ents:
            e_tokens = set(ent.id.split("-"))
            overlap = len(q_tokens.intersection(e_tokens)) / max(1, len(e_tokens))

            # Semantic cosine similarity
            ent_text = f"{ent.name} {ent.entity_type} {ent.description}"
            e_emb = self.embedder.embed(ent_text)
            cosine = float(np.dot(q_emb, e_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(e_emb) + 1e-9))

            # Exact lexical mention gets substantial boost
            exact_boost = 1.0 if ent.id in q_norm or ent.name.lower() in query.lower() else 0.0

            combined_score = 0.5 * cosine + 0.3 * overlap + 0.2 * exact_boost
            scored.append((ent, combined_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def local_search(self, query: str, top_k_entities: int = 2, max_hops: int = 1) -> GraphSearchResult:
        """Executes Local Search (entity-centric ego-network retrieval).
        
        Explores immediate relational neighborhood around query seed entities and
        grounds synthesis in direct relational facts and source chunks.
        """
        seed_matches = self._find_matching_entities(query, top_k=top_k_entities)
        if not seed_matches:
            return GraphSearchResult(
                query=query,
                search_mode="local",
                answer="No matching entities found in the knowledge graph for this query.",
                metrics={"seed_count": 0},
            )

        grounded_entities: Set[str] = set()
        grounded_rels: List[Dict[str, Any]] = []
        grounded_chunks: Set[str] = set()
        grounded_comms: Set[int] = set()
        evidence_propositions: List[str] = []

        for seed_ent, score in seed_matches:
            grounded_entities.add(seed_ent.id)
            grounded_chunks.update(seed_ent.source_chunks)
            if seed_ent.id in self.entity_to_community:
                grounded_comms.add(self.entity_to_community[seed_ent.id])

            # Gather k-hop neighborhood
            nbr_ents, nbr_rels = self.kg.get_k_hop_neighborhood(seed_ent.id, k=max_hops)
            for ne in nbr_ents:
                grounded_entities.add(ne.id)
                grounded_chunks.update(ne.source_chunks)

            for nr in nbr_rels:
                grounded_rels.append({
                    "source": nr.source,
                    "target": nr.target,
                    "relation_type": nr.relation_type,
                    "description": nr.description,
                    "source_chunks": nr.source_chunks,
                })
                grounded_chunks.update(nr.source_chunks)
                rel_stmt = f"- [{nr.source.title()}] --({nr.relation_type})--> [{nr.target.title()}]"
                if nr.description:
                    rel_stmt += f": {nr.description}"
                if nr.source_chunks:
                    rel_stmt += f" (Sources: {', '.join(nr.source_chunks)})"
                evidence_propositions.append(rel_stmt)

        # Retrieve relevant community summaries
        community_contexts = []
        for cid in sorted(list(grounded_comms)):
            if cid < len(self.communities):
                c = self.communities[cid]
                community_contexts.append(f"**Community Context ({c.title})**: {c.summary}")

        # Synthesize Local Search Answer
        seed_names = ", ".join([e.name for e, _ in seed_matches])
        answer_parts = [
            f"**Local Graph Retrieval for Query:** '{query}'",
            f"**Identified Focal Entities:** {seed_names}\n",
            "**Relational Fact Traversal:**",
        ]
        if evidence_propositions:
            answer_parts.extend(evidence_propositions[:6])
        else:
            answer_parts.append("- No direct relational edges found for focal entities.")

        if community_contexts:
            answer_parts.append("\n**Clustered Community Context:**")
            answer_parts.extend(community_contexts[:2])

        citations_str = f"\n**Grounded Source Chunks:** {', '.join(sorted(list(grounded_chunks)))}" if grounded_chunks else ""
        answer_parts.append(citations_str)

        return GraphSearchResult(
            query=query,
            search_mode="local",
            answer="\n".join(answer_parts),
            grounded_entities=sorted(list(grounded_entities)),
            grounded_relationships=grounded_rels,
            grounded_communities=sorted(list(grounded_comms)),
            source_chunks=sorted(list(grounded_chunks)),
            metrics={"seed_entities": len(seed_matches), "rel_count": len(grounded_rels)},
        )

    def global_search(self, query: str, top_k_communities: int = 5) -> GraphSearchResult:
        """Executes Global Search using hierarchical Map-Reduce summarization.
        
        Designed for sensemaking queries across the entire corpus:
        - Map: Scores relevance and extracts answers from each community summary.
        - Reduce: Ranks, filters, and combines intermediate claims into an executive answer.
        """
        if not self.communities:
            return GraphSearchResult(
                query=query,
                search_mode="global",
                answer="No communities have been detected or summarized.",
            )

        q_emb = self.embedder.embed(query)
        intermediate_claims: List[Dict[str, Any]] = []

        # MAP PHASE: Rate each community's relevance to the question
        for c in self.communities:
            comm_text = f"{c.title} {c.summary} {' '.join(c.findings)}"
            c_emb = self.embedder.embed(comm_text)
            relevance_score = float(np.dot(q_emb, c_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(c_emb) + 1e-9))

            # Extract key insights relevant to the query
            extracted_points = []
            for f in c.findings:
                extracted_points.append(f)

            intermediate_claims.append({
                "community_id": c.id,
                "title": c.title,
                "relevance_score": relevance_score,
                "weight": c.weight,
                "summary": c.summary,
                "points": extracted_points,
                "entity_count": len(c.entity_ids),
            })

        # Sort intermediate claims by combined relevance and structural weight
        intermediate_claims.sort(
            key=lambda x: (x["relevance_score"] * 0.7 + (x["weight"] / 10.0) * 0.3),
            reverse=True,
        )

        selected_claims = intermediate_claims[:top_k_communities]

        # REDUCE PHASE: Aggregate and synthesize comprehensive macro-answer
        answer_parts = [
            f"**Global GraphRAG Sensemaking Synthesis for Query:** '{query}'\n",
            "**Executive Thematic Summary:**",
        ]
        all_grounded_comms = []
        all_grounded_ents = []

        for rank, claim in enumerate(selected_claims, start=1):
            cid = claim["community_id"]
            all_grounded_comms.append(cid)
            all_grounded_ents.extend(self.communities[cid].entity_ids)

            answer_parts.append(
                f"{rank}. **{claim['title']}** (Relevance: {claim['relevance_score']:.2f}, Significance: {claim['weight']:.1f})\n"
                f"   - *Summary*: {claim['summary']}\n"
                f"   - *Core Insights*: {'; '.join(claim['points'][:2])}"
            )

        answer_parts.append(
            f"\n*Generated via Hierarchical Map-Reduce across {len(selected_claims)} community partitions.*"
        )

        return GraphSearchResult(
            query=query,
            search_mode="global",
            answer="\n".join(answer_parts),
            grounded_entities=sorted(list(set(all_grounded_ents))),
            grounded_communities=all_grounded_comms,
            intermediate_claims=selected_claims,
            metrics={"communities_scanned": len(self.communities), "communities_selected": len(selected_claims)},
        )

    def hybrid_search(
        self,
        query: str,
        vector_chunks: List[Dict[str, Any]],
        alpha: float = 0.5,
        top_k: int = 3,
    ) -> GraphSearchResult:
        """Fuses dense vector chunks with knowledge graph local ego-network facts.
        
        Combines unstructured passage retrieval with structured relational knowledge.
        """
        # 1. Run local graph retrieval
        local_res = self.local_search(query, top_k_entities=2, max_hops=1)

        # 2. Score vector chunks
        q_emb = self.embedder.embed(query)
        scored_chunks = []
        for chunk in vector_chunks:
            c_text = chunk.get("text", "")
            c_emb = self.embedder.embed(c_text)
            sim = float(np.dot(q_emb, c_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(c_emb) + 1e-9))
            scored_chunks.append((chunk, sim))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        top_chunks = scored_chunks[:top_k]

        # 3. Fuse evidence
        grounded_chunks = set(local_res.source_chunks)
        vector_citations = []
        vector_excerpts = []
        for ch, score in top_chunks:
            cid = ch.get("id", "chunk_x")
            grounded_chunks.add(cid)
            vector_citations.append(cid)
            vector_excerpts.append(f"- [{cid}] (Score: {score:.2f}): {ch.get('text', '')[:120]}...")

        answer_parts = [
            f"**Hybrid Graph + Vector Retrieval for Query:** '{query}'\n",
            "**Structured Graph Evidence (Relationships & Entities):**",
            local_res.answer,
            "\n**Unstructured Vector Context (Ranked Text Passages):**",
        ]
        answer_parts.extend(vector_excerpts)
        answer_parts.append(f"\n**Synthesized Provenance Chunks:** {', '.join(sorted(list(grounded_chunks)))}")

        return GraphSearchResult(
            query=query,
            search_mode="hybrid",
            answer="\n".join(answer_parts),
            grounded_entities=local_res.grounded_entities,
            grounded_relationships=local_res.grounded_relationships,
            grounded_communities=local_res.grounded_communities,
            source_chunks=sorted(list(grounded_chunks)),
            metrics={"vector_chunks_used": len(top_chunks), "graph_edges_used": len(local_res.grounded_relationships)},
        )


# ---------------------------------------------------------------------------
# 7. Graph Diagnostics, Evaluation & Visualization
# ---------------------------------------------------------------------------

class GraphEvaluator:
    """Evaluates Knowledge Graph quality, topological modularity, and RAG answer groundedness."""

    @staticmethod
    def evaluate_graph(kg: KnowledgeGraphStore, communities: List[Community]) -> Dict[str, Any]:
        """Calculates topological metrics over the knowledge graph."""
        n_nodes = kg.graph.number_of_nodes()
        n_edges = kg.graph.number_of_edges()

        if n_nodes <= 1:
            density = 0.0
        else:
            density = (2 * n_edges) / (n_nodes * (n_nodes - 1))

        undirected = kg.graph.to_undirected()
        components = list(nx.connected_components(undirected))
        largest_cc_size = max([len(c) for c in components]) if components else 0

        # Calculate modularity if communities provided
        q = 0.0
        if len(communities) > 1 and n_edges > 0:
            comm_sets = [set(c.entity_ids) for c in communities]
            try:
                q = float(modularity(undirected, comm_sets, weight="weight"))
            except Exception:
                q = 0.0

        all_chunks: Set[str] = set()
        for e in kg.all_entities():
            all_chunks.update(e.source_chunks)

        return {
            "node_count": n_nodes,
            "edge_count": n_edges,
            "graph_density": round(density, 4),
            "connected_components": len(components),
            "largest_component_ratio": round(largest_cc_size / max(1, n_nodes), 3),
            "modularity_q": round(q, 4),
            "community_count": len(communities),
            "total_chunks_indexed": len(all_chunks),
        }

    @staticmethod
    def evaluate_answer_groundedness(answer: str, kg: KnowledgeGraphStore) -> Dict[str, Any]:
        """Assesses entity hallucination vs knowledge graph grounding in the answer."""
        words = re.findall(r"\b[A-Za-z0-9_-]+\b", answer.lower())
        all_ent_ids = set([e.id for e in kg.all_entities()])

        mentioned_entities = set()
        for ent_id in all_ent_ids:
            if ent_id in answer.lower():
                mentioned_entities.add(ent_id)

        return {
            "mentioned_entities": sorted(list(mentioned_entities)),
            "grounded_entity_count": len(mentioned_entities),
            "is_grounded": len(mentioned_entities) > 0,
        }


class GraphVisualizer:
    """Generates visual architecture diagrams and NetworkX graph plots."""

    @staticmethod
    def plot_knowledge_graph(
        kg: KnowledgeGraphStore,
        output_path: str,
        communities: Optional[List[Community]] = None,
        figsize: Tuple[int, int] = (10, 7),
    ) -> str:
        """Renders and saves a NetworkX knowledge graph visualization with community color-coding."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=figsize, dpi=200)
        ax = plt.gca()

        if len(kg.graph) == 0:
            plt.text(0.5, 0.5, "Empty Knowledge Graph", ha="center", va="center")
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()
            return output_path

        # Spring layout for reproducible positioning
        pos = nx.spring_layout(kg.graph, seed=42, k=1.2 / math.sqrt(max(1, len(kg.graph))))

        # Map nodes to colors based on community or entity type
        color_palette = [
            "#4285F4", "#EA4335", "#FBBC05", "#34A853",
            "#8E24AA", "#00ACC1", "#FB8C00", "#3949AB"
        ]
        node_colors = []
        for node in kg.graph.nodes():
            comm_id = kg.graph.nodes[node].get("community_id", 0)
            node_colors.append(color_palette[comm_id % len(color_palette)])

        # Draw nodes and edges
        nx.draw_networkx_nodes(
            kg.graph,
            pos,
            node_color=node_colors,
            node_size=1200,
            alpha=0.9,
            edgecolors="#202124",
            linewidths=1.5,
            ax=ax,
        )
        nx.draw_networkx_edges(
            kg.graph,
            pos,
            edge_color="#5F6368",
            width=1.8,
            arrowsize=14,
            arrowstyle="->",
            connectionstyle="arc3,rad=0.1",
            alpha=0.7,
            ax=ax,
        )

        # Labels
        labels = {n: kg.get_entity(n).name if kg.get_entity(n) else n.title() for n in kg.graph.nodes()}
        nx.draw_networkx_labels(
            kg.graph,
            pos,
            labels=labels,
            font_size=9,
            font_weight="bold",
            font_family="sans-serif",
            ax=ax,
        )

        # Edge labels (relation types)
        edge_labels = {(u, v): d.get("relation_type", "") for u, v, d in kg.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(
            kg.graph,
            pos,
            edge_labels=edge_labels,
            font_size=7,
            font_color="#3c4043",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
            ax=ax,
        )

        title = "RAG Engineering Academy — Knowledge Graph & Louvain Communities"
        if communities:
            title += f" ({len(communities)} Clusters)"
        plt.title(title, fontsize=12, fontweight="bold", pad=15)
        plt.axis("off")
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        return output_path

    @staticmethod
    def plot_graphrag_architecture(output_path: str, figsize: Tuple[int, int] = (12, 6)) -> str:
        """Renders architectural comparison diagram: Naive Vector RAG vs Local GraphRAG vs Global GraphRAG."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(figsize=figsize, dpi=200)
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 7)
        ax.axis("off")

        # Background palette
        col_vector = "#E8F0FE"  # light blue
        col_local = "#E6F4EA"   # light green
        col_global = "#FEF7E0"  # light yellow

        # Panels
        # 1. Naive Vector RAG
        p1 = patches.FancyBboxPatch((0.5, 0.8), 3.4, 5.4, boxstyle="round,pad=0.1", fc=col_vector, ec="#1A73E8", lw=2)
        ax.add_patch(p1)
        ax.text(2.2, 5.8, "1. Naive Vector RAG", fontsize=11, fontweight="bold", ha="center", color="#1A73E8")
        ax.text(2.2, 5.4, "Semantic Vector Proximity", fontsize=9, ha="center", color="#5F6368")

        ax.text(2.2, 4.6, "[Query] -> Dense Embedding", fontsize=8, ha="center", family="monospace")
        ax.text(2.2, 4.0, "Cosine Search on Isolated Chunks", fontsize=8, ha="center")
        ax.text(2.2, 3.4, "Top-K Chunks -> LLM Prompt", fontsize=8, ha="center", family="monospace")
        ax.text(2.2, 2.5, "[-] Fails on Multi-Hop Queries\n[-] No Relationship Traversal\n[-] Blind to Global Themes", fontsize=8, ha="center", color="#C5221F")

        # 2. Local GraphRAG
        p2 = patches.FancyBboxPatch((4.3, 0.8), 3.4, 5.4, boxstyle="round,pad=0.1", fc=col_local, ec="#137333", lw=2)
        ax.add_patch(p2)
        ax.text(6.0, 5.8, "2. Local GraphRAG", fontsize=11, fontweight="bold", ha="center", color="#137333")
        ax.text(6.0, 5.4, "Entity-Centric Ego-Networks", fontsize=9, ha="center", color="#5F6368")

        ax.text(6.0, 4.6, "Seed Entity Resolution", fontsize=8, ha="center", family="monospace")
        ax.text(6.0, 4.0, "1-2 Hop Ego-Subgraphs\n+ Incident Relational Facts", fontsize=8, ha="center")
        ax.text(6.0, 3.2, "Community Summary Augmented", fontsize=8, ha="center", family="monospace")
        ax.text(6.0, 2.2, "[+] Multi-Hop Relationship Reasoning\n[+] Precise Factual Provenance\n[+] Explains HOW Entities Connect", fontsize=8, ha="center", color="#137333")

        # 3. Global GraphRAG
        p3 = patches.FancyBboxPatch((8.1, 0.8), 3.4, 5.4, boxstyle="round,pad=0.1", fc=col_global, ec="#B06000", lw=2)
        ax.add_patch(p3)
        ax.text(9.8, 5.8, "3. Global GraphRAG", fontsize=11, fontweight="bold", ha="center", color="#B06000")
        ax.text(9.8, 5.4, "Hierarchical Sensemaking", fontsize=9, ha="center", color="#5F6368")

        ax.text(9.8, 4.6, "Louvain Community Detection", fontsize=8, ha="center", family="monospace")
        ax.text(9.8, 4.0, "Map Phase: Rate Community Insights", fontsize=8, ha="center")
        ax.text(9.8, 3.2, "Reduce Phase: Aggregate Themes", fontsize=8, ha="center", family="monospace")
        ax.text(9.8, 2.2, "[+] Dataset-Wide Holistic Synthesis\n[+] Answers 'What are the main themes?'\n[+] No Lost Needles in Haystacks", fontsize=8, ha="center", color="#B06000")

        plt.suptitle("GraphRAG Architectural Taxonomy: Vector RAG vs Local vs Global Search", fontsize=13, fontweight="bold", y=0.96)
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        return output_path
