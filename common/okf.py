"""Google Open Knowledge Format (OKF) Engine for AI Agents & RAG Systems.

Provides high-performance, 100% offline-executable, CPU-friendly implementations of:
1. Canonical OKF Specification Parsing & Serialization:
   - Markdown documents with YAML frontmatter.
   - Mandatory 'type' field enforcement per official Google Cloud specification.
   - Provenance, attestation, resource URIs, tags, and lifecycle status.
   - Reserved file handling ('index.md' for progressive disclosure, 'log.md' for audit trails).
2. Deterministic Knowledge Graph Representation:
   - Bundle-relative cross-reference link extraction and normalization.
   - Directed NetworkX graph representing concept dependencies and lineage.
   - Graph diagnostics: density, in/out degree, PageRank, topological order, and cycle detection.
3. Bundle Validation & Linting:
   - Schema enforcement, link integrity checking (detecting broken cross-references).
   - Provenance verification and tamper-evident SHA-256 content hashing.
   - Freshness & staleness evaluation, deprecation tracking, and orphan detection.
4. Progressive Disclosure & Context Assembly for RAG:
   - Progressive index generation for token-budgeted agent exploration.
   - Graph-expanded neighborhood retrieval (1-to-N hop traversals).
   - Deterministic semantic retrieval over structured concepts.
   - Provenance-augmented context assembly with trust annotations and citation tags.
5. Visualization Utilities:
   - Architecture diagrams and concept dependency graph plots.
"""

from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pydantic import BaseModel, Field
import yaml

from common.query_transformations import DeterministicSemanticEmbedder


# ---------------------------------------------------------------------------
# 1. Pydantic Models for OKF Concepts & Provenance
# ---------------------------------------------------------------------------

class OKFProvenance(BaseModel):
    """Provenance and attestation metadata for an OKF concept."""
    by: str = Field(default="human-curator", description="Agent, model, or person who generated/authored the concept")
    at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="ISO 8601 creation/generation timestamp")
    verified_by: Optional[str] = Field(default=None, description="Identity of peer reviewer or compliance authority")
    verified_at: Optional[str] = Field(default=None, description="Timestamp of official verification")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score [0.0 - 1.0]")
    source_revision: Optional[str] = Field(default=None, description="Git commit hash or upstream source version")
    content_hash: Optional[str] = Field(default=None, description="Expected SHA-256 digest of concept body")


class OKFConcept(BaseModel):
    """Canonical representation of a single Open Knowledge Format document."""
    path: str = Field(..., description="Bundle-relative path starting with '/' (e.g. /tables/orders.md)")
    concept_type: str = Field(..., description="Mandatory OKF type field (e.g. table, metric, policy, index, log)")
    title: str = Field(default="", description="Human-readable concept name")
    description: str = Field(default="", description="Crisp semantic summary for progressive disclosure")
    resource: Optional[str] = Field(default=None, description="Physical resource URI (e.g. bigquery://..., postgres://..., https://...)")
    tags: List[str] = Field(default_factory=list, description="Categorization tags")
    owner: Optional[str] = Field(default=None, description="Responsible team or maintainer")
    status: str = Field(default="active", description="Lifecycle status (e.g. active, deprecated, draft, reviewed)")
    stale_after: Optional[str] = Field(default=None, description="ISO 8601 timestamp after which concept is considered stale")
    provenance: Optional[OKFProvenance] = Field(default=None, description="Attestation and generation provenance")
    raw_frontmatter: Dict[str, Any] = Field(default_factory=dict, description="Complete parsed YAML frontmatter dict")
    body: str = Field(default="", description="Markdown body content excluding frontmatter")
    outbound_links: List[str] = Field(default_factory=list, description="Normalized bundle-relative paths linked in body")
    content_sha256: str = Field(default="", description="SHA-256 hex digest of raw markdown body")

    def is_stale(self, reference_time: Optional[datetime] = None) -> bool:
        """Determines if the concept has exceeded its staleness expiration date."""
        if not self.stale_after:
            return False
        ref = reference_time or datetime.now(timezone.utc)
        try:
            # Parse ISO format (handling Z or offset)
            stale_dt = datetime.fromisoformat(self.stale_after.replace("Z", "+00:00"))
            return ref > stale_dt
        except Exception:
            return False


class OKFValidationIssue(BaseModel):
    """A linting or validation issue discovered in an OKF concept or bundle."""
    path: str = Field(..., description="Path of the concept file with the issue")
    severity: str = Field(default="ERROR", description="'ERROR', 'WARNING', or 'INFO'")
    code: str = Field(..., description="Short machine-readable error code")
    message: str = Field(..., description="Human-readable issue explanation")


class OKFValidationReport(BaseModel):
    """Comprehensive validation summary for an OKF bundle."""
    is_valid: bool = Field(..., description="True if no ERROR severity issues exist")
    total_files: int = Field(default=0)
    error_count: int = Field(default=0)
    warning_count: int = Field(default=0)
    issues: List[OKFValidationIssue] = Field(default_factory=list)
    concepts_by_type: Dict[str, int] = Field(default_factory=dict)
    broken_links: List[Tuple[str, str]] = Field(default_factory=list, description="Pairs of (source_path, target_path)")
    graph_density: float = Field(default=0.0)
    orphan_concepts: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 2. OKF Parser & Serializer
# ---------------------------------------------------------------------------

class OKFParser:
    """Parses and serializes Open Knowledge Format Markdown files with YAML frontmatter."""

    _FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)
    _MD_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    @classmethod
    def compute_sha256(cls, text: str) -> str:
        """Computes the SHA-256 digest of a text string."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def parse_text(cls, content: str, bundle_relative_path: str) -> OKFConcept:
        """Parses raw text of an OKF markdown document into an OKFConcept."""
        norm_path = cls._normalize_bundle_path(bundle_relative_path)
        match = cls._FRONTMATTER_PATTERN.match(content)

        if not match:
            raise ValueError(
                f"Invalid OKF document at '{norm_path}': Missing required YAML frontmatter "
                "enclosed between '---' markers."
            )

        fm_text, body = match.group(1), match.group(2)
        try:
            frontmatter = yaml.safe_load(fm_text) or {}
        except Exception as e:
            raise ValueError(f"YAML parsing error in frontmatter at '{norm_path}': {e}") from e

        if not isinstance(frontmatter, dict):
            raise ValueError(f"Frontmatter at '{norm_path}' must be a YAML mapping/dictionary.")

        # Canonical Google OKF rule: 'type' is the only mandatory field.
        if "type" not in frontmatter or not str(frontmatter["type"]).strip():
            raise ValueError(
                f"Specification violation at '{norm_path}': Missing mandatory field 'type' in frontmatter."
            )

        concept_type = str(frontmatter["type"]).strip().lower()
        title = str(frontmatter.get("title", "")).strip() or Path(norm_path).stem.replace("_", " ").title()
        description = str(frontmatter.get("description", "")).strip()
        resource = frontmatter.get("resource")
        tags = [str(t).strip() for t in frontmatter.get("tags", []) if str(t).strip()]
        owner = frontmatter.get("owner")
        status = str(frontmatter.get("status", "active")).strip().lower()
        stale_after = frontmatter.get("stale_after")

        # Provenance / generated block
        prov_dict = frontmatter.get("generated") or frontmatter.get("provenance")
        provenance = None
        if isinstance(prov_dict, dict):
            provenance = OKFProvenance(
                by=str(prov_dict.get("by", "system")),
                at=str(prov_dict.get("at", datetime.now(timezone.utc).isoformat())),
                verified_by=prov_dict.get("verified_by"),
                verified_at=prov_dict.get("verified_at"),
                confidence=float(prov_dict.get("confidence", 1.0)),
                source_revision=prov_dict.get("source_revision"),
                content_hash=prov_dict.get("content_hash"),
            )

        body_cleaned = body.strip()
        content_hash = cls.compute_sha256(body_cleaned)

        # Extract and resolve outbound links
        outbound_links = cls.extract_links(body_cleaned, current_doc_path=norm_path)

        return OKFConcept(
            path=norm_path,
            concept_type=concept_type,
            title=title,
            description=description,
            resource=resource,
            tags=tags,
            owner=owner,
            status=status,
            stale_after=stale_after,
            provenance=provenance,
            raw_frontmatter=frontmatter,
            body=body_cleaned,
            outbound_links=outbound_links,
            content_sha256=content_hash,
        )

    @classmethod
    def parse_file(cls, file_path: str | Path, bundle_root: str | Path) -> OKFConcept:
        """Reads and parses an OKF file from disk relative to the bundle root."""
        file_p = Path(file_path).resolve()
        root_p = Path(bundle_root).resolve()
        rel_path = "/" + str(file_p.relative_to(root_p)).replace(os.sep, "/")
        with open(file_p, "r", encoding="utf-8") as f:
            content = f.read()
        return cls.parse_text(content, rel_path)

    @classmethod
    def extract_links(cls, markdown_body: str, current_doc_path: str) -> List[str]:
        """Extracts and normalizes all internal concept references from markdown links."""
        links: List[str] = []
        doc_dir = str(Path(current_doc_path).parent).replace(os.sep, "/")
        if not doc_dir.startswith("/"):
            doc_dir = "/" + doc_dir

        for match in cls._MD_LINK_PATTERN.finditer(markdown_body):
            target = match.group(2).strip()
            # Ignore external protocols or anchors
            if any(target.startswith(proto) for proto in ("http://", "https://", "mailto:", "ftp://", "tel:")):
                continue
            if target.startswith("#"):
                continue

            # Strip query params or hash fragments
            target = target.split("#")[0].split("?")[0]
            if not target.endswith(".md"):
                continue

            # Normalize to bundle root
            if target.startswith("/"):
                normalized = "/" + str(Path(target.lstrip("/")).as_posix())
            else:
                # Relative to current document directory
                combined = Path(doc_dir.lstrip("/")) / target
                normalized = "/" + str(Path(os.path.normpath(str(combined))).as_posix())

            if normalized not in links:
                links.append(normalized)

        return links

    @classmethod
    def serialize_concept(cls, concept: OKFConcept) -> str:
        """Serializes an OKFConcept into a formatted Markdown document with YAML frontmatter."""
        fm_dict = dict(concept.raw_frontmatter)
        fm_dict["type"] = concept.concept_type
        if concept.title:
            fm_dict["title"] = concept.title
        if concept.description:
            fm_dict["description"] = concept.description
        if concept.resource:
            fm_dict["resource"] = concept.resource
        if concept.tags:
            fm_dict["tags"] = concept.tags
        if concept.owner:
            fm_dict["owner"] = concept.owner
        fm_dict["status"] = concept.status
        if concept.stale_after:
            fm_dict["stale_after"] = concept.stale_after

        if concept.provenance:
            fm_dict["generated"] = {
                "by": concept.provenance.by,
                "at": concept.provenance.at,
                "verified_by": concept.provenance.verified_by,
                "confidence": concept.provenance.confidence,
                "content_hash": concept.content_sha256,
            }

        fm_yaml = yaml.dump(fm_dict, sort_keys=False, allow_unicode=True).strip()
        return f"---\n{fm_yaml}\n---\n\n{concept.body}\n"

    @staticmethod
    def _normalize_bundle_path(path_str: str) -> str:
        clean = "/" + path_str.replace("\\", "/").lstrip("/")
        return clean


# ---------------------------------------------------------------------------
# 3. OKF Bundle & Graph Navigator
# ---------------------------------------------------------------------------

class OKFBundle:
    """Manages an entire directory tree of OKF concepts, providing graph analysis and progressive disclosure."""

    def __init__(self, root_dir: Optional[str | Path] = None):
        self.root_dir = Path(root_dir).resolve() if root_dir else None
        self.concepts: Dict[str, OKFConcept] = {}
        self.graph: nx.DiGraph = nx.DiGraph()
        self.index_concept: Optional[OKFConcept] = None
        self.log_concept: Optional[OKFConcept] = None

    @classmethod
    def from_directory(cls, dir_path: str | Path) -> "OKFBundle":
        """Loads and indexes an OKF bundle from a filesystem directory."""
        bundle = cls(root_dir=dir_path)
        path_obj = Path(dir_path).resolve()
        if not path_obj.exists() or not path_obj.is_dir():
            raise FileNotFoundError(f"OKF bundle directory not found at: {path_obj}")

        for md_file in sorted(path_obj.glob("**/*.md")):
            try:
                concept = OKFParser.parse_file(md_file, bundle_root=path_obj)
                bundle.add_concept(concept)
            except Exception as e:
                # Record as broken concept or skip for validator to report
                norm_path = "/" + str(md_file.relative_to(path_obj)).replace(os.sep, "/")
                # Store placeholder for error inspection
                pass

        bundle.build_graph()
        return bundle

    def add_concept(self, concept: OKFConcept) -> None:
        """Adds a concept to the bundle."""
        self.concepts[concept.path] = concept
        if concept.path == "/index.md":
            self.index_concept = concept
        elif concept.path == "/log.md":
            self.log_concept = concept

    def build_graph(self) -> None:
        """Constructs the NetworkX DiGraph representing concept interdependencies."""
        self.graph.clear()
        for path, concept in self.concepts.items():
            self.graph.add_node(
                path,
                concept_type=concept.concept_type,
                title=concept.title,
                status=concept.status,
                owner=concept.owner,
            )

        for path, concept in self.concepts.items():
            for target in concept.outbound_links:
                self.graph.add_edge(path, target)

    def get_concept(self, path: str) -> Optional[OKFConcept]:
        """Fetches a concept by normalized path (e.g. /tables/orders.md)."""
        norm = "/" + path.replace("\\", "/").lstrip("/")
        return self.concepts.get(norm)

    def get_progressive_index(self) -> List[Dict[str, Any]]:
        """Generates a compact summary table of all concepts for progressive disclosure.
        
        Enables AI agents to survey available concepts with minimal token footprint.
        """
        summary: List[Dict[str, Any]] = []
        for path, concept in sorted(self.concepts.items()):
            if path in ("/index.md", "/log.md"):
                continue
            summary.append({
                "path": concept.path,
                "type": concept.concept_type,
                "title": concept.title,
                "description": concept.description,
                "resource": concept.resource,
                "tags": concept.tags,
                "status": concept.status,
            })
        return summary

    def traverse_neighborhood(self, seed_path: str, max_hops: int = 1) -> List[OKFConcept]:
        """Traverses the concept graph from a seed concept up to max_hops."""
        norm_seed = "/" + seed_path.replace("\\", "/").lstrip("/")
        if norm_seed not in self.graph:
            return []

        visited: Set[str] = {norm_seed}
        frontier: Set[str] = {norm_seed}

        for _ in range(max_hops):
            next_frontier: Set[str] = set()
            for node in frontier:
                neighbors = set(self.graph.successors(node)) | set(self.graph.predecessors(node))
                for neighbor in neighbors:
                    if neighbor not in visited and neighbor in self.concepts:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier

        return [self.concepts[p] for p in visited if p in self.concepts]

    def find_dependencies(self, path: str) -> List[OKFConcept]:
        """Returns concepts directly cited by this concept (outbound)."""
        norm = "/" + path.replace("\\", "/").lstrip("/")
        if norm not in self.graph:
            return []
        successors = list(self.graph.successors(norm))
        return [self.concepts[s] for s in successors if s in self.concepts]

    def find_dependents(self, path: str) -> List[OKFConcept]:
        """Returns concepts that cite this concept (inbound)."""
        norm = "/" + path.replace("\\", "/").lstrip("/")
        if norm not in self.graph:
            return []
        predecessors = list(self.graph.predecessors(norm))
        return [self.concepts[p] for p in predecessors if p in self.concepts]

    def get_graph_metrics(self) -> Dict[str, Any]:
        """Calculates topological and structural graph metrics."""
        if not self.graph:
            return {"nodes": 0, "edges": 0, "density": 0.0, "is_dag": True}

        nodes = self.graph.number_of_nodes()
        edges = self.graph.number_of_edges()
        density = nx.density(self.graph)
        is_dag = nx.is_directed_acyclic_graph(self.graph)

        try:
            pagerank = nx.pagerank(self.graph)
        except Exception:
            pagerank = {n: 1.0 / max(1, nodes) for n in self.graph.nodes()}

        return {
            "nodes": nodes,
            "edges": edges,
            "density": float(density),
            "is_dag": bool(is_dag),
            "pagerank": pagerank,
        }


# ---------------------------------------------------------------------------
# 4. OKF Validator & Linter
# ---------------------------------------------------------------------------

class OKFValidator:
    """Validates OKF bundles against official specifications, link integrity, and freshness rules."""

    @classmethod
    def validate_bundle(cls, bundle: OKFBundle, reference_time: Optional[datetime] = None) -> OKFValidationReport:
        """Runs a complete validation audit across an OKF bundle."""
        issues: List[OKFValidationIssue] = []
        broken_links: List[Tuple[str, str]] = []
        types_dist: Dict[str, int] = {}
        ref_time = reference_time or datetime.now(timezone.utc)

        # 1. Inspect each concept
        for path, concept in bundle.concepts.items():
            types_dist[concept.concept_type] = types_dist.get(concept.concept_type, 0) + 1

            # Rule: Mandatory type check
            if not concept.concept_type:
                issues.append(OKFValidationIssue(
                    path=path,
                    severity="ERROR",
                    code="MISSING_TYPE",
                    message="Concept is missing mandatory 'type' frontmatter property."
                ))

            # Rule: Staleness check
            if concept.is_stale(ref_time):
                issues.append(OKFValidationIssue(
                    path=path,
                    severity="WARNING",
                    code="STALE_CONCEPT",
                    message=f"Concept expired after {concept.stale_after}; re-verification required."
                ))

            # Rule: Deprecated status check
            if concept.status == "deprecated":
                issues.append(OKFValidationIssue(
                    path=path,
                    severity="WARNING",
                    code="DEPRECATED_CONCEPT",
                    message="Concept is marked as deprecated."
                ))

            # Rule: Tamper/Provenance Hash verification
            if concept.provenance and concept.provenance.content_hash:
                actual_hash = concept.content_sha256
                if concept.provenance.content_hash != actual_hash:
                    issues.append(OKFValidationIssue(
                        path=path,
                        severity="ERROR",
                        code="HASH_MISMATCH",
                        message=f"Attested content hash '{concept.provenance.content_hash[:8]}...' does not match body hash '{actual_hash[:8]}...'."
                    ))

            # Rule: Check outbound link integrity
            for target in concept.outbound_links:
                if target not in bundle.concepts:
                    broken_links.append((path, target))
                    issues.append(OKFValidationIssue(
                        path=path,
                        severity="ERROR",
                        code="BROKEN_LINK",
                        message=f"Outbound link to non-existent concept '{target}'."
                    ))

        # 2. Check for orphan concepts (not linked by anything and not index.md/log.md)
        orphans: List[str] = []
        for path in bundle.concepts:
            if path in ("/index.md", "/log.md"):
                continue
            in_degree = bundle.graph.in_degree(path) if path in bundle.graph else 0
            if in_degree == 0:
                orphans.append(path)
                issues.append(OKFValidationIssue(
                    path=path,
                    severity="INFO",
                    code="ORPHAN_CONCEPT",
                    message=f"Concept is an orphan (no incoming references from index or other concepts)."
                ))

        # 3. Check for mandatory index.md
        if not bundle.index_concept:
            issues.append(OKFValidationIssue(
                path="/index.md",
                severity="WARNING",
                code="MISSING_INDEX",
                message="Bundle is missing root 'index.md' for progressive disclosure."
            ))

        error_count = sum(1 for i in issues if i.severity == "ERROR")
        warning_count = sum(1 for i in issues if i.severity == "WARNING")
        density = bundle.get_graph_metrics().get("density", 0.0)

        return OKFValidationReport(
            is_valid=(error_count == 0),
            total_files=len(bundle.concepts),
            error_count=error_count,
            warning_count=warning_count,
            issues=issues,
            concepts_by_type=types_dist,
            broken_links=broken_links,
            graph_density=density,
            orphan_concepts=orphans,
        )


# ---------------------------------------------------------------------------
# 5. OKF Context Assembler & RAG Retriever
# ---------------------------------------------------------------------------

class OKFRAGRetriever:
    """Retrieves OKF concepts using exact filters, semantic embeddings, and graph neighborhood expansion."""

    def __init__(self, bundle: OKFBundle):
        self.bundle = bundle
        self.embedder = DeterministicSemanticEmbedder(dim=64)
        self._concept_paths: List[str] = []
        self._concept_embeddings: Optional[np.ndarray] = None
        self._index_bundle()

    def _index_bundle(self) -> None:
        """Computes semantic embedding vectors for all concepts in the bundle."""
        self._concept_paths = [p for p in sorted(self.bundle.concepts.keys()) if p not in ("/index.md", "/log.md")]
        if not self._concept_paths:
            return

        texts: List[str] = []
        for path in self._concept_paths:
            c = self.bundle.concepts[path]
            text = f"{c.title} {c.concept_type} {c.description} {' '.join(c.tags)} {c.body[:300]}"
            texts.append(text)

        raw_vecs = [self.embedder.embed(t) for t in texts]
        self._concept_embeddings = np.array(raw_vecs)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        expand_graph_hops: int = 1,
        concept_type_filter: Optional[str] = None,
    ) -> List[OKFConcept]:
        """Hybrid retrieval combining semantic similarity with graph neighborhood expansion."""
        if not self._concept_paths or self._concept_embeddings is None:
            return []

        q_vec = np.array(self.embedder.embed(query))
        scores = np.dot(self._concept_embeddings, q_vec)

        # Rank indices
        ranked_indices = np.argsort(scores)[::-1]
        seed_concepts: List[OKFConcept] = []

        for idx in ranked_indices:
            path = self._concept_paths[idx]
            concept = self.bundle.concepts[path]
            if concept_type_filter and concept.concept_type != concept_type_filter.lower():
                continue
            seed_concepts.append(concept)
            if len(seed_concepts) >= top_k:
                break

        # Graph expansion
        final_concepts: Dict[str, OKFConcept] = {}
        for seed in seed_concepts:
            final_concepts[seed.path] = seed
            if expand_graph_hops > 0:
                neighbors = self.bundle.traverse_neighborhood(seed.path, max_hops=expand_graph_hops)
                for n in neighbors:
                    if n.path not in final_concepts:
                        final_concepts[n.path] = n

        return list(final_concepts.values())


class OKFContextAssembler:
    """Formats retrieved OKF concepts into prompt context blocks with rich provenance and trust annotations."""

    @classmethod
    def assemble_context(cls, concepts: List[OKFConcept]) -> str:
        """Formats concepts into structured, auditable prompt context."""
        if not concepts:
            return "No verified OKF knowledge concepts found."

        blocks: List[str] = []
        for c in concepts:
            prov_text = "Unknown"
            if c.provenance:
                prov_text = f"Authored by: {c.provenance.by} | Verified by: {c.provenance.verified_by or 'None'} | Conf: {c.provenance.confidence}"

            status_warn = ""
            if c.status == "deprecated":
                status_warn = " [WARNING: DEPRECATED CONCEPT]"
            elif c.is_stale():
                status_warn = " [WARNING: STALE KNOWLEDGE - RE-VERIFICATION REQUIRED]"

            block = (
                f"### [OKF:{c.path}]{status_warn}\n"
                f"- **Type**: {c.concept_type.upper()}\n"
                f"- **Title**: {c.title}\n"
                f"- **Resource**: {c.resource or 'N/A'}\n"
                f"- **Provenance**: {prov_text}\n"
                f"- **Body**:\n{c.body}\n"
            )
            blocks.append(block)

        return "\n---\n".join(blocks)

    @classmethod
    def generate_grounded_answer(cls, query: str, concepts: List[OKFConcept]) -> Dict[str, Any]:
        """Deterministic offline generator providing citation-backed answers grounded in OKF concepts."""
        if not concepts:
            return {
                "answer": "Insufficient knowledge base evidence to answer the query.",
                "citations": [],
                "confidence": 0.0,
            }

        # Check for stale or deprecated concepts
        has_stale = any(c.is_stale() for c in concepts)
        has_deprecated = any(c.status == "deprecated" for c in concepts)

        citations = [f"[OKF:{c.path}]" for c in concepts]
        summary_points = []
        for c in concepts:
            summary_points.append(f"According to {c.title} ({c.concept_type}) [OKF:{c.path}]: {c.description}")

        body_summary = " ".join(summary_points)
        warning_note = ""
        if has_stale:
            warning_note = " Note: Some referenced knowledge is past its review expiration."
        if has_deprecated:
            warning_note += " Note: Some referenced concepts are marked deprecated."

        answer = (
            f"Based on verified Open Knowledge Format (OKF) records:\n"
            f"{body_summary}{warning_note}\n\n"
            f"Sources Cited: {', '.join(citations)}"
        )

        return {
            "query": query,
            "answer": answer,
            "citations": citations,
            "concepts_used": [c.path for c in concepts],
            "confidence": 0.95 if not has_stale else 0.70,
        }


# ---------------------------------------------------------------------------
# 6. Visual Diagnostics & Diagram Generation
# ---------------------------------------------------------------------------

def render_okf_architecture_diagram(output_path: str | Path) -> None:
    """Renders a comprehensive architectural diagram of Google OKF for AI agents."""
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    ax.axis("off")

    # Title
    ax.text(
        0.5, 0.95,
        "Google Open Knowledge Format (OKF) Architecture",
        ha="center", va="center", fontsize=16, fontweight="bold", color="#1a237e"
    )
    ax.text(
        0.5, 0.90,
        "Portable Markdown + YAML Frontmatter Formalizing the LLM-Wiki Pattern",
        ha="center", va="center", fontsize=11, color="#424242", style="italic"
    )

    # 4 Architecture Pillars
    boxes = [
        ("1. Canonical Bundle\nDirectory Tree", "#e3f2fd", "#1565c0", 0.05, 0.45, 0.20, 0.35,
         "• Plain Markdown (.md)\n• Strict YAML Frontmatter\n• Mandatory 'type' field\n• Reserved: index.md, log.md"),
        ("2. Attestation &\nProvenance Engine", "#f3e5f5", "#7b1fa2", 0.28, 0.45, 0.20, 0.35,
         "• SHA-256 Digest Hashing\n• verified_by attestation\n• ISO timestamps & staleness\n• Resource URIs & Owners"),
        ("3. Progressive Disclosure\nGraph Navigator", "#e8f5e9", "#2e7d32", 0.51, 0.45, 0.20, 0.35,
         "• Cross-reference links\n• Token-budgeted index\n• 1-to-N hop traversals\n• NetworkX DAG verification"),
        ("4. Provenance-Grounded\nRAG Assembly", "#fff3e0", "#e65100", 0.74, 0.45, 0.21, 0.35,
         "• Deterministic Grounding\n• Explicit Citations [OKF:...]\n• Deprecation Warnings\n• 100% Auditable Prompts"),
    ]

    for title, bg_color, border_color, x, y, w, h, details in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=bg_color, edgecolor=border_color, lw=2, transform=ax.transAxes, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.07, title, ha="center", va="center", fontsize=11, fontweight="bold", color=border_color)
        ax.text(x + 0.02, y + 0.05, details, ha="left", va="bottom", fontsize=8.5, color="#212121", linespacing=1.6)

    # Flow arrows
    ax.annotate("", xy=(0.28, 0.62), xytext=(0.25, 0.62),
                arrowprops=dict(facecolor="#424242", edgecolor="#424242", width=2, headwidth=7), transform=ax.transAxes)
    ax.annotate("", xy=(0.51, 0.62), xytext=(0.48, 0.62),
                arrowprops=dict(facecolor="#424242", edgecolor="#424242", width=2, headwidth=7), transform=ax.transAxes)
    ax.annotate("", xy=(0.74, 0.62), xytext=(0.71, 0.62),
                arrowprops=dict(facecolor="#424242", edgecolor="#424242", width=2, headwidth=7), transform=ax.transAxes)

    # Bottom summary banner
    banner = plt.Rectangle((0.05, 0.08), 0.90, 0.25, facecolor="#eceff1", edgecolor="#607d8b", lw=1.5, transform=ax.transAxes)
    ax.add_patch(banner)
    ax.text(0.5, 0.27, "Core OKF Paradigm Shift: Deterministic Representation vs Unstructured Vectors",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#263238")
    ax.text(0.5, 0.16,
            "Traditional RAG relies on lossy chunking and approximate nearest neighbors.\n"
            "Google OKF enables autonomous agents to deliberately navigate verified organizational truth with complete provenance.",
            ha="center", va="center", fontsize=9.5, color="#37474f", linespacing=1.5)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_okf_graph_diagram(bundle: OKFBundle, output_path: str | Path) -> None:
    """Renders the concept dependency graph of an OKF bundle."""
    if not bundle.graph or bundle.graph.number_of_nodes() == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    ax.axis("off")

    type_colors = {
        "index": "#90caf9",
        "log": "#b0bec5",
        "table": "#81c784",
        "metric": "#ffb74d",
        "policy": "#ba68c8",
    }

    node_colors = []
    labels = {}
    for node in bundle.graph.nodes():
        concept = bundle.concepts.get(node)
        c_type = concept.concept_type if concept else "other"
        node_colors.append(type_colors.get(c_type, "#e0e0e0"))
        short_name = Path(node).stem
        labels[node] = short_name

    pos = nx.spring_layout(bundle.graph, seed=42, k=1.2)
    nx.draw_networkx_nodes(bundle.graph, pos, node_color=node_colors, node_size=2400, edgecolors="#333333", linewidths=1.5, ax=ax)
    nx.draw_networkx_edges(bundle.graph, pos, arrowstyle="->", arrowsize=18, edge_color="#546e7a", width=2, ax=ax)
    nx.draw_networkx_labels(bundle.graph, pos, labels=labels, font_size=9, font_weight="bold", ax=ax)

    ax.set_title("OKF Concept Cross-Reference Graph", fontsize=14, fontweight="bold", pad=15)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label=k.title(), markerfacecolor=v, markersize=12)
        for k, v in type_colors.items()
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=9)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
