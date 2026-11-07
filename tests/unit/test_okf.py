"""Unit tests for Google Open Knowledge Format (OKF) Engine.

Validates:
1. Canonical OKF YAML frontmatter parsing and mandatory 'type' enforcement.
2. Link extraction, path normalization, and SHA-256 tamper-evident content hashing.
3. Serialization round-tripping.
4. Bundle discovery, reserved file handling (index.md, log.md), and NetworkX DiGraph creation.
5. Progressive disclosure API for token-budgeted AI agent context management.
6. Graph neighborhood traversal and dependency discovery.
7. Bundle validation, broken link detection, staleness checks, and hash verification.
8. OKF-augmented RAG retrieval (semantic + graph expansion) and grounded synthesis with citations.
9. Visual diagram generation.
"""

from datetime import datetime, timezone
from pathlib import Path
import pytest

from common.okf import (
    OKFBundle,
    OKFConcept,
    OKFContextAssembler,
    OKFParser,
    OKFProvenance,
    OKFRAGRetriever,
    OKFValidator,
    render_okf_architecture_diagram,
    render_okf_graph_diagram,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "sample_okf_bundle"


def test_okf_concept_parsing_valid():
    sample_text = """---
type: table
title: "Inventory Stock"
description: "Current on-hand warehouse inventory counts."
resource: "postgres://prod/warehouse/stock"
tags: ["warehouse", "logistics"]
owner: "supply-chain"
status: "active"
generated:
  by: "etl-sync-job"
  at: "2026-06-01T08:00:00Z"
  verified_by: "warehouse-lead"
  confidence: 0.99
---

# Inventory Stock
Tracks item availability.
Related: [Customer Orders](/tables/orders.md)
"""
    concept = OKFParser.parse_text(sample_text, bundle_relative_path="/tables/inventory.md")

    assert concept.path == "/tables/inventory.md"
    assert concept.concept_type == "table"
    assert concept.title == "Inventory Stock"
    assert concept.description == "Current on-hand warehouse inventory counts."
    assert concept.resource == "postgres://prod/warehouse/stock"
    assert "warehouse" in concept.tags
    assert concept.owner == "supply-chain"
    assert concept.status == "active"
    assert concept.provenance is not None
    assert concept.provenance.by == "etl-sync-job"
    assert concept.provenance.verified_by == "warehouse-lead"
    assert concept.provenance.confidence == 0.99
    assert concept.outbound_links == ["/tables/orders.md"]
    assert len(concept.content_sha256) == 64


def test_okf_concept_mandatory_type_violation():
    invalid_text = """---
title: "Untyped Concept"
description: "Missing required type property"
---

Some content
"""
    with pytest.raises(ValueError, match="Missing mandatory field 'type'"):
        OKFParser.parse_text(invalid_text, bundle_relative_path="/docs/untyped.md")


def test_okf_serializer_roundtrip():
    original = OKFConcept(
        path="/metrics/churn_rate.md",
        concept_type="metric",
        title="Customer Churn Rate",
        description="Percentage of active subscriptions cancelled within 30 days.",
        resource="looker://models/finance/churn",
        tags=["kpi", "retention"],
        owner="growth-team",
        status="active",
        body="Churn rate formula:\n$$\\text{Churn} = \\frac{\\text{Lost}}{\\text{Total}}$$",
        outbound_links=[],
    )

    serialized_md = OKFParser.serialize_concept(original)
    assert serialized_md.startswith("---\n")
    assert "type: metric" in serialized_md

    reparsed = OKFParser.parse_text(serialized_md, bundle_relative_path="/metrics/churn_rate.md")
    assert reparsed.concept_type == "metric"
    assert reparsed.title == "Customer Churn Rate"
    assert reparsed.description == original.description
    assert reparsed.resource == original.resource


def test_okf_bundle_discovery_and_indexing():
    bundle = OKFBundle.from_directory(FIXTURES_DIR)

    assert len(bundle.concepts) >= 6
    assert "/index.md" in bundle.concepts
    assert "/log.md" in bundle.concepts
    assert "/tables/orders.md" in bundle.concepts
    assert "/tables/customers.md" in bundle.concepts
    assert "/metrics/gross_margin.md" in bundle.concepts
    assert "/policies/refund_sla.md" in bundle.concepts

    assert bundle.index_concept is not None
    assert bundle.index_concept.concept_type == "index"
    assert bundle.log_concept is not None
    assert bundle.log_concept.concept_type == "log"


def test_okf_bundle_graph_metrics_and_lineage():
    bundle = OKFBundle.from_directory(FIXTURES_DIR)
    metrics = bundle.get_graph_metrics()

    assert metrics["nodes"] >= 6
    assert metrics["edges"] >= 5
    assert metrics["density"] > 0.0
    assert "pagerank" in metrics
    assert len(metrics["pagerank"]) == metrics["nodes"]

    # Inbound / Outbound relationships
    orders_deps = bundle.find_dependencies("/tables/orders.md")
    assert any(c.path == "/tables/customers.md" for c in orders_deps)

    orders_dependents = bundle.find_dependents("/tables/orders.md")
    assert any(c.path == "/metrics/gross_margin.md" for c in orders_dependents)


def test_okf_progressive_disclosure_summary():
    bundle = OKFBundle.from_directory(FIXTURES_DIR)
    prog_index = bundle.get_progressive_index()

    # index.md and log.md are excluded from concept summary to save tokens
    assert all(item["path"] not in ("/index.md", "/log.md") for item in prog_index)

    # Check key columns present
    for item in prog_index:
        assert "path" in item
        assert "type" in item
        assert "title" in item
        assert "description" in item


def test_okf_graph_neighborhood_traversal():
    bundle = OKFBundle.from_directory(FIXTURES_DIR)
    neighborhood = bundle.traverse_neighborhood("/metrics/gross_margin.md", max_hops=1)
    paths = [c.path for c in neighborhood]

    assert "/metrics/gross_margin.md" in paths
    assert "/tables/orders.md" in paths
    assert "/policies/refund_sla.md" in paths


def test_okf_validator_clean_bundle():
    bundle = OKFBundle.from_directory(FIXTURES_DIR)
    report = OKFValidator.validate_bundle(bundle)

    assert report.is_valid is True
    assert report.error_count == 0
    assert len(report.broken_links) == 0
    assert report.total_files >= 6
    assert "table" in report.concepts_by_type
    assert "metric" in report.concepts_by_type
    assert "policy" in report.concepts_by_type


def test_okf_validator_detects_broken_link_and_stale_concept():
    bundle = OKFBundle()
    stale_concept = OKFConcept(
        path="/metrics/old_kpi.md",
        concept_type="metric",
        title="Old KPI",
        stale_after="2020-01-01T00:00:00Z",
        outbound_links=["/tables/non_existent.md"],
        body="Links to [NonExistent](/tables/non_existent.md)",
    )
    bundle.add_concept(stale_concept)
    bundle.build_graph()

    report = OKFValidator.validate_bundle(
        bundle, reference_time=datetime(2026, 6, 1, tzinfo=timezone.utc)
    )

    assert report.is_valid is False
    assert report.error_count >= 1
    assert any(issue.code == "BROKEN_LINK" for issue in report.issues)
    assert any(issue.code == "STALE_CONCEPT" for issue in report.issues)


def test_okf_validator_tamper_detection():
    bundle = OKFBundle()
    concept = OKFConcept(
        path="/tables/tampered.md",
        concept_type="table",
        title="Tampered Table",
        body="Altered body content after attestation signature.",
        provenance=OKFProvenance(
            by="auditor",
            content_hash="0000000000000000000000000000000000000000000000000000000000000000",
        ),
        content_sha256=OKFParser.compute_sha256("Altered body content after attestation signature."),
    )
    bundle.add_concept(concept)
    bundle.build_graph()

    report = OKFValidator.validate_bundle(bundle)
    assert report.is_valid is False
    assert any(issue.code == "HASH_MISMATCH" for issue in report.issues)


def test_okf_rag_retriever_semantic_and_graph_expansion():
    bundle = OKFBundle.from_directory(FIXTURES_DIR)
    retriever = OKFRAGRetriever(bundle)

    # Query targeting gross margin
    retrieved = retriever.retrieve("gross margin formula profit sales", top_k=1, expand_graph_hops=1)
    retrieved_paths = [c.path for c in retrieved]

    assert "/metrics/gross_margin.md" in retrieved_paths
    # Graph expansion should pull in upstream order dependencies
    assert "/tables/orders.md" in retrieved_paths or "/policies/refund_sla.md" in retrieved_paths


def test_okf_context_assembler_provenance_citations():
    bundle = OKFBundle.from_directory(FIXTURES_DIR)
    concepts = [
        bundle.get_concept("/tables/orders.md"),
        bundle.get_concept("/metrics/gross_margin.md"),
    ]
    prompt_context = OKFContextAssembler.assemble_context(concepts)

    assert "### [OKF:/tables/orders.md]" in prompt_context
    assert "### [OKF:/metrics/gross_margin.md]" in prompt_context
    assert "Authored by: data-lineage-agent/v2" in prompt_context
    assert "bigquery://prod-data-platform/sales/orders" in prompt_context

    result = OKFContextAssembler.generate_grounded_answer("What is the gross margin formula?", concepts)
    assert len(result["citations"]) == 2
    assert "[OKF:/tables/orders.md]" in result["citations"]
    assert "[OKF:/metrics/gross_margin.md]" in result["citations"]
    assert result["confidence"] > 0.8


def test_okf_diagram_generation(tmp_path):
    arch_path = tmp_path / "test_okf_arch.png"
    graph_path = tmp_path / "test_okf_graph.png"

    render_okf_architecture_diagram(arch_path)
    bundle = OKFBundle.from_directory(FIXTURES_DIR)
    render_okf_graph_diagram(bundle, graph_path)

    assert arch_path.exists()
    assert arch_path.stat().st_size > 1000
    assert graph_path.exists()
    assert graph_path.stat().st_size > 1000
