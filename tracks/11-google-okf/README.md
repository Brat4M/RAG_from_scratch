# Track 11 — Google Open Knowledge Format (OKF)

Mastering verified OKF specifications, progressive disclosure catalogs, knowledge graph dependency navigation, and provenance-grounded RAG.

---

## Overview

In enterprise AI engineering, autonomous agents face the **context-assembly crisis**:
1. **Loss of Structure & Formulas:** Unstructured text chunking arbitrarily chops database schemas, revenue formulas, and compliance policies, destroying critical semantic boundaries.
2. **Loss of Provenance & Trust:** Standard vector databases do not track who authored a definition, which auditor verified it, when it expires, or what physical system it maps to.
3. **Context Window Flooding:** Ingesting whole enterprise wikis exhausts token limits and triggers "lost-in-the-middle" attention degradation.

**Google Open Knowledge Format (OKF)**—introduced by Google Cloud (originally authored by Sam McVeety and Amir Hormati)—formalizes the **LLM-Wiki pattern** into a portable, vendor-neutral file standard:
- **Plain Markdown (`.md`):** Human-readable, version-controlled with Git, zero proprietary database lock-in.
- **YAML Frontmatter:** Machine-readable metadata anchored at the top of each file. Per the canonical Google specification, **only `type` is strictly required**.
- **Progressive Disclosure:** Reserved directory catalogs (`index.md`) and lineage history (`log.md`) allow agents to iteratively explore knowledge hierarchies with token-budgeted precision before loading heavy document bodies.
- **Interconnected Knowledge Graph:** Standard Markdown hyperlinks (`[Customer Orders](/tables/orders.md)`) create directed semantic edges across tables, metrics, and policies without requiring a standalone graph database.

---

## Key Architectures Implemented

### 1. Canonical OKF Document Model & Parser
- Enforces the official Google OKF rule: `type` is the single mandatory frontmatter attribute.
- Extracts standard metadata: `title`, `description`, `resource` (URI), `tags`, `owner`, `status`, and `stale_after`.
- Computes tamper-evident SHA-256 digests (`content_sha256`) of the markdown body for attestation auditing.
- Resolves bundle-relative (`/tables/orders.md`) and document-relative links to canonical root paths.

### 2. NetworkX Directed Concept Graph
- Converts cross-reference markdown links into a directed knowledge graph.
- Calculates topological and centrality metrics: density, in/out degree, PageRank, and DAG status.
- Enables 1-to-N hop neighborhood traversals (`traverse_neighborhood`) and dependency lineage analysis.

### 3. Automated Validation & Bundle Linting (`OKFValidator`)
- **Missing Required Fields:** Audits for missing `type` property.
- **Broken Link Detection:** Identifies links pointing to non-existent concept files.
- **Staleness Tracking:** Alerts when a concept exceeds its `stale_after` expiration timestamp.
- **Tamper Detection:** Flags mismatches between attested hashes and real body digests.
- **Orphan Detection:** Identifies isolated concepts not linked by any other document or index.

### 4. Progressive Disclosure & Context Assembly for RAG
- **Progressive Index (`get_progressive_index`):** Compact summary table providing high-level semantic descriptions to AI agents with minimal token overhead.
- **Hybrid Retrieval (`OKFRAGRetriever`):** Combines continuous semantic embeddings with deterministic graph neighborhood expansion.
- **Provenance-Augmented Prompts (`OKFContextAssembler`):** Generates structured context blocks carrying author identity, verification status, and explicit citation tags (`[OKF:/tables/orders.md]`).

---

## Notebooks

| Notebook | Focus | Key Topics |
| :--- | :--- | :--- |
| [`01_google_open_knowledge_format.ipynb`](./01_google_open_knowledge_format.ipynb) | Google OKF Architecture & Grounded RAG | YAML frontmatter parsing, mandatory type enforcement, NetworkX dependency graphs, progressive disclosure, bundle linting, provenance citations, and live Groq LLM integration |

---

## Architectural Diagrams

- [`track11_okf_architecture.png`](../../assets/diagrams/track11_okf_architecture.png): Four pillars of Google OKF (Canonical Bundle, Attestation & Provenance, Progressive Disclosure, and Grounded RAG).
- [`track11_okf_knowledge_graph.png`](../../assets/diagrams/track11_okf_knowledge_graph.png): Directed concept dependency network showing cross-references between tables, metrics, and compliance policies.

---

## Verification & Testing

Track 11 is covered by comprehensive unit tests in [`tests/unit/test_okf.py`](../../tests/unit/test_okf.py):
- Parsing and YAML frontmatter validation.
- Mandatory `type` specification violation checking.
- Serialization round-trip integrity.
- Bundle discovery and reserved file handling (`index.md`, `log.md`).
- Graph metrics, topological sorting, and neighborhood traversal.
- Progressive disclosure summary generation.
- Bundle validation, broken link detection, staleness checks, and tamper detection.
- RAG retrieval (semantic + graph expansion) and provenance citation generation.
- Diagram rendering.

To execute the test suite:
```bash
.venv312/bin/pytest tests/unit/test_okf.py
```
