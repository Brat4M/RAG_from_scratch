# Track 14 — Enterprise RAG: Multi-Tenancy, Fine-Grained ACL, Security Guardrails & Operational Scaling

Production-grade architectures for multi-tenant isolation, document-level access control lists (ACL), KMS envelope encryption, security guardrails, two-tier semantic caching, asynchronous batch ingestion, zero-downtime blue-green migration, and immutable cryptographic audit trails.

---

## Overview

Transitioning a retrieval-augmented generation prototype into a regulated enterprise platform demands far more than basic vector lookup. Mission-critical deployments in banking, healthcare, legal, and multi-tenant SaaS must enforce non-negotiable security boundaries, prevent cross-tenant data leakage, defend against indirect prompt injections, encrypt sensitive payloads at rest, and maintain tamper-evident audit records for compliance standards (SOC 2 Type II, HIPAA, ISO 27001).

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                       Enterprise Query & User Identity                      │
 │       (User ID, Tenant ID, Roles, Groups, Clearance: 0-Public to 3-Restricted)     │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                         Security Guardrails Engine                          │
 │         (Prompt Injection Defense, PII Masking: SSN, Credit Card, Keys)     │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                        Two-Tier Semantic Cache                              │
 │   Tier 1: Exact SHA-256 Hash  |  Tier 2: Vector Cosine Similarity (τ ≥ 0.90) │
 └───────────────────┬─────────────────────────────────────┬───────────────────┘
                     │ (Hit: <1ms, 0 tokens)               │ (Miss)
                     ▼                                     ▼
        ┌─────────────────────────┐       ┌────────────────────────────────────┐
        │ Fast Synthesized Return │       │   Policy Pushdown Pre-Filtering    │
        └─────────────────────────┘       │   (Strict Tenant Boundary & ACLs)  │
                                          └─────────────────┬──────────────────┘
                                                            │
                                                            ▼
                                          ┌────────────────────────────────────┐
                                          │      Vector Similarity Search      │
                                          │     (HNSW Filtered Graph Traversal)│
                                          └─────────────────┬──────────────────┘
                                                            │
                                                            ▼
                                          ┌────────────────────────────────────┐
                                          │     KMS Payload Decryption         │
                                          │  (Per-Tenant DEK & Master HSM KEK) │
                                          └─────────────────┬──────────────────┘
                                                            │
                                                            ▼
                                          ┌────────────────────────────────────┐
                                          │     Cryptographic Audit Ledger     │
                                          │  (SHA-256 Chained Non-Repudiation) │
                                          └────────────────────────────────────┘
```

---

## Notebooks

| Notebook | Focus Areas | Key Implementations |
|---|---|---|
| [`01_enterprise_rag_architecture.ipynb`](./01_enterprise_rag_architecture.ipynb) | Multi-Tenancy, ACL, Security Guardrails, KMS, Caching, Ingestion, Blue-Green Migration, Auditing | `ACLEvaluator`, `SecurityGuardrails`, `EnvelopeEncryptionKMS`, `SemanticCache`, `AsyncIngestionPipeline`, `ModelMigrationManager`, `EnterpriseAuditLogger`, `EnterpriseRAGPipeline` |

---

## Architectural Invariants

### 1. Hierarchical Access Control Policy Precedence
To ensure security without privilege escalation bugs, access control evaluation enforces strict precedence:
$$\text{Tenant Isolation} \rightarrow \text{Explicit Deny} \rightarrow \text{Owner/Admin Override} \rightarrow \text{Clearance Threshold} \rightarrow \text{Explicit Allow} \rightarrow \text{Role/Group Intersection} \rightarrow \text{Default Deny}$$

### 2. Pre-Filtering vs. Post-Filtering Trade-Offs
- **Pre-Filtering (Pushdown Predicates)**: Evaluates user authorization inside the index before candidate selection. Guarantees that $100\%$ of retrieved candidates are authorized, avoiding result starvation.
- **Post-Filtering (Fetch & Scrub)**: Retrieves global top-$N$ candidates and removes unauthorized chunks at runtime. Highly vulnerable to starvation when the highest-scoring vectors belong to restricted documents.

### 3. KMS Envelope Encryption
- Master Key Encryption Key (KEK) is stored securely in an HSM/KMS.
- Per-tenant Data Encryption Keys (DEKs) are wrapped by the KEK.
- All stored vector document payloads are encrypted with the tenant's DEK with HMAC-SHA256 integrity verification.

### 4. Two-Tier Semantic Caching
- **L1 Cache**: Exact SHA-256 match on `(tenant_id, clearance_level, normalized_query)`.
- **L2 Cache**: Cosine similarity $\ge \tau$ (e.g. $0.90$) within the same tenant and exact clearance tier.
- Yields up to $95\%$ latency reduction and saves $200\text{--}500$ tokens per query.

### 5. Blue-Green Model Migration State Machine
Upgrades embedding models with zero customer downtime:
$$\text{IDLE} \xrightarrow{\text{start}} \text{SHADOWING} \xrightarrow{\text{canary}} \text{CANARY } (p\%) \xrightarrow{\text{validate}} \text{CUTOVER} \xrightarrow{\text{complete}} \text{COMPLETED}$$

### 6. Chained Cryptographic Audit Ledger
Maintains an append-only, tamper-evident ledger for non-repudiation:
$$\text{Record Hash}_i = \text{SHA-256}(i \parallel \text{Timestamp} \parallel \text{Tenant} \parallel \text{User} \parallel \text{QueryHash} \parallel \text{Chunks} \parallel \text{Record Hash}_{i-1})$$

---

## Running the Unit Tests

```bash
.venv312/bin/pytest tests/unit/test_enterprise_rag.py -v
```
