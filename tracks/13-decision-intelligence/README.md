# Track 13 — Decision Intelligence with RAG

Combining evidence retrieval with structured decision workflows, quantitative risk models, economic optimization, and tamper-evident auditability.

---

## Overview

Traditional RAG pipelines stop at text generation: they retrieve documents, format context, and prompt an LLM to generate an answer. However, in mission-critical enterprise environments—such as credit facility underwriting, infrastructure procurement, pharmaceutical trial selection, and regulatory compliance—**retrieval is not a decision**.

Decision Intelligence (DI) is the engineering discipline that transforms probabilistic, unstructured evidence into provably sound, risk-bounded, economically optimal, and auditable outcomes.

```
       ┌──────────────────────┐
       │   Evidence Sources   │  (OKF Bundles, JEV Index, Vector DB)
       └──────────┬───────────┘
                  │ Semantic & Structured Retrieval
                  ▼
       ┌──────────────────────┐
       │  Extraction & Normal.│  (Pydantic Type Validation, Noul Guard)
       └──────────┬───────────┘
                  │ Quantitative Attributes & Evidence Provenance
                  ▼
       ┌──────────────────────┐
       │     Policy Gates     │  (Hard Compliance Boundary Checks)
       └──────────┬───────────┘
                  │ Admissible Candidates
                  ▼
       ┌──────────────────────┐
       │    MCDA & Ranking    │  (TOPSIS, WSM, AHP Weights)
       └──────────┬───────────┘
                  │ Relative Closeness & Multi-Objective Tradeoffs
                  ▼
       ┌──────────────────────┐
       │ Risk & Econ Modeling │  (Expected Loss, VaR, Utility, Knapsack)
       └──────────┬───────────┘
                  │ Risk-Adjusted Payoffs
                  ▼
       ┌──────────────────────┐
       │ Confidence Protocol  │  (Abstain / Escalate HITL / Decide)
       └──────────┬───────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │ Tamper-Evident Audit │  (SHA-256 Provenance Ledger)
       └──────────────────────┘
```

---

## Core Architecture Pillars

### 1. Information Retrieval vs Decision Making
- **Retrieval**: Surfaces candidate claims, entity disclosures, and quantitative attributes from unstructured documents, OKF knowledge bundles, and JEV indexes.
- **Decision Intelligence**: Evaluates trade-offs, enforces hard compliance boundaries, models quantitative risks, optimizes economic payoffs under budget constraints, and provides explainable, reproducible verdicts.

### 2. Multi-Criteria Decision Analysis (MCDA)
- **Weighted Sum Model (WSM)**: Normalizes benefit criteria ($\frac{x - \min}{\max - \min}$) and cost criteria ($\frac{\max - x}{\max - \min}$) into bounded utility scores.
- **TOPSIS**: Evaluates relative closeness $C_i = \frac{S_i^-}{S_i^+ + S_i^-}$ to positive ideal ($A^+$) and negative ideal ($A^-$) solution vectors in Euclidean hyperspace.
- **Analytic Hierarchy Process (AHP)**: Validates pairwise transitivity across evaluation criteria and verifies Consistency Ratio $CR \le 0.10$.

### 3. Quantitative Risk Scoring Engine
- **Expected Loss**: $E[L] = \sum P_k \times \text{Impact}_k$.
- **Uncertainty-Penalized Composite Risk**: Penalizes candidates when evidentiary backing has low confidence or missing citations:
  $$R_{\text{composite}} = E[L] \times (1 + \lambda \times (1 - C))$$
- **Value at Risk ($\text{VaR}_{95}$)**: Parametric and variance estimation of tail risk.

### 4. Economic Optimization Models
- **Expected Monetary Value (EMV)**: Payoff modeling across states of nature.
- **Von Neumann-Morgenstern Utility**: Models risk aversion ($\alpha > 0$) with concave utility functions to avoid catastrophic downside.
- **0-1 Knapsack Resource Allocation**: Optimizes candidate portfolio selection under strict budget capacity constraints.

### 5. Automated Policy Decision Trees & TypeSafe JEV
- Implements deterministic threshold gates (`>=`, `<=`, `==`, etc.).
- Integrates TypeSafe JEV sentinels: missing or unverified attributes yield `Noul(reason=...)`, preventing hallucinated passes or silent default bugs.

### 6. Confidence, Abstention & Escalation Protocol
- **ABSTAINED**: Triggered when evidence confidence $C < \theta_{\text{abstain}}$.
- **ESCALATED_HUMAN_REVIEW**: Triggered when margin between top candidates $\Delta < \epsilon_{\text{margin}}$ or risk exceeds risk ceiling.
- **DECIDED**: Clear optimal candidate with verified evidentiary backing.

### 7. Counterfactual Sensitivity & Tamper-Evident Audit Trail
- Computes weight and attribute perturbations required to flip winning alternatives.
- Emits cryptographic SHA-256 hash chains linking input criteria, extracted attributes, policy gate outcomes, and candidate rankings.

---

## Notebooks

| Notebook | Description | Status |
|---|---|---|
| [`01_decision_intelligence_with_rag.ipynb`](file:///Users/bharathm/Documents/Self%20Learning/RAG_from_scratch/tracks/13-decision-intelligence/01_decision_intelligence_with_rag.ipynb) | End-to-end Decision Intelligence covering MCDA, risk modeling, economic optimization, JEV policy gates, abstention protocols, and cryptographic audits. | **COMPLETED** |

---

## Shared Fixtures

- `fixtures/sample_decision_records/cloud_vendor_selection.json`: Real-world enterprise vector database and cloud infrastructure vendor selection problem with 4 candidate alternatives and 5 multi-objective criteria.
- `fixtures/sample_decision_records/lending_origination_decision.json`: Commercial credit facility origination dataset with corporate financial disclosures, risk indicators, and underwriting policy gates.

---

## Test Suite

Verified with 100% passing tests:
```bash
.venv312/bin/pytest tests/unit/test_decision_intelligence.py
```
