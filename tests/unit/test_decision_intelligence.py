"""tests/unit/test_decision_intelligence.py — Unit Tests for Track 13 Decision Intelligence.

Covers:
1. Multi-Criteria Decision Analysis: WSM, TOPSIS, and AHP consistency verification.
2. Quantitative Risk Scoring: Expected Loss, VaR, and evidence uncertainty penalty.
3. Economic Optimization: EMV, Von Neumann-Morgenstern utility, and 0-1 Knapsack.
4. Policy Decision Trees & JEV Integration: Gates, threshold operators, and Noul sentinel handling.
5. Confidence, Abstention & Human Escalation Protocol.
6. Counterfactual Sensitivity Analysis and Cryptographic Audit Fingerprints.
7. End-to-end Orchestrator execution with shared fixtures.
"""

import json
from pathlib import Path
import pytest
import numpy as np

from common.decision_intelligence import (
    Alternative,
    AutomatedPolicyEngine,
    ConfidenceAbstentionProtocol,
    Criterion,
    CriterionType,
    DecisionAuditRecord,
    DecisionRAGOrchestrator,
    DecisionSensitivityAnalyzer,
    DecisionStatus,
    EconomicOptimizer,
    MCDAEngine,
    PolicyGate,
    RiskScoringEngine,
    render_decision_intelligence_diagrams,
)
from common.typesafe_jev import Noul, Score

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "sample_decision_records"


# ==============================================================================
# 1. Multi-Criteria Decision Analysis (MCDA) Tests
# ==============================================================================

def test_mcda_weighted_sum_model_benefit_and_cost():
    """Validates WSM correctly maximizes benefit criteria and minimizes cost criteria."""
    criteria = [
        Criterion(name="throughput", weight=0.6, criterion_type=CriterionType.BENEFIT),
        Criterion(name="cost", weight=0.4, criterion_type=CriterionType.COST),
    ]

    alts = [
        Alternative(id="opt_a", name="Option A", attributes={"throughput": 100.0, "cost": 20.0}),
        Alternative(id="opt_b", name="Option B", attributes={"throughput": 50.0, "cost": 10.0}),
    ]

    res = MCDAEngine.weighted_sum_model(alts, criteria)
    rankings = res["rankings"]
    scores = res["scores"]

    assert len(rankings) == 2
    # Option A has max throughput (norm=1.0) and max cost (norm=0.0) -> 0.6 * 1.0 + 0.4 * 0.0 = 0.60
    # Option B has min throughput (norm=0.0) and min cost (norm=1.0) -> 0.6 * 0.0 + 0.4 * 1.0 = 0.40
    assert rankings[0]["alternative_id"] == "opt_a"
    assert pytest.approx(scores["opt_a"]["wsm_total_score"], abs=1e-3) == 0.60
    assert pytest.approx(scores["opt_b"]["wsm_total_score"], abs=1e-3) == 0.40


def test_mcda_topsis_vector_normalization_and_relative_closeness():
    """Validates TOPSIS relative closeness formula and positive/negative ideal solutions."""
    criteria = [
        Criterion(name="speed", weight=0.5, criterion_type=CriterionType.BENEFIT),
        Criterion(name="latency", weight=0.5, criterion_type=CriterionType.COST),
    ]

    alts = [
        Alternative(id="fast", name="Fast System", attributes={"speed": 90.0, "latency": 10.0}),
        Alternative(id="slow", name="Slow System", attributes={"speed": 30.0, "latency": 50.0}),
    ]

    res = MCDAEngine.topsis(alts, criteria)
    rankings = res["rankings"]

    assert len(rankings) == 2
    assert rankings[0]["alternative_id"] == "fast"
    # Fast alternative should have significantly higher relative closeness C_i
    assert rankings[0]["score"] > rankings[1]["score"]
    assert 0.0 <= rankings[0]["score"] <= 1.0
    assert 0.0 <= rankings[1]["score"] <= 1.0


def test_mcda_ahp_consistency_ratio_check():
    """Tests Analytic Hierarchy Process pairwise matrix consistency verification."""
    # A perfectly consistent 3x3 matrix
    # Criteria: A, B, C. A is twice as important as B, B is twice as important as C -> A is 4x as important as C.
    pairwise = [
        [1.0, 2.0, 4.0],
        [0.5, 1.0, 2.0],
        [0.25, 0.5, 1.0],
    ]
    names = ["A", "B", "C"]

    ahp_res = MCDAEngine.analytic_hierarchy_process(pairwise, names)
    assert ahp_res["is_consistent"] is True
    assert ahp_res["consistency_ratio"] <= 0.10
    assert pytest.approx(sum(ahp_res["weights"].values()), abs=1e-4) == 1.0
    assert ahp_res["weights"]["A"] > ahp_res["weights"]["B"] > ahp_res["weights"]["C"]


# ==============================================================================
# 2. Quantitative Risk Scoring Engine Tests
# ==============================================================================

def test_risk_scoring_expected_loss_and_var():
    """Tests expected loss computation and parametric VaR bounds."""
    risk_factors = [
        {"name": "server_outage", "probability": 0.05, "impact": 100.0},
        {"name": "minor_latency_spike", "probability": 0.20, "impact": 10.0},
    ]

    loss = RiskScoringEngine.calculate_expected_loss(risk_factors)
    # Expected: (0.05 * 100) + (0.20 * 10) = 5.0 + 2.0 = 7.0
    assert pytest.approx(loss, abs=1e-3) == 7.0

    comp_risk = RiskScoringEngine.calculate_composite_risk(
        risk_factors,
        evidence_confidence=0.50,  # 50% confidence should penalize risk
        uncertainty_penalty_factor=0.50
    )
    # Multiplier: 1.0 + 0.5 * (1 - 0.5) = 1.25 -> 7.0 * 1.25 = 8.75
    assert pytest.approx(comp_risk["penalized_risk_score"], abs=1e-3) == 8.75
    assert comp_risk["value_at_risk_95"] >= comp_risk["expected_loss"]


# ==============================================================================
# 3. Economic Optimization Models Tests
# ==============================================================================

def test_economic_optimizer_emv_and_utility():
    """Validates EMV and Von Neumann-Morgenstern utility risk preference curves."""
    payoffs = {"bull_market": 50000.0, "bear_market": -10000.0}
    probs = {"bull_market": 0.70, "bear_market": 0.30}

    emv = EconomicOptimizer.expected_monetary_value(payoffs, probs)
    # (0.7 * 50000) + (0.3 * -10000) = 35000 - 3000 = 32000
    assert pytest.approx(emv, abs=1e-2) == 32000.0

    # Risk-neutral utility (alpha = 0) equals raw payoff
    u_neutral = EconomicOptimizer.von_neumann_utility(1000.0, risk_tolerance_alpha=0.0)
    assert u_neutral == 1000.0

    # Risk-averse concave utility produces diminishing marginal returns
    u_averse = EconomicOptimizer.von_neumann_utility(1000.0, risk_tolerance_alpha=0.01)
    assert u_averse > 0


def test_economic_optimizer_knapsack_resource_allocation():
    """Tests 0-1 Knapsack resource constrained project portfolio selection."""
    items = [
        {"id": "proj_1", "name": "Vector Index Cache", "cost": 20.0, "value": 40.0},
        {"id": "proj_2", "name": "Semantic Reranker", "cost": 30.0, "value": 50.0},
        {"id": "proj_3", "name": "Graph Ingestion", "cost": 40.0, "value": 70.0},
    ]
    budget = 50.0  # Cap allows proj_1 (20) + proj_2 (30) = cost 50, val 90; OR proj_3 (40) = cost 40, val 70.

    res = EconomicOptimizer.knapsack_portfolio_selection(items, budget_capacity=budget)
    assert res["total_cost"] <= budget
    assert "proj_1" in res["selected_ids"]
    assert "proj_2" in res["selected_ids"]
    assert pytest.approx(res["total_value"], abs=1e-2) == 90.0


# ==============================================================================
# 4. Policy Decision Gates & TypeSafe JEV Tests
# ==============================================================================

def test_policy_gate_evaluations_and_noul_rejection():
    """Validates PolicyGate logic and ensures Noul sentinels fail gates deterministically."""
    gate = PolicyGate(
        name="min_sla_gate",
        criterion="sla_uptime",
        operator=">=",
        threshold=99.9,
        rejection_action=DecisionStatus.REJECTED
    )

    passed, _ = gate.evaluate(99.95)
    assert passed is True

    failed, _ = gate.evaluate(99.50)
    assert failed is False

    # Sentinel test: Noul must cleanly fail gate rather than throw exception
    noul_passed, noul_msg = gate.evaluate(Noul(reason="Missing SLA in vendor contract"))
    assert noul_passed is False
    assert "Noul" in noul_msg


def test_automated_policy_engine_escalates_on_threshold():
    """Tests AutomatedPolicyEngine flags escalation when an escalating gate fails."""
    gates = [
        PolicyGate(
            name="security_check",
            criterion="security_score",
            operator=">=",
            threshold=85.0,
            rejection_action=DecisionStatus.ESCALATED_HUMAN_REVIEW
        )
    ]
    engine = AutomatedPolicyEngine(gates)
    alt = Alternative(id="alt_1", name="Marginal Sec Alt", attributes={"security_score": 80.0})

    eval_res = engine.evaluate_alternative(alt)
    assert eval_res["is_admissible"] is False
    assert eval_res["recommended_action"] == DecisionStatus.ESCALATED_HUMAN_REVIEW


# ==============================================================================
# 5. Confidence, Abstention & Human Escalation Protocol Tests
# ==============================================================================

def test_confidence_abstention_triggers():
    """Validates the protocol triggers abstention on low confidence and escalation on ambiguity."""
    alt_low_conf = Alternative(id="alt_low", name="Low Conf Alt", confidence=0.55)
    rankings = [{"alternative_id": "alt_low", "score": 0.90}]

    status, reason, winner = ConfidenceAbstentionProtocol.assess_decision(
        top_rankings=rankings,
        admissible_alternatives=[alt_low_conf],
        min_evidence_confidence=0.70
    )
    assert status == DecisionStatus.ABSTAINED
    assert "Abstention triggered" in reason

    # Ambiguity test: top 2 candidates within 0.01 score margin
    alt1 = Alternative(id="cand_1", name="Cand 1", confidence=0.95)
    alt2 = Alternative(id="cand_2", name="Cand 2", confidence=0.95)
    ambiguous_rankings = [
        {"alternative_id": "cand_1", "score": 0.852},
        {"alternative_id": "cand_2", "score": 0.849},
    ]

    status_ambig, reason_ambig, _ = ConfidenceAbstentionProtocol.assess_decision(
        top_rankings=ambiguous_rankings,
        admissible_alternatives=[alt1, alt2],
        ambiguity_margin_threshold=0.02
    )
    assert status_ambig == DecisionStatus.ESCALATED_HUMAN_REVIEW
    assert "Human escalation triggered" in reason_ambig


# ==============================================================================
# 6. Counterfactual Sensitivity & Audit Ledger Tests
# ==============================================================================

def test_counterfactual_sensitivity_analysis():
    """Tests that sensitivity analysis detects critical criteria weight perturbations."""
    criteria = [
        Criterion(name="performance", weight=0.6, criterion_type=CriterionType.BENEFIT),
        Criterion(name="cost", weight=0.4, criterion_type=CriterionType.COST),
    ]
    alts = [
        Alternative(id="opt1", name="Option 1", attributes={"performance": 95.0, "cost": 90.0}),
        Alternative(id="opt2", name="Option 2", attributes={"performance": 85.0, "cost": 20.0}),
    ]

    sens = DecisionSensitivityAnalyzer.analyze_weight_sensitivity(alts, criteria, perturbation_step=0.1)
    assert "base_winner" in sens
    assert "flips" in sens
    assert 0.0 <= sens["robustness_score"] <= 1.0


def test_immutable_audit_record_fingerprint():
    """Tests that DecisionAuditRecord generates a deterministic SHA-256 fingerprint."""
    record = DecisionAuditRecord(
        decision_id="dec_test_123",
        timestamp="2026-11-08T00:00:00Z",
        objective="Select vendor",
        status=DecisionStatus.DECIDED,
        selected_alternative_id="vendor_alpha",
        rankings=[{"rank": 1, "alternative_id": "vendor_alpha", "score": 0.85}],
        scores={},
        risk_assessment={},
        economic_metrics={},
        policy_audit_trail=[],
        counterfactual_sensitivity={},
    )
    fp1 = record.compute_fingerprint()
    assert len(fp1) == 64  # SHA-256 hex length

    # Tampering test: altering selected alternative alters fingerprint
    record.selected_alternative_id = "vendor_hacked"
    fp2 = record.compute_fingerprint()
    assert fp1 != fp2


# ==============================================================================
# 7. End-to-End Orchestrator with Shared Fixtures Tests
# ==============================================================================

def test_end_to_end_cloud_vendor_selection_fixture():
    """Runs end-to-end DecisionRAGOrchestrator on cloud_vendor_selection.json fixture."""
    fixture_path = FIXTURES_DIR / "cloud_vendor_selection.json"
    assert fixture_path.exists(), f"Missing fixture at {fixture_path}"

    with open(fixture_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    criteria = [Criterion(**c) for c in data["criteria"]]
    policy_gates = [PolicyGate(**pg) for pg in data["policy_gates"]]
    alternatives = [Alternative(**a) for a in data["alternatives"]]

    orchestrator = DecisionRAGOrchestrator(
        criteria=criteria,
        policy_gates=policy_gates,
        budget_constraint=data.get("budget_constraint_usd")
    )

    audit_rec = orchestrator.solve(
        objective=data["objective"],
        alternatives=alternatives,
        min_evidence_confidence=0.75,
        ambiguity_margin=0.03,
    )

    # Vendor gamma should be rejected or escalated by security_compliance_score gate (< 80)
    gamma_eval = next((t for t in audit_rec.policy_audit_trail if t["alternative_id"] == "vendor_gamma"), None)
    assert gamma_eval is not None
    assert gamma_eval["is_admissible"] is False

    # Audit record should have valid provenance hash
    assert audit_rec.provenance_hash != ""
    assert len(audit_rec.provenance_hash) == 64
    assert audit_rec.status in (DecisionStatus.DECIDED, DecisionStatus.ESCALATED_HUMAN_REVIEW)


def test_render_diagrams_generation(tmp_path):
    """Verifies that render_decision_intelligence_diagrams outputs png files."""
    files = render_decision_intelligence_diagrams(tmp_path)
    assert len(files) == 2
    for file_path in files:
        p = Path(file_path)
        assert p.exists()
        assert p.stat().st_size > 1000
