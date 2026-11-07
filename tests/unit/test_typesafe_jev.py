"""Unit tests for Track 12 — TypeSafe JEV Decision Engine and JSON-Element-Vector Indexing.

Tests cover:
1. JEVPath parsing, JSON pointer, dot notation, and wildcard pattern matching.
2. JEVElement structural representation and metadata.
3. JEVIndexer flattening of nested JSON hierarchies and dense embedding generation.
4. JEVSearchEngine semantic vector search, path filtering, and predicate matching.
5. TypeSafe decision primitives: Choice[T], Score calibration, Noul abstention sentinel, is_noul guard.
6. DecisionGate invariant verification: thresholds, evidence coverage, forbidden risk flags, latency tracking.
7. StructuredOutputExtractor markdown stripping and Pydantic V2 schema validation.
8. SelfRepairLoop automated heuristic repair (type coercion, number normalization, list wrapping).
9. TypeSafeJEVUnderwritingEngine end-to-end evaluation with realistic fixtures.
10. Head-to-head benchmark: naive conditional logic vs TypeSafe JEV crash resilience.
"""

from pathlib import Path
import pytest
from pydantic import BaseModel, Field

from common.typesafe_jev import (
    Choice,
    DecisionGate,
    DecisionOutcome,
    GateResult,
    JEVElement,
    JEVIndexer,
    JEVPath,
    JEVSearchEngine,
    LoanApplicantProfile,
    LoanApplicationRecord,
    Noul,
    Score,
    SelfRepairLoop,
    StructuredOutputExtractor,
    TypeSafeJEVUnderwritingEngine,
    compare_conditional_vs_jev,
    is_noul,
    naive_conditional_underwriting,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "sample_jev_records"


def test_jev_path_representations_and_matching():
    """Verify JEVPath dot notation, JSON pointer, and wildcard pattern matching."""
    p1 = JEVPath.from_string("applicant.financials.revenue")
    assert p1.to_dot_notation() == "applicant.financials.revenue"
    assert p1.to_json_pointer() == "/applicant/financials/revenue"
    assert p1.matches_pattern("applicant.*.revenue")
    assert p1.matches_pattern("applicant.*")
    assert p1.matches_pattern("*")
    assert not p1.matches_pattern("policies.*")

    p2 = JEVPath.from_string("orders[0].items[2].sku")
    assert p2.to_dot_notation() == "orders.0.items.2.sku"
    assert p2.to_json_pointer() == "/orders/0/items/2/sku"


def test_jev_indexer_and_element_flattening():
    """Verify JEVIndexer flattens complex nested dictionary and creates dense embeddings."""
    indexer = JEVIndexer()
    sample_data = {
        "company": "Apex Corp",
        "founded": 2018,
        "active": True,
        "metrics": {
            "arr_usd": 1500000.0,
            "headcount": 24,
        },
        "tags": ["saas", "b2b", "ai"],
    }
    doc = indexer.index_document(doc_id="test_doc", data=sample_data)

    assert doc.doc_id == "test_doc"
    assert len(doc.elements) >= 7

    # Check leaf elements
    path_map = {e.path: e for e in doc.elements}
    assert "company" in path_map
    assert path_map["company"].value == "Apex Corp"
    assert path_map["company"].element_type == "string"
    assert path_map["company"].vector is not None
    assert len(path_map["company"].vector) == 64

    assert "metrics.arr_usd" in path_map
    assert path_map["metrics.arr_usd"].value == 1500000.0
    assert path_map["metrics.arr_usd"].element_type == "number"

    assert "tags.0" in path_map
    assert path_map["tags.0"].value == "saas"


def test_jev_search_engine_hybrid_retrieval():
    """Verify JEVSearchEngine semantic vector search with structural path and predicate filters."""
    indexer = JEVIndexer()
    engine = JEVSearchEngine()

    doc_data = {
        "financials": {
            "revenue": 500000,
            "ebitda": 120000,
            "debt": 50000,
        },
        "compliance": {
            "kyc_verified": True,
            "aml_check": "passed",
        },
    }
    doc = indexer.index_document("corp_1", doc_data)
    engine.add_document(doc)

    # 1. Path-filtered search
    results = engine.search(query="profit earnings cashflow", top_k=3, path_prefix="financials")
    assert len(results) > 0
    assert all(elem.path.startswith("financials") for elem, _ in results)

    # 2. Type-filtered search
    num_results = engine.search(query="debt amount", top_k=5, element_type="number")
    assert len(num_results) > 0
    assert all(elem.element_type == "number" for elem, _ in num_results)

    # 3. Predicate-filtered search
    filtered = engine.search(
        query="all metrics",
        top_k=5,
        predicate=lambda e: isinstance(e.value, (int, float)) and e.value > 100000,
    )
    assert len(filtered) >= 2
    for elem, _ in filtered:
        assert elem.value > 100000


def test_choice_score_and_noul_primitives():
    """Verify formal semantics of Choice[T], Score bounds, and Noul sentinel."""
    # Choice[T]
    choice = Choice[DecisionOutcome](
        outcome=DecisionOutcome.APPROVE,
        confidence=0.92,
        rationale="All underwriting invariants satisfied.",
        evidence_paths=["applicant.financials.revenue", "applicant.credit_score"],
    )
    assert choice.outcome == DecisionOutcome.APPROVE
    assert choice.confidence == 0.92
    assert not is_noul(choice)

    # Score bounded validation
    score = Score(
        value=0.84,
        confidence=0.90,
        factors={"credit": 0.85, "cashflow": 0.82},
    )
    assert score.is_passing(0.70)
    assert not score.is_passing(0.90)
    assert "Score: 0.840" in score.explain()

    # Out-of-bounds score should raise ValidationError
    with pytest.raises(Exception):
        Score(value=1.5, confidence=0.8)

    # Noul Sentinel
    noul = Noul(reason="Missing bank statements verification", missing_evidence=["bank_statements"])
    assert noul.is_noul
    assert is_noul(noul)
    assert is_noul(None)
    assert not is_noul(choice)


def test_decision_gate_invariants_and_violations():
    """Verify DecisionGate blocks Noul, checks minimum score, and enforces evidence coverage."""
    gate = DecisionGate(
        min_score=0.70,
        min_confidence=0.80,
        required_evidence_paths=["tax_returns", "bank_statements"],
        forbidden_risk_flags=["bankruptcy"],
    )

    # Test 1: Noul proposed -> Gate fails with violation
    noul = Noul(reason="Applicant credit check inconclusive")
    res1 = gate.evaluate(noul, None, [])
    assert not res1.passed
    assert res1.final_decision == noul
    assert any("no_noul_allowed_for_approval" in v for v in res1.rule_violations)

    # Test 2: Low score proposed -> Escalated to REFER_MANUAL
    choice_approve = Choice[DecisionOutcome](
        outcome=DecisionOutcome.APPROVE,
        confidence=0.85,
        rationale="Promising startup business",
        evidence_paths=["tax_returns", "bank_statements"],
    )
    low_score = Score(value=0.55, confidence=0.85)
    elem1 = JEVElement(path="verified_documents.tax_returns", key="tax", value=True, element_type="boolean")
    elem2 = JEVElement(path="verified_documents.bank_statements", key="bank", value=True, element_type="boolean")

    res2 = gate.evaluate(choice_approve, low_score, [elem1, elem2])
    assert not res2.passed
    assert res2.final_decision.outcome == DecisionOutcome.REFER_MANUAL
    assert any("min_score_check" in v for v in res2.rule_violations)

    # Test 3: Clean approved application passes gate
    high_score = Score(value=0.88, confidence=0.92)
    res3 = gate.evaluate(choice_approve, high_score, [elem1, elem2])
    assert res3.passed
    assert res3.final_decision.outcome == DecisionOutcome.APPROVE
    assert len(res3.rule_violations) == 0
    assert res3.execution_latency_ms >= 0.0


def test_structured_output_extractor_and_pydantic_validation():
    """Verify StructuredOutputExtractor handles markdown code blocks and validates models."""
    sample_json = """
    ```json
    {
        "entity_name": "Helios Dynamics Inc",
        "years_in_business": 5.0,
        "owner_credit_score": 780,
        "incorporation_state": "DE"
    }
    ```
    """
    model, errs = StructuredOutputExtractor.parse_and_validate(sample_json, LoanApplicantProfile)
    assert errs is None
    assert model is not None
    assert model.entity_name == "Helios Dynamics Inc"
    assert model.owner_credit_score == 780

    # Malformed JSON block
    bad_json = "```json { entity_name: broken unquoted } ```"
    bad_model, bad_errs = StructuredOutputExtractor.parse_and_validate(bad_json, LoanApplicantProfile)
    assert bad_model is None
    assert bad_errs is not None
    assert bad_errs[0]["type"] == "json_invalid"


def test_self_repair_loop_heals_type_mismatches_and_word_numbers():
    """Verify SelfRepairLoop intercepts ValidationError and repairs common malformations."""
    repair_loop = SelfRepairLoop(max_retries=3)

    malformed_profile = {
        "entity_name": "Quantum Bio",
        "years_in_business": "three point five",  # word number string -> should heal to 3.5
        "owner_credit_score": "740",             # numeric string -> should heal to 740 int
    }

    res = repair_loop.repair_and_validate(malformed_profile, LoanApplicantProfile)
    assert res.success
    assert res.data is not None
    assert res.data.years_in_business == 3.5
    assert res.data.owner_credit_score == 740
    assert len(res.applied_patches) >= 1


def test_underwriting_engine_with_fixtures():
    """Verify TypeSafeJEVUnderwritingEngine on real fixture files."""
    policy_path = FIXTURES_DIR / "underwriting_policies.json"
    engine = TypeSafeJEVUnderwritingEngine(policy_file=policy_path)

    # 1. Valid Applicant -> APPROVE
    valid_file = FIXTURES_DIR / "applicant_valid.json"
    gate_res_valid, app_rec, elems = engine.evaluate_application(valid_file.read_text(encoding="utf-8"))
    assert gate_res_valid.passed
    assert gate_res_valid.final_decision.outcome == DecisionOutcome.APPROVE
    assert gate_res_valid.score.value >= 0.70

    # 2. Marginal Applicant -> REFER_MANUAL
    marginal_file = FIXTURES_DIR / "applicant_marginal.json"
    gate_res_marg, _, _ = engine.evaluate_application(marginal_file.read_text(encoding="utf-8"))
    assert gate_res_marg.final_decision.outcome == DecisionOutcome.REFER_MANUAL

    # 3. Malformed Applicant -> Repaired & Evaluated safely
    malformed_file = FIXTURES_DIR / "applicant_malformed.json"
    gate_res_mal, rec_mal, _ = engine.evaluate_application(malformed_file.read_text(encoding="utf-8"))
    assert rec_mal is not None
    assert rec_mal.applicant.years_in_business == 3.5
    assert rec_mal.financials.annual_revenue_usd == 680000.0


def test_compare_conditional_vs_jev_benchmark():
    """Verify head-to-head comparison: naive logic fails on malformed cases, JEV maintains 100% safety."""
    policy_path = FIXTURES_DIR / "underwriting_policies.json"
    engine = TypeSafeJEVUnderwritingEngine(policy_file=policy_path)

    cases = [
        # Case 0: Clean valid dict
        {
            "application_id": "APP-1",
            "applicant": {"entity_name": "Alpha", "years_in_business": 4.0, "owner_credit_score": 750},
            "financials": {"annual_revenue_usd": 500000.0, "debt_to_income_ratio": 0.25, "collateral_valuation_usd": 200000.0},
            "loan_request": {"requested_amount_usd": 100000.0, "loan_purpose": "growth", "requested_term_months": 36},
            "verified_documents": ["tax_returns", "bank_statements"],
        },
        # Case 1: Malformed types (string numbers) -> Naive will crash with TypeError in numeric comparison
        {
            "application_id": "APP-2",
            "applicant": {"entity_name": "Beta", "years_in_business": 2.0, "owner_credit_score": "750"},
            "financials": {"annual_revenue_usd": 400000.0, "debt_to_income_ratio": "0.30", "collateral_valuation_usd": 150000.0},
            "loan_request": {"requested_amount_usd": 80000.0, "loan_purpose": "equipment", "requested_term_months": 24},
            "verified_documents": ["tax_returns", "bank_statements"],
        },
        # Case 2: Missing required key -> Naive crashes with KeyError
        {
            "application_id": "APP-3",
            "applicant": {"entity_name": "Gamma"},
        },
    ]

    report = compare_conditional_vs_jev(cases, engine)

    assert report["total_cases"] == 3
    # Naive logic should fail on cases with string comparison or missing key
    assert report["naive_crashes_or_errors"] >= 2
    assert report["naive_crash_rate"] > 0.0

    # TypeSafe JEV must never crash (0.0% crash rate)
    assert report["jev_crash_rate"] == 0.0
    assert report["jev_safe_handling_rate"] == 1.0
