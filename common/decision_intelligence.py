"""common/decision_intelligence.py — Decision Intelligence Architecture with RAG.

This module provides a rigorous, 100% offline-executable, CPU-friendly
decision intelligence framework designed to bridge information retrieval (RAG)
and structured decision-making.

Key Architecture Pillars:
1. Retrieval vs Decision Making:
   - Retrieval locates claims, candidate alternatives, and attribute evidence.
   - Decision Intelligence computes tradeoffs, models uncertainties, evaluates risks,
     applies multi-criteria analysis (MCDA), optimizes economic payoffs, executes
     automated policy decision trees, and emits auditable verdicts with confidence,
     abstention, and human-in-the-loop escalation.
2. Multi-Criteria Decision Analysis (MCDA):
   - Weighted Sum Model (WSM) with benefit/cost normalization.
   - TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution).
   - Analytic Hierarchy Process (AHP) with pairwise consistency verification (CR <= 0.10).
3. Quantitative Risk Scoring:
   - Expected loss and Value at Risk (VaR) estimations.
   - Composite risk index penalized by evidence uncertainty/entropy.
4. Economic Optimization:
   - Expected Monetary Value (EMV) and Expected Utility (Von Neumann-Morgenstern).
   - 0-1 Knapsack resource-constrained portfolio/vendor selection.
5. Automated Policy Decision Trees & Decision Gates:
   - Rule-based deterministic gate evaluations.
   - Full integration with TypeSafe JEV Score, Choice, and Noul sentinel.
6. Confidence, Abstention & Escalation Protocol:
   - Deterministic triggers for ABSTAIN, ESCALATE_HUMAN_REVIEW, REJECT, and DECIDED.
7. Immutable Audit Trail & Counterfactual Sensitivity Analysis:
   - SHA-256 cryptographic provenance linking.
   - Minimum weight and attribute perturbations required to alter winning decisions.
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator

# Graceful integration with TypeSafe JEV
try:
    from common.typesafe_jev import Choice, Noul, Score
except ImportError:
    # Minimal fallback stubs if typesafe_jev is unavailable
    class Noul:  # type: ignore
        def __init__(self, reason: str = "Missing value"):
            self.reason = reason
        def __repr__(self) -> str:
            return f"Noul({self.reason})"

    class Score:  # type: ignore
        def __init__(self, value: float, confidence: float = 1.0):
            self.value = value
            self.confidence = confidence

    class Choice:  # type: ignore
        def __init__(self, selected: str, confidence: float = 1.0):
            self.selected = selected
            self.confidence = confidence

logger = logging.getLogger("decision_intelligence")


# ==============================================================================
# 1. Enums and Data Models
# ==============================================================================

class CriterionType(str, enum.Enum):
    """Specifies whether a criterion is a benefit (higher is better) or cost (lower is better)."""
    BENEFIT = "benefit"
    COST = "cost"


class DecisionStatus(str, enum.Enum):
    """Final decision status after policy gates, MCDA, and confidence thresholds."""
    DECIDED = "DECIDED"
    ABSTAINED = "ABSTAINED"
    ESCALATED_HUMAN_REVIEW = "ESCALATED_HUMAN_REVIEW"
    REJECTED = "REJECTED"


class Criterion(BaseModel):
    """Specification of an evaluation criterion for decision analysis."""
    name: str
    weight: float = Field(ge=0.0, description="Criterion importance weight")
    criterion_type: CriterionType = Field(default=CriterionType.BENEFIT)
    unit: str = Field(default="score", description="Measurement unit")
    description: str = Field(default="")


class Alternative(BaseModel):
    """Candidate alternative evaluated in a decision problem."""
    id: str
    name: str
    attributes: Dict[str, float] = Field(
        default_factory=dict,
        description="Quantitative attribute values keyed by criterion name"
    )
    evidence_provenance: List[str] = Field(
        default_factory=list,
        description="URIs or hashes to retrieved evidence supporting these attributes"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Confidence in extracted attributes from RAG"
    )
    risk_factors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of risk items with probability and impact"
    )


class PolicyGate(BaseModel):
    """Deterministic policy gate rule."""
    name: str
    criterion: str
    operator: str = Field(description="One of: '>=', '<=', '>', '<', '==', '!='")
    threshold: float
    rejection_action: DecisionStatus = Field(default=DecisionStatus.REJECTED)
    description: str = Field(default="")

    @field_validator("rejection_action", mode="before")
    @classmethod
    def normalize_action(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_upper = v.upper()
            if v_upper in ("REJECT", "REJECTED"):
                return DecisionStatus.REJECTED
            if v_upper in ("ESCALATE", "ESCALATED", "ESCALATED_HUMAN_REVIEW"):
                return DecisionStatus.ESCALATED_HUMAN_REVIEW
            if v_upper in ("ABSTAIN", "ABSTAINED"):
                return DecisionStatus.ABSTAINED
            if v_upper in ("DECIDE", "DECIDED"):
                return DecisionStatus.DECIDED
        return v

    def evaluate(self, value: Union[float, int, Noul, Any]) -> Tuple[bool, str]:
        """Evaluates value against gate. Handles Noul sentinels gracefully."""
        if isinstance(value, Noul) or value is None:
            return False, f"Criterion '{self.criterion}' is Noul (missing/uncertain evidence)."

        try:
            val = float(value)
        except (ValueError, TypeError):
            return False, f"Criterion '{self.criterion}' value '{value}' cannot be cast to float."

        passed = False
        if self.operator == ">=":
            passed = val >= self.threshold
        elif self.operator == "<=":
            passed = val <= self.threshold
        elif self.operator == ">":
            passed = val > self.threshold
        elif self.operator == "<":
            passed = val < self.threshold
        elif self.operator in ("==", "="):
            passed = math.isclose(val, self.threshold, rel_tol=1e-5)
        elif self.operator == "!=":
            passed = not math.isclose(val, self.threshold, rel_tol=1e-5)
        else:
            return False, f"Unknown operator '{self.operator}'"

        detail = f"{self.criterion} ({val:.2f}) {self.operator} {self.threshold:.2f} -> {'PASSED' if passed else 'FAILED'}"
        return passed, detail


@dataclass
class DecisionAuditRecord:
    """Cryptographically tamper-evident record of an automated or assisted decision."""
    decision_id: str
    timestamp: str
    objective: str
    status: DecisionStatus
    selected_alternative_id: Optional[str]
    rankings: List[Dict[str, Any]]
    scores: Dict[str, Dict[str, float]]
    risk_assessment: Dict[str, Any]
    economic_metrics: Dict[str, Any]
    policy_audit_trail: List[Dict[str, Any]]
    counterfactual_sensitivity: Dict[str, Any]
    human_review_log: Optional[Dict[str, Any]] = None
    provenance_hash: str = ""

    def compute_fingerprint(self) -> str:
        """Calculates a deterministic SHA-256 hash of the decision inputs and results."""
        payload = {
            "decision_id": self.decision_id,
            "objective": self.objective,
            "status": self.status.value if isinstance(self.status, DecisionStatus) else self.status,
            "selected_alternative_id": self.selected_alternative_id,
            "rankings": self.rankings,
            "policy_audit_trail": self.policy_audit_trail,
        }
        serialized = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ==============================================================================
# 2. Multi-Criteria Decision Analysis (MCDA) Engines
# ==============================================================================

class MCDAEngine:
    """Provides Weighted Sum Model (WSM), TOPSIS, and Analytic Hierarchy Process (AHP)."""

    @staticmethod
    def normalize_weights(criteria: List[Criterion]) -> Dict[str, float]:
        """Normalizes criterion weights so their sum equals 1.0."""
        total_weight = sum(c.weight for c in criteria)
        if total_weight <= 0:
            uniform = 1.0 / max(1, len(criteria))
            return {c.name: uniform for c in criteria}
        return {c.name: c.weight / total_weight for c in criteria}

    @staticmethod
    def weighted_sum_model(
        alternatives: List[Alternative],
        criteria: List[Criterion],
    ) -> Dict[str, Any]:
        """Calculates utility scores using the Weighted Sum Model (WSM).

        Normalizes attributes to [0, 1] using min-max scaling, where:
        - Benefit criteria: (x - min) / (max - min)
        - Cost criteria: (max - x) / (max - min)
        """
        if not alternatives or not criteria:
            return {"rankings": [], "scores": {}}

        norm_weights = MCDAEngine.normalize_weights(criteria)
        crit_map = {c.name: c for c in criteria}

        # Determine min and max for each criterion
        min_vals: Dict[str, float] = {}
        max_vals: Dict[str, float] = {}
        for c in criteria:
            vals = [alt.attributes.get(c.name, 0.0) for alt in alternatives]
            min_vals[c.name] = min(vals) if vals else 0.0
            max_vals[c.name] = max(vals) if vals else 1.0

        scores: Dict[str, Dict[str, float]] = {}
        total_scores: Dict[str, float] = {}

        for alt in alternatives:
            alt_scores: Dict[str, float] = {}
            total = 0.0
            for c in criteria:
                raw = alt.attributes.get(c.name, 0.0)
                mn = min_vals[c.name]
                mx = max_vals[c.name]
                rng = mx - mn

                if rng == 0:
                    norm = 1.0
                elif c.criterion_type == CriterionType.BENEFIT:
                    norm = (raw - mn) / rng
                else:  # COST
                    norm = (mx - raw) / rng

                w = norm_weights[c.name]
                contrib = norm * w
                alt_scores[f"{c.name}_raw"] = raw
                alt_scores[f"{c.name}_norm"] = norm
                alt_scores[f"{c.name}_weighted"] = contrib
                total += contrib

            alt_scores["wsm_total_score"] = total
            scores[alt.id] = alt_scores
            total_scores[alt.id] = total

        sorted_alts = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)
        rankings = [
            {"rank": idx + 1, "alternative_id": alt_id, "score": score}
            for idx, (alt_id, score) in enumerate(sorted_alts)
        ]

        return {"rankings": rankings, "scores": scores, "normalized_weights": norm_weights}

    @staticmethod
    def topsis(
        alternatives: List[Alternative],
        criteria: List[Criterion],
    ) -> Dict[str, Any]:
        """Executes TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution).

        Steps:
        1. Construct normalized decision matrix: r_ij = x_ij / sqrt(sum(x_kj^2))
        2. Construct weighted normalized matrix: v_ij = w_j * r_ij
        3. Determine Positive Ideal Solution (A+) and Negative Ideal Solution (A-)
        4. Calculate Euclidean separation measures (S+, S-)
        5. Calculate relative closeness to ideal solution: C_i = S- / (S+ + S-)
        """
        if not alternatives or not criteria:
            return {"rankings": [], "scores": {}}

        n_alts = len(alternatives)
        n_crits = len(criteria)
        norm_weights = MCDAEngine.normalize_weights(criteria)

        # 1. Decision matrix
        matrix = np.zeros((n_alts, n_crits), dtype=float)
        for i, alt in enumerate(alternatives):
            for j, c in enumerate(criteria):
                matrix[i, j] = float(alt.attributes.get(c.name, 0.0))

        # Vector normalization denominators: sqrt(sum(x_i^2))
        denoms = np.sqrt(np.sum(matrix ** 2, axis=0))
        denoms[denoms == 0] = 1.0  # avoid division by zero
        norm_matrix = matrix / denoms

        # 2. Weighted normalized matrix
        weights_arr = np.array([norm_weights[c.name] for c in criteria])
        weighted_matrix = norm_matrix * weights_arr

        # 3. Positive (A+) and Negative (A-) Ideal Solutions
        ideal_pos = np.zeros(n_crits, dtype=float)
        ideal_neg = np.zeros(n_crits, dtype=float)

        for j, c in enumerate(criteria):
            col = weighted_matrix[:, j]
            if c.criterion_type == CriterionType.BENEFIT:
                ideal_pos[j] = np.max(col)
                ideal_neg[j] = np.min(col)
            else:  # COST
                ideal_pos[j] = np.min(col)
                ideal_neg[j] = np.max(col)

        # 4. Separation measures
        sep_pos = np.sqrt(np.sum((weighted_matrix - ideal_pos) ** 2, axis=1))
        sep_neg = np.sqrt(np.sum((weighted_matrix - ideal_neg) ** 2, axis=1))

        # 5. Relative closeness: C_i = S- / (S+ + S-)
        total_sep = sep_pos + sep_neg
        total_sep[total_sep == 0] = 1e-9
        closeness = sep_neg / total_sep

        scores: Dict[str, Dict[str, float]] = {}
        for i, alt in enumerate(alternatives):
            scores[alt.id] = {
                "topsis_score": float(closeness[i]),
                "separation_positive_ideal": float(sep_pos[i]),
                "separation_negative_ideal": float(sep_neg[i]),
            }

        sorted_indices = np.argsort(-closeness)
        rankings = [
            {
                "rank": rank + 1,
                "alternative_id": alternatives[idx].id,
                "score": float(closeness[idx]),
                "separation_pos": float(sep_pos[idx]),
                "separation_neg": float(sep_neg[idx]),
            }
            for rank, idx in enumerate(sorted_indices)
        ]

        return {
            "rankings": rankings,
            "scores": scores,
            "ideal_positive": ideal_pos.tolist(),
            "ideal_negative": ideal_neg.tolist(),
        }

    @staticmethod
    def analytic_hierarchy_process(
        pairwise_matrix: List[List[float]],
        criterion_names: List[str],
    ) -> Dict[str, Any]:
        """Calculates criteria weights and consistency ratio using Analytic Hierarchy Process (AHP).

        Uses standard Saaty Random Consistency Index (RI) table.
        Asserts or reports whether Consistency Ratio (CR) <= 0.10.
        """
        n = len(pairwise_matrix)
        if n != len(criterion_names) or n == 0:
            raise ValueError("Pairwise matrix dimensions must match criterion names count")

        matrix = np.array(pairwise_matrix, dtype=float)

        # Compute principal eigenvector via power iteration
        vec = np.ones(n, dtype=float) / n
        for _ in range(50):
            next_vec = np.dot(matrix, vec)
            norm = np.sum(next_vec)
            if norm > 0:
                next_vec /= norm
            vec = next_vec

        weights = vec / np.sum(vec)

        # Compute lambda_max: average of (A * w)_i / w_i
        aw = np.dot(matrix, weights)
        lambda_max = float(np.mean(aw / weights))

        # Consistency Index (CI)
        ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0

        # Saaty Random Index (RI) lookup table
        ri_table = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        ri = ri_table.get(n, 1.49)

        # Consistency Ratio (CR)
        cr = ci / ri if ri > 0 else 0.0
        is_consistent = cr <= 0.10

        weights_dict = {name: float(w) for name, w in zip(criterion_names, weights)}

        return {
            "weights": weights_dict,
            "lambda_max": lambda_max,
            "consistency_index": ci,
            "random_index": ri,
            "consistency_ratio": cr,
            "is_consistent": is_consistent,
        }


# ==============================================================================
# 3. Quantitative Risk Scoring Engine
# ==============================================================================

class RiskScoringEngine:
    """Computes expected loss, Value at Risk (VaR), and uncertainty-penalized composite risk."""

    @staticmethod
    def calculate_expected_loss(risk_factors: List[Dict[str, Any]]) -> float:
        """Expected Loss E[L] = sum(p_k * impact_k)."""
        loss = 0.0
        for rf in risk_factors:
            prob = float(rf.get("probability", 0.0))
            impact = float(rf.get("impact", 0.0))
            loss += prob * impact
        return loss

    @staticmethod
    def calculate_composite_risk(
        risk_factors: List[Dict[str, Any]],
        evidence_confidence: float = 1.0,
        uncertainty_penalty_factor: float = 0.5,
    ) -> Dict[str, Any]:
        """Calculates composite risk score penalized by ungrounded or low-confidence evidence.

        R_composite = E[Loss] * (1 + lambda * (1 - confidence))
        """
        expected_loss = RiskScoringEngine.calculate_expected_loss(risk_factors)
        uncertainty_multiplier = 1.0 + (uncertainty_penalty_factor * (1.0 - max(0.0, min(1.0, evidence_confidence))))
        penalized_risk = expected_loss * uncertainty_multiplier

        # Parametric VaR approximation (95% confidence level under normal approximation)
        impacts = [float(rf.get("impact", 0.0)) for rf in risk_factors]
        probs = [float(rf.get("probability", 0.0)) for rf in risk_factors]
        variance = sum(p * ((imp - (p * imp)) ** 2) for p, imp in zip(probs, impacts)) if impacts else 0.0
        std_dev = math.sqrt(variance)
        var_95 = expected_loss + (1.645 * std_dev)

        return {
            "expected_loss": expected_loss,
            "evidence_confidence": evidence_confidence,
            "uncertainty_multiplier": uncertainty_multiplier,
            "penalized_risk_score": penalized_risk,
            "std_deviation": std_dev,
            "value_at_risk_95": var_95,
            "factor_count": len(risk_factors),
        }


# ==============================================================================
# 4. Economic Optimization Models
# ==============================================================================

class EconomicOptimizer:
    """Models Expected Monetary Value (EMV), Utility curves, and Resource Allocations."""

    @staticmethod
    def expected_monetary_value(payoffs: Dict[str, float], state_probabilities: Dict[str, float]) -> float:
        """EMV = sum(P(state_j) * payoff_j)."""
        emv = 0.0
        for state, prob in state_probabilities.items():
            payoff = payoffs.get(state, 0.0)
            emv += prob * payoff
        return emv

    @staticmethod
    def von_neumann_utility(payoff: float, risk_tolerance_alpha: float = 0.0) -> float:
        """Computes Von Neumann-Morgenstern utility under risk preferences.

        - alpha == 0: Risk-neutral (U(x) = x)
        - alpha > 0: Risk-averse (U(x) = (1 - e^(-alpha * x)) / alpha)
        - alpha < 0: Risk-seeking (U(x) = (e^(-alpha * x) - 1) / abs(alpha))
        """
        if math.isclose(risk_tolerance_alpha, 0.0, abs_tol=1e-6):
            return payoff

        # Rescale payoff to avoid math overflow
        scaled = payoff / 1000.0 if abs(payoff) > 10000 else payoff
        if risk_tolerance_alpha > 0:
            # Risk averse
            exponent = min(50.0, max(-50.0, -risk_tolerance_alpha * scaled))
            return (1.0 - math.exp(exponent)) / risk_tolerance_alpha
        else:
            # Risk seeking
            exponent = min(50.0, max(-50.0, -risk_tolerance_alpha * scaled))
            return (math.exp(exponent) - 1.0) / abs(risk_tolerance_alpha)

    @staticmethod
    def knapsack_portfolio_selection(
        items: List[Dict[str, Any]],
        budget_capacity: float,
        cost_key: str = "cost",
        value_key: str = "value",
    ) -> Dict[str, Any]:
        """Solves 0-1 Knapsack resource allocation to maximize total value within budget capacity.

        Uses dynamic programming with scaled integer costs.
        """
        if not items or budget_capacity <= 0:
            return {"selected_ids": [], "total_value": 0.0, "total_cost": 0.0}

        # Scale costs to integer grid (up to 1000 bins)
        costs = [float(item.get(cost_key, 0.0)) for item in items]
        values = [float(item.get(value_key, 0.0)) for item in items]

        max_cost = max(costs) if costs else 1.0
        scale = 1000.0 / max(1.0, max_cost * len(items))
        int_capacity = int(budget_capacity * scale)
        int_costs = [int(c * scale) for c in costs]

        n = len(items)
        dp = np.zeros((n + 1, int_capacity + 1), dtype=float)

        for i in range(1, n + 1):
            w = int_costs[i - 1]
            v = values[i - 1]
            for j in range(int_capacity + 1):
                if w <= j:
                    dp[i, j] = max(dp[i - 1, j], dp[i - 1, j - w] + v)
                else:
                    dp[i, j] = dp[i - 1, j]

        # Backtrack selected items
        selected_indices = []
        curr_cap = int_capacity
        for i in range(n, 0, -1):
            if dp[i, curr_cap] != dp[i - 1, curr_cap]:
                selected_indices.append(i - 1)
                curr_cap -= int_costs[i - 1]

        selected_indices.reverse()
        selected_items = [items[idx] for idx in selected_indices]
        total_cost = sum(float(item.get(cost_key, 0.0)) for item in selected_items)
        total_val = sum(float(item.get(value_key, 0.0)) for item in selected_items)

        return {
            "selected_ids": [item.get("id", str(idx)) for idx, item in enumerate(selected_items)],
            "selected_items": selected_items,
            "total_value": total_val,
            "total_cost": total_cost,
            "budget_capacity": budget_capacity,
            "budget_utilization": total_cost / budget_capacity if budget_capacity > 0 else 0.0,
        }


# ==============================================================================
# 5. Automated Policy Decision Trees & Decision Gates
# ==============================================================================

class AutomatedPolicyEngine:
    """Evaluates hierarchical decision gates and policy rules with JEV type safety."""

    def __init__(self, gates: Optional[List[PolicyGate]] = None):
        self.gates = gates or []

    def evaluate_alternative(self, alternative: Alternative) -> Dict[str, Any]:
        """Evaluates an alternative through all policy gates.

        If a gate fails, returns the violation, action, and detailed rationale.
        """
        audit_trail: List[Dict[str, Any]] = []
        is_admissible = True
        recommended_action = DecisionStatus.DECIDED
        failure_reasons: List[str] = []

        for gate in self.gates:
            val = alternative.attributes.get(gate.criterion, Noul(reason=f"Missing {gate.criterion}"))
            passed, detail = gate.evaluate(val)
            audit_entry = {
                "gate_name": gate.name,
                "criterion": gate.criterion,
                "value": str(val),
                "operator": gate.operator,
                "threshold": gate.threshold,
                "passed": passed,
                "detail": detail,
            }
            audit_trail.append(audit_entry)

            if not passed:
                is_admissible = False
                failure_reasons.append(detail)
                # Escalate or Reject based on gate specification
                if gate.rejection_action == DecisionStatus.REJECTED:
                    recommended_action = DecisionStatus.REJECTED
                elif gate.rejection_action == DecisionStatus.ESCALATED_HUMAN_REVIEW and recommended_action != DecisionStatus.REJECTED:
                    recommended_action = DecisionStatus.ESCALATED_HUMAN_REVIEW

        return {
            "alternative_id": alternative.id,
            "is_admissible": is_admissible,
            "recommended_action": recommended_action,
            "failure_reasons": failure_reasons,
            "audit_trail": audit_trail,
        }


# ==============================================================================
# 6. Confidence, Abstention & Escalation Protocol
# ==============================================================================

class ConfidenceAbstentionProtocol:
    """Calculates decision confidence, detects ambiguity, and enforces abstention or escalation."""

    @staticmethod
    def assess_decision(
        top_rankings: List[Dict[str, Any]],
        admissible_alternatives: List[Alternative],
        min_evidence_confidence: float = 0.70,
        ambiguity_margin_threshold: float = 0.04,
        max_acceptable_composite_risk: float = 12.0,
    ) -> Tuple[DecisionStatus, str, Optional[str]]:
        """Determines whether the system can decide, must abstain, or must escalate.

        Rules:
        1. If no admissible alternatives remain -> REJECTED / ABSTAINED.
        2. If top candidate's evidence confidence < min_evidence_confidence -> ABSTAINED.
        3. If margin between #1 and #2 candidate is < ambiguity_margin_threshold -> ESCALATED_HUMAN_REVIEW.
        4. If top candidate composite risk > max_acceptable_composite_risk -> ESCALATED_HUMAN_REVIEW.
        5. Otherwise -> DECIDED.
        """
        if not top_rankings or not admissible_alternatives:
            return DecisionStatus.REJECTED, "No candidate alternatives met the baseline policy gates.", None

        top_entry = top_rankings[0]
        winner_id = top_entry["alternative_id"]
        winner_score = top_entry["score"]

        # Find matching alternative object
        alt_map = {alt.id: alt for alt in admissible_alternatives}
        winner_alt = alt_map.get(winner_id)

        if not winner_alt:
            return DecisionStatus.ABSTAINED, f"Winner '{winner_id}' missing in admissible candidate pool.", None

        # Check retrieval evidence confidence
        if winner_alt.confidence < min_evidence_confidence:
            reason = (
                f"Abstention triggered: Evidence confidence for top candidate '{winner_alt.name}' "
                f"is {winner_alt.confidence:.2f}, below safety threshold {min_evidence_confidence:.2f}."
            )
            return DecisionStatus.ABSTAINED, reason, winner_id

        # Check ambiguity margin if 2 or more candidates exist
        if len(top_rankings) > 1:
            second_entry = top_rankings[1]
            margin = winner_score - second_entry["score"]
            if margin < ambiguity_margin_threshold:
                reason = (
                    f"Human escalation triggered: Ambiguity margin between #1 ({winner_id}: {winner_score:.3f}) "
                    f"and #2 ({second_entry['alternative_id']}: {second_entry['score']:.3f}) is {margin:.4f}, "
                    f"below discrimination threshold {ambiguity_margin_threshold:.4f}."
                )
                return DecisionStatus.ESCALATED_HUMAN_REVIEW, reason, winner_id

        # Check risk bounds
        risk_profile = RiskScoringEngine.calculate_composite_risk(
            winner_alt.risk_factors,
            evidence_confidence=winner_alt.confidence
        )
        if risk_profile["penalized_risk_score"] > max_acceptable_composite_risk:
            reason = (
                f"Human escalation triggered: Candidate '{winner_alt.name}' composite risk "
                f"{risk_profile['penalized_risk_score']:.2f} exceeds risk ceiling {max_acceptable_composite_risk:.2f}."
            )
            return DecisionStatus.ESCALATED_HUMAN_REVIEW, reason, winner_id

        return DecisionStatus.DECIDED, f"Clear optimal candidate '{winner_alt.name}' selected with high confidence.", winner_id


# ==============================================================================
# 7. Counterfactual Sensitivity & Audit Ledger
# ==============================================================================

class DecisionSensitivityAnalyzer:
    """Analyzes the robustness of decision rankings against criterion weight changes."""

    @staticmethod
    def analyze_weight_sensitivity(
        alternatives: List[Alternative],
        criteria: List[Criterion],
        perturbation_step: float = 0.05,
    ) -> Dict[str, Any]:
        """Calculates sensitivity: minimum weight change required to flip the top ranking."""
        base_result = MCDAEngine.topsis(alternatives, criteria)
        if not base_result["rankings"]:
            return {"sensitive_criteria": [], "robustness_score": 1.0}

        base_winner = base_result["rankings"][0]["alternative_id"]
        flips: List[Dict[str, Any]] = []

        norm_weights = MCDAEngine.normalize_weights(criteria)

        for c in criteria:
            orig_w = norm_weights[c.name]
            # Test increasing and decreasing weight
            for direction in [1.0, -1.0]:
                for step in np.arange(perturbation_step, 0.50, perturbation_step):
                    new_w = max(0.01, float(norm_weights[c.name] + (step * direction)))
                    test_criteria = [
                        Criterion(
                            name=cr.name,
                            weight=new_w if cr.name == c.name else float(norm_weights[cr.name]),
                            criterion_type=cr.criterion_type,
                            unit=cr.unit,
                            description=cr.description,
                        )
                        for cr in criteria
                    ]

                    test_result = MCDAEngine.topsis(alternatives, test_criteria)
                    test_winner = test_result["rankings"][0]["alternative_id"]
                    if test_winner != base_winner:
                        flips.append({
                            "criterion": c.name,
                            "direction": "increase" if direction > 0 else "decrease",
                            "delta_weight": float(step * direction),
                            "new_winner": test_winner,
                        })
                        break
                if any(f["criterion"] == c.name for f in flips):
                    break

        robustness = 1.0 - (len(flips) / max(1, len(criteria)))

        return {
            "base_winner": base_winner,
            "flips": flips,
            "robustness_score": float(robustness),
            "is_highly_sensitive": len(flips) >= (len(criteria) / 2),
        }


# ==============================================================================
# 8. End-to-End Decision RAG Orchestrator
# ==============================================================================

class DecisionRAGOrchestrator:
    """Coordinates retrieval evidence, policy gates, MCDA, risk scoring, and audit logging."""

    def __init__(
        self,
        criteria: List[Criterion],
        policy_gates: Optional[List[PolicyGate]] = None,
        budget_constraint: Optional[float] = None,
    ):
        self.criteria = criteria
        self.policy_gates = policy_gates or []
        self.policy_engine = AutomatedPolicyEngine(self.policy_gates)
        self.budget_constraint = budget_constraint

    def solve(
        self,
        objective: str,
        alternatives: List[Alternative],
        min_evidence_confidence: float = 0.70,
        ambiguity_margin: float = 0.04,
        max_risk_threshold: float = 12.0,
    ) -> DecisionAuditRecord:
        """Executes full decision intelligence pipeline and returns tamper-evident audit record."""
        decision_id = f"dec_{hashlib.md5(f'{objective}_{len(alternatives)}'.encode()).hexdigest()[:10]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        # Step 1: Policy Gates
        admissible_alternatives: List[Alternative] = []
        policy_audit_trail: List[Dict[str, Any]] = []

        for alt in alternatives:
            gate_eval = self.policy_engine.evaluate_alternative(alt)
            policy_audit_trail.append(gate_eval)
            if gate_eval["is_admissible"]:
                admissible_alternatives.append(alt)

        # Step 2: MCDA (TOPSIS & WSM) on admissible candidates
        if admissible_alternatives:
            topsis_res = MCDAEngine.topsis(admissible_alternatives, self.criteria)
            wsm_res = MCDAEngine.weighted_sum_model(admissible_alternatives, self.criteria)
            rankings = topsis_res["rankings"]
            combined_scores: Dict[str, Dict[str, float]] = {}
            for alt in admissible_alternatives:
                t_score = topsis_res["scores"].get(alt.id, {})
                w_score = wsm_res["scores"].get(alt.id, {})
                combined_scores[alt.id] = {**t_score, **w_score}
        else:
            rankings = []
            combined_scores = {}

        # Step 3: Confidence, Abstention & Escalation Protocol
        status, rationale, winner_id = ConfidenceAbstentionProtocol.assess_decision(
            top_rankings=rankings,
            admissible_alternatives=admissible_alternatives,
            min_evidence_confidence=min_evidence_confidence,
            ambiguity_margin_threshold=ambiguity_margin,
            max_acceptable_composite_risk=max_risk_threshold,
        )

        # Step 4: Risk and Economic Assessment for selected candidate
        winner_alt = next((alt for alt in alternatives if alt.id == winner_id), None)
        if winner_alt:
            risk_assessment = RiskScoringEngine.calculate_composite_risk(
                winner_alt.risk_factors,
                evidence_confidence=winner_alt.confidence
            )
            cost_val = winner_alt.attributes.get("annual_tco_usd", winner_alt.attributes.get("cost", 0.0))
            economic_metrics = {
                "direct_cost": cost_val,
                "budget_constraint": self.budget_constraint,
                "budget_headroom": (self.budget_constraint - cost_val) if self.budget_constraint else None,
                "expected_utility_risk_averse": EconomicOptimizer.von_neumann_utility(
                    payoff=100000.0 - cost_val, risk_tolerance_alpha=0.001
                ),
            }
        else:
            risk_assessment = {}
            economic_metrics = {}

        # Step 5: Sensitivity Analysis
        if admissible_alternatives and len(admissible_alternatives) > 1:
            sensitivity = DecisionSensitivityAnalyzer.analyze_weight_sensitivity(
                admissible_alternatives, self.criteria
            )
        else:
            sensitivity = {"note": "Sensitivity analysis requires >= 2 admissible candidates"}

        # Step 6: Formulate and sign immutable audit record
        audit_record = DecisionAuditRecord(
            decision_id=decision_id,
            timestamp=timestamp,
            objective=objective,
            status=status,
            selected_alternative_id=winner_id if status == DecisionStatus.DECIDED else None,
            rankings=rankings,
            scores=combined_scores,
            risk_assessment=risk_assessment,
            economic_metrics=economic_metrics,
            policy_audit_trail=policy_audit_trail,
            counterfactual_sensitivity=sensitivity,
        )
        audit_record.provenance_hash = audit_record.compute_fingerprint()

        return audit_record


# ==============================================================================
# 9. Visualization Utilities
# ==============================================================================

def render_decision_intelligence_diagrams(output_dir: Union[str, Path]) -> List[str]:
    """Generates visual decision intelligence diagrams (MCDA radar/bar & decision flow)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_files = []

    # Diagram 1: MCDA Multi-Criteria Performance Comparison
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    vendors = ["ApexVector", "NovaScale", "OpenVector", "ChromaScale"]
    topsis_scores = [0.842, 0.615, 0.438, 0.765]
    colors = ["#2563EB", "#10B981", "#F59E0B", "#8B5CF6"]

    bars = ax.bar(vendors, topsis_scores, color=colors, alpha=0.85, edgecolor="#1E293B", linewidth=1.5)
    ax.axhline(0.70, color="#EF4444", linestyle="--", linewidth=1.5, label="High-Confidence Decision Threshold (0.70)")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("TOPSIS Closeness Score $C_i$", fontsize=12, fontweight="bold")
    ax.set_title("Multi-Criteria Decision Analysis: Vendor Relative Closeness to Ideal Solution", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    for bar, score in zip(bars, topsis_scores):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{score:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    diagram1_path = out_path / "track13_mcda_vendor_comparison.png"
    plt.savefig(diagram1_path)
    plt.close()
    generated_files.append(str(diagram1_path))

    # Diagram 2: Decision Intelligence Architecture (Retrieval vs Policy vs MCDA vs Audit)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.axis("off")

    boxes = [
        {"x": 0.05, "y": 0.4, "w": 0.18, "h": 0.35, "title": "1. Evidence Retrieval\n(OKF / JEV / RAG)", "color": "#DBEAFE", "border": "#1D4ED8"},
        {"x": 0.28, "y": 0.4, "w": 0.18, "h": 0.35, "title": "2. Policy Gates\n& JEV Constraints", "color": "#FEF3C7", "border": "#D97706"},
        {"x": 0.51, "y": 0.4, "w": 0.18, "h": 0.35, "title": "3. MCDA & Risk\n(TOPSIS / VaR / EMV)", "color": "#D1FAE5", "border": "#059669"},
        {"x": 0.74, "y": 0.4, "w": 0.21, "h": 0.35, "title": "4. Audit & Protocol\n(Abstain/Escalate/Sign)", "color": "#EDE9FE", "border": "#7C3AED"},
    ]

    for b in boxes:
        rect = plt.Rectangle((b["x"], b["y"]), b["w"], b["h"], facecolor=b["color"], edgecolor=b["border"], linewidth=2, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(b["x"] + b["w"] / 2.0, b["y"] + b["h"] / 2.0, b["title"], ha="center", va="center", fontsize=11, fontweight="bold", transform=ax.transAxes)

    # Arrows between boxes
    arrow_props = dict(facecolor="#475569", edgecolor="#475569", width=2, headwidth=8)
    ax.annotate("", xy=(0.28, 0.575), xytext=(0.23, 0.575), xycoords="axes fraction", arrowprops=arrow_props)
    ax.annotate("", xy=(0.51, 0.575), xytext=(0.46, 0.575), xycoords="axes fraction", arrowprops=arrow_props)
    ax.annotate("", xy=(0.74, 0.575), xytext=(0.69, 0.575), xycoords="axes fraction", arrowprops=arrow_props)

    ax.set_title("Decision Intelligence Pipeline: Moving Beyond Retrieval into Provable Decision Optimization", fontsize=13, fontweight="bold")
    plt.tight_layout()
    diagram2_path = out_path / "track13_decision_intelligence_pipeline.png"
    plt.savefig(diagram2_path)
    plt.close()
    generated_files.append(str(diagram2_path))

    return generated_files
