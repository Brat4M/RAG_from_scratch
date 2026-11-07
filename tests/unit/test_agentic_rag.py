"""Unit tests for Agentic RAG: Tooling, ReAct Loops, Planning, DeepAgent, and Guardrails."""

import pytest
import sqlite3
from typing import Dict, Any, List, Tuple, Optional

from common.agentic_rag import (
    ToolPermission,
    ToolDefinition,
    ToolRegistry,
    SafeCalculator,
    create_in_memory_database,
    create_default_tool_registry,
    AgentBudget,
    AgentStep,
    AgentTrajectory,
    ReActAgent,
    PlanStep,
    ExecutionPlan,
    PlanAndSolveEngine,
    RetrievalSpecialist,
    DataAnalystSpecialist,
    CriticSpecialist,
    DeepAgentCoordinator,
    AgentEvaluator,
    AgentFailureMode,
)


# ---------------------------------------------------------------------------
# 1. Safe Calculator Unit Tests
# ---------------------------------------------------------------------------

def test_safe_calculator_basic_arithmetic():
    assert SafeCalculator.evaluate("2 + 2") == 4.0
    assert SafeCalculator.evaluate("10 - 4.5") == 5.5
    assert SafeCalculator.evaluate("6 * 7") == 42.0
    assert SafeCalculator.evaluate("15 / 3") == 5.0
    assert SafeCalculator.evaluate("2 ** 3") == 8.0
    assert SafeCalculator.evaluate("10 % 3") == 1.0


def test_safe_calculator_composite_and_builtins():
    expr = "round((61000000 - 39000000) / 61000000 * 100, 2)"
    result = SafeCalculator.evaluate(expr)
    assert result == 36.07

    assert SafeCalculator.evaluate("abs(-42)") == 42.0
    assert SafeCalculator.evaluate("min(10, 5, 20)") == 5.0
    assert SafeCalculator.evaluate("max(10, 5, 20)") == 20.0


def test_safe_calculator_zero_division():
    with pytest.raises(ZeroDivisionError):
        SafeCalculator.evaluate("100 / 0")


def test_safe_calculator_security_sandbox():
    # Disallow imports, variable lookups, or arbitrary code execution
    with pytest.raises(ValueError):
        SafeCalculator.evaluate("__import__('os').system('ls')")

    with pytest.raises(ValueError):
        SafeCalculator.evaluate("open('secret.txt').read()")


# ---------------------------------------------------------------------------
# 2. Tool Registry and Permission Verification
# ---------------------------------------------------------------------------

def test_tool_registry_registration_and_schema():
    registry = ToolRegistry()
    tool = ToolDefinition(
        name="dummy_tool",
        description="A test tool",
        parameters={
            "query": {"type": "string", "description": "The search term", "required": True},
            "limit": {"type": "integer", "description": "Max items", "required": False},
        },
        func=lambda query, limit=5: f"Found {limit} items for {query}",
        permission=ToolPermission.READ_ONLY,
    )
    registry.register(tool)

    assert registry.get("dummy_tool") is not None
    assert registry.get("non_existent") is None

    schemas = registry.get_schemas()
    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "dummy_tool"
    assert "query" in schemas[0]["function"]["parameters"]["properties"]
    assert "query" in schemas[0]["function"]["parameters"]["required"]


def test_tool_registry_execution_and_error_handling():
    registry = ToolRegistry()

    def buggy_func(x: int):
        if x < 0:
            raise ValueError("Negative numbers not allowed")
        return x * 2

    registry.register(
        ToolDefinition(
            name="doubler",
            description="Doubles positive numbers",
            parameters={"x": {"type": "integer"}},
            func=buggy_func,
        )
    )

    # Success case
    res_ok = registry.execute("doubler", {"x": 5})
    assert res_ok.success is True
    assert res_ok.output == 10
    assert res_ok.error is None
    assert res_ok.execution_time_ms >= 0

    # Failure case
    res_fail = registry.execute("doubler", {"x": -2})
    assert res_fail.success is False
    assert res_fail.output is None
    assert "Negative numbers not allowed" in (res_fail.error or "")

    # Non-existent tool
    res_missing = registry.execute("unknown_tool", {})
    assert res_missing.success is False
    assert "not registered" in (res_missing.error or "")


def test_tool_registry_permission_enforcement():
    # Create registry that only permits READ_ONLY
    restricted_registry = ToolRegistry(allowed_permissions={ToolPermission.READ_ONLY})

    safe_tool = ToolDefinition(
        name="read_only_tool",
        description="Safe tool",
        parameters={},
        func=lambda: "Safe Data",
        permission=ToolPermission.READ_ONLY,
    )
    dangerous_tool = ToolDefinition(
        name="modify_tool",
        description="Write tool",
        parameters={},
        func=lambda: "Modified Data",
        permission=ToolPermission.EXECUTE,
    )

    restricted_registry.register(safe_tool)
    restricted_registry.register(dangerous_tool)

    # Allowed
    res_safe = restricted_registry.execute("read_only_tool", {})
    assert res_safe.success is True
    assert res_safe.output == "Safe Data"

    # Blocked by permission policy
    res_danger = restricted_registry.execute("modify_tool", {})
    assert res_danger.success is False
    assert "Permission Denied" in (res_danger.error or "")


# ---------------------------------------------------------------------------
# 3. Standard Tools Functionality
# ---------------------------------------------------------------------------

def test_default_tool_registry_search_and_query():
    registry = create_default_tool_registry()

    # Test document retrieval
    doc_res = registry.execute("search_documents", {"query": "NovaTech revenue", "top_k": 1})
    assert doc_res.success is True
    assert len(doc_res.output) >= 1
    assert "NovaTech AI" in doc_res.output[0]["content"]

    # Test structured database read
    db_res = registry.execute(
        "query_database",
        {"sql_query": "SELECT revenue FROM corporate_finances WHERE company='NovaTech AI' AND quarter='Q3-2026'"},
    )
    assert db_res.success is True
    assert db_res.output[0]["revenue"] == 61000000.0

    # Test read-only security check on database tool
    db_fail = registry.execute(
        "query_database",
        {"sql_query": "DROP TABLE corporate_finances"},
    )
    assert db_fail.success is False
    assert "prohibited" in (db_fail.error or "") or "only permits read-only" in (db_fail.error or "")

    # Test web search mock
    web_res = registry.execute("search_external_web", {"query": "industry benchmark margins"})
    assert web_res.success is True
    assert "75% to 80%" in web_res.output["snippet"]


# ---------------------------------------------------------------------------
# 4. Budget & Loop Detection Guardrails
# ---------------------------------------------------------------------------

def test_agent_budget_iteration_limit():
    budget = AgentBudget(max_iterations=3, max_tool_calls=10)
    assert not budget.is_exhausted()[0]

    budget.record_iteration()
    budget.record_iteration()
    assert not budget.is_exhausted()[0]

    budget.record_iteration()
    exhausted, reason = budget.is_exhausted()
    assert exhausted is True
    assert "Iteration budget exhausted" in (reason or "")


def test_agent_budget_loop_detection_hash():
    budget = AgentBudget(max_iterations=10, max_tool_calls=10, max_retries_per_tool=2)

    # Call 1: ok
    valid1, err1 = budget.record_tool_call("search_documents", {"query": "exact same query"})
    assert valid1 is True
    assert err1 is None

    # Call 2 (repeat 1): ok within retry limit
    valid2, err2 = budget.record_tool_call("search_documents", {"query": "exact same query"})
    assert valid2 is True
    assert err2 is None

    # Call 3 (repeat 2): loop detector triggers
    valid3, err3 = budget.record_tool_call("search_documents", {"query": "exact same query"})
    assert valid3 is False
    assert "Loop detected" in (err3 or "")


# ---------------------------------------------------------------------------
# 5. ReAct Agent Core Execution
# ---------------------------------------------------------------------------

def test_react_agent_multi_hop_trajectory():
    registry = create_default_tool_registry()
    agent = ReActAgent(tool_registry=registry, budget=AgentBudget(max_iterations=6, max_tool_calls=6))

    query = "Find NovaTech AI Q3 revenue and calculate its operating profit margin."
    trajectory = agent.run(query)

    assert trajectory.success is True
    assert trajectory.termination_reason == "GOAL_COMPLETED"
    assert len(trajectory.steps) >= 3

    # Check that search_documents, query_database, and calculate were all called
    actions = [s.action for s in trajectory.steps if s.action]
    assert "search_documents" in actions
    assert "query_database" in actions
    assert "calculate" in actions

    # Verify final answer contents
    assert "NovaTech AI" in trajectory.final_answer
    assert "61.0M" in trajectory.final_answer
    assert "36.07" in trajectory.final_answer or "36" in trajectory.final_answer


def test_react_agent_budget_exhaustion_recovery():
    registry = create_default_tool_registry()
    # Very constrained budget of 1 iteration
    agent = ReActAgent(tool_registry=registry, budget=AgentBudget(max_iterations=1, max_tool_calls=1))

    trajectory = agent.run("Find NovaTech AI Q3 revenue and calculate its operating profit margin.")
    assert trajectory.success is False
    assert trajectory.termination_reason == "BUDGET_EXHAUSTED"
    assert "budget exhausted" in trajectory.final_answer.lower()


def test_react_agent_loop_prevention_intervention():
    registry = create_default_tool_registry()
    agent = ReActAgent(tool_registry=registry, budget=AgentBudget(max_iterations=10, max_retries_per_tool=2))

    # A stuck policy that repeatedly calls the same tool
    def stuck_policy(query, history):
        return ("Stuck thought", "search_documents", {"query": "endless identical loop"}, None)

    trajectory = agent.run("Test query", offline_policy=stuck_policy)
    assert trajectory.success is False
    assert trajectory.termination_reason == "LOOP_DETECTED"
    assert "guardrails" in trajectory.final_answer.lower()


# ---------------------------------------------------------------------------
# 6. Plan-and-Solve Dynamic Planning
# ---------------------------------------------------------------------------

def test_plan_and_solve_formulation_and_execution():
    registry = create_default_tool_registry()
    engine = PlanAndSolveEngine(tool_registry=registry)

    query = "Compare NovaTech AI and Helios Dynamics Q3 financial performance"
    plan = engine.formulate_plan(query)

    assert len(plan.steps) == 3
    assert plan.steps[0].tool_name == "query_database"
    assert plan.steps[1].tool_name == "query_database"
    assert plan.steps[2].tool_name == "calculate"

    completed_plan, final_answer = engine.execute_plan(plan)
    assert completed_plan.is_complete is True
    assert "Execution Plan Completed" in final_answer
    assert all(s.status == "completed" for s in completed_plan.steps)


def test_plan_and_solve_dynamic_replanning_on_failure():
    registry = create_default_tool_registry()
    engine = PlanAndSolveEngine(tool_registry=registry, max_replans=2)

    # Initial plan with a failing step
    plan = ExecutionPlan(
        goal="Test fallback plan",
        steps=[
            PlanStep(
                step_id=1,
                description="Query invalid table",
                tool_name="query_database",
                tool_args={"sql_query": "SELECT * FROM non_existent_table"},
            )
        ],
    )

    completed_plan, _ = engine.execute_plan(plan)
    assert completed_plan.replan_count == 1
    # Check that a fallback step was appended
    assert len(completed_plan.steps) == 2
    assert completed_plan.steps[1].tool_name == "search_documents"


# ---------------------------------------------------------------------------
# 7. DeepAgent Multi-Agent Coordination
# ---------------------------------------------------------------------------

def test_deepagent_coordinator_workflow():
    registry = create_default_tool_registry()
    coordinator = DeepAgentCoordinator(registry)

    result = coordinator.solve("Analyze NovaTech AI Q3 operating margins and corporate health")

    assert "Supervisor received goal" in result["trace"][0]
    assert result["critic_verdict"] == "PASS"
    assert result["grounding_score"] >= 0.5
    assert "NovaTech AI" in result["final_answer"]
    assert "specialist_outputs" in result
    assert result["specialist_outputs"]["retrieval"]["status"] == "success"


def test_critic_specialist_verdicts():
    critic = CriticSpecialist()

    evidence = [
        "NovaTech AI reported Q3 revenue of $61.0M and operating expenses of $39.0M.",
        "Net income reached $22.0M with gross margins of 78%."
    ]

    # Grounded claim -> PASS
    pass_res = critic.evaluate(
        "NovaTech AI had Q3 revenue of $61.0M and net income of $22.0M.",
        evidence,
    )
    assert pass_res["verdict"] == "PASS"
    assert pass_res["grounding_ratio"] >= 0.5

    # Completely ungrounded claim -> REVISE
    fail_res = critic.evaluate(
        "Quantum computing satellites launched to Mars with laser propulsion.",
        evidence,
    )
    assert fail_res["verdict"] == "REVISE"
    assert fail_res["grounding_ratio"] < 0.5


# ---------------------------------------------------------------------------
# 8. Agent Evaluator & Failure Taxonomy
# ---------------------------------------------------------------------------

def test_agent_evaluator_metrics_and_taxonomy():
    registry = create_default_tool_registry()
    agent = ReActAgent(tool_registry=registry)

    query = "Find NovaTech AI Q3 revenue and calculate its operating profit margin."
    trajectory = agent.run(query)

    eval_result = AgentEvaluator.evaluate_trajectory(
        trajectory=trajectory,
        expected_tools={"search_documents", "query_database", "calculate"},
        ground_truth_facts=["NovaTech AI", "61.0M", "39.0M"],
    )

    assert eval_result["success"] is True
    assert eval_result["failure_mode"] == AgentFailureMode.SUCCESS.value
    assert eval_result["tool_accuracy"] == 1.0
    assert eval_result["redundancy_ratio"] == 0.0
    assert eval_result["fact_recall"] >= 0.6
    assert eval_result["step_count"] >= 3
