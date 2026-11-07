# Track 07 — Autonomous Agentic RAG and DeepAgent Systems

Multi-step tool calling, dynamic query planning, bounded execution loops, and multi-agent coordination.

## Overview

In traditional static RAG and stateful graphs, control flow is fixed or conditional on predetermined branches. **Track 07** introduces autonomous **Agentic RAG**, where the language model dynamically reasons about which tools to invoke, formulates action parameters, inspects environment observations, and reflects on goal satisfaction.

## Curriculum & Notebooks

| Notebook | Focus | Key Topics | Status |
|---|---|---|---|
| [`01_agentic_rag_and_deepagent.ipynb`](01_agentic_rag_and_deepagent.ipynb) | Autonomous Agentic RAG & Multi-Agent Coordination | Tool Registries & Permissions, ReAct Execution Loop, Call Signature Hashing, Plan-and-Solve Decomposition, DeepAgent Supervisor-Specialist Architecture, Quantitative Evaluation & Failure Taxonomy | **VALIDATED** |

## Core Architectural Components

Implemented in [`common/agentic_rag.py`](../../common/agentic_rag.py):
1. **Tooling & Access Control**:
   - `ToolPermission` tiers (`READ_ONLY`, `RESTRICTED`, `EXECUTE`, `ADMIN`).
   - `ToolRegistry` with OpenAI/Groq function calling schema generation.
   - `SafeCalculator` with AST sandboxing (no unsafe `eval`).
   - Standard tools: `search_documents`, `query_database` (read-only SQLite), `calculate`, `search_external_web`.
2. **Execution Budgets & Loop Guardrails**:
   - `AgentBudget`: iteration limits, tool call caps, and wall-clock timeouts.
   - Cycle detection via SHA-256 call signature hashing (`tool_name + args`).
3. **ReAct Loop (`ReActAgent`)**:
   - Autonomous Thought $\to$ Action $\to$ Observation $\to$ Reflection cycles.
   - 100% offline-executable deterministic multi-hop reasoning.
4. **Plan-and-Solve Engine (`PlanAndSolveEngine`)**:
   - Complex query decomposition into dependency DAGs (`ExecutionPlan`, `PlanStep`).
   - Dynamic replanning on intermediate step failure.
5. **DeepAgent Multi-Agent Coordinator (`DeepAgentCoordinator`)**:
   - Supervisor agent dispatching to `RetrievalSpecialist` and `DataAnalystSpecialist`.
   - Independent verification via `CriticSpecialist` (citation grounding checks).
6. **Agent Benchmarking & Failure Taxonomy (`AgentEvaluator`)**:
   - Metrics: Tool Selection Accuracy ($F_1$), Redundancy Ratio, Grounded Fact Recall.
   - Classification: `SUCCESS`, `INFINITE_LOOP`, `BUDGET_EXHAUSTED`, `TOOL_PARAMETER_ERROR`.

## Visual Assets

- [`assets/diagrams/track07_agentic_rag_react_loop.png`](../../assets/diagrams/track07_agentic_rag_react_loop.png)
- [`assets/diagrams/track07_plan_and_solve_multiagent.png`](../../assets/diagrams/track07_plan_and_solve_multiagent.png)

## Verification & Tests

Tested by 18 unit tests in [`tests/unit/test_agentic_rag.py`](../../tests/unit/test_agentic_rag.py):
```bash
.venv312/bin/pytest tests/unit/test_agentic_rag.py -v
```
