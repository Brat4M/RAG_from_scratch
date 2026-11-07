"""Unit tests for Track 06 — LangGraph Stateful RAG.

Validates StateGraph construction, node execution, conditional edge routing,
CRAG cyclical query rewriting, Self-RAG reflection, loop prevention budgets,
MemorySaver persistence, and Human-in-the-Loop (HITL) interrupt/resume workflows.
"""

import pytest
from langgraph.checkpoint.memory import MemorySaver
from common.langgraph_rag import (
    create_initial_state,
    DeterministicGraphRetriever,
    DocumentRelevanceGrader,
    GraphQueryTransformer,
    GraphAnswerGenerator,
    SelfRAGHallucinationGrader,
    SelfRAGAnswerGrader,
    build_crag_graph,
    build_self_rag_graph,
    build_adaptive_rag_graph,
    decide_crag_path,
)


def test_initial_state_creation():
    """Verify state initialization schema, defaults, and trace setup."""
    state = create_initial_state("What is CRAG?", max_retries=3)
    assert state["question"] == "What is CRAG?"
    assert state["original_question"] == "What is CRAG?"
    assert state["documents"] == []
    assert state["generation"] is None
    assert state["loop_count"] == 0
    assert state["max_retries"] == 3
    assert state["documents_status"] == "empty"
    assert state["trace"] == ["START"]


def test_retriever_ranking_and_filtering():
    """Verify deterministic retriever top-k lexical overlap ranking."""
    retriever = DeterministicGraphRetriever()
    results = retriever.retrieve("LangGraph cyclic state", top_k=2)
    assert len(results) >= 1
    assert any("LangGraph" in doc["page_content"] for doc in results)
    assert results[0]["metadata"]["doc_id"] == "doc-03"


def test_document_relevance_grader():
    """Verify document grading threshold and status classification."""
    grader = DocumentRelevanceGrader(overlap_threshold=1)
    relevant_doc = {
        "page_content": "LangGraph handles cyclic state machines.",
        "metadata": {"doc_id": "1"},
    }
    irrelevant_doc = {
        "page_content": "Quantum entanglement in condensed matter physics.",
        "metadata": {"doc_id": "2"},
    }

    filtered, status = grader.grade_documents([relevant_doc, irrelevant_doc], "What is LangGraph?")
    assert len(filtered) == 1
    assert filtered[0]["metadata"]["doc_id"] == "1"
    assert status == "relevant"

    # Fully irrelevant set
    filtered_none, status_none = grader.grade_documents([irrelevant_doc], "What is LangGraph?")
    assert len(filtered_none) == 0
    assert status_none == "irrelevant"


def test_query_transformer_expansion():
    """Verify that query transformation enriches query and logs iteration number."""
    transformer = GraphQueryTransformer()
    rewritten = transformer.transform("what is rag?", loop_count=1)
    assert "Retrieval-Augmented Generation" in rewritten
    assert "(iteration 1)" in rewritten


def test_answer_generator_grounding_and_refusal():
    """Verify grounded citation generation and empty-context refusal."""
    generator = GraphAnswerGenerator()

    # Grounded case
    docs = [{"page_content": "Fact A is verified.", "metadata": {"doc_id": "1"}}]
    ans = generator.generate("Query A", docs)
    assert "Based on verified context (1 sources):" in ans
    assert "Fact A is verified." in ans

    # Refusal case
    refusal = generator.generate("Query B", [])
    assert "Refusal:" in refusal


def test_self_rag_hallucination_and_utility_graders():
    """Verify self-reflection graders on grounded vs hallucinated content."""
    h_grader = SelfRAGHallucinationGrader()
    a_grader = SelfRAGAnswerGrader()

    docs = [{"page_content": "LangGraph supports human-in-the-loop breakpoints.", "metadata": {}}]

    # Grounded + Useful
    valid_gen = "LangGraph supports human-in-the-loop breakpoints."
    assert h_grader.grade(valid_gen, docs) == "grounded"
    assert a_grader.grade(valid_gen, "What does LangGraph support?") == "useful"

    # Hallucinated
    hallucinated_gen = "LangGraph uses secret extraterrestrial signals."
    assert h_grader.grade(hallucinated_gen, docs) == "hallucinated"

    # Off-topic / Not Useful
    offtopic_gen = "The weather in Seattle is rainy and cold."
    assert a_grader.grade(offtopic_gen, "What does LangGraph support?") == "not_useful"


def test_crag_direct_path():
    """Verify CRAG graph routes directly to generation when documents pass relevance."""
    app = build_crag_graph()
    state = create_initial_state("Explain chunking strategies and token boundary")
    output = app.invoke(state)

    assert "generate" in output["trace"]
    assert "fallback" not in output["trace"]
    assert output["generation"] is not None
    assert "Chunking" in output["generation"]


def test_crag_cyclical_rewrite_and_recovery():
    """Verify CRAG graph triggers transform_query loop upon relevance failure and then recovers."""
    app = build_crag_graph()
    # Query has 'cycle' which triggers synonym expansion on rewrite
    state = create_initial_state("Tell me about cycle mechanisms")
    output = app.invoke(state)

    assert "transform_query(cycle=1)" in output["trace"]
    assert "generate" in output["trace"]
    assert output["loop_count"] >= 1
    assert "LangGraph" in output["generation"]


def test_crag_loop_prevention_budget_exhaustion():
    """Verify that unresolvable queries terminate safely at fallback once max_retries is reached."""
    app = build_crag_graph()
    state = create_initial_state("Cosmological redshift and gravitational lensing", max_retries=2)
    output = app.invoke(state)

    assert output["loop_count"] == 2
    assert "fallback" in output["trace"]
    assert "generate" not in output["trace"]
    assert "Refusal:" in output["generation"]
    assert output["web_fallback_needed"] is True


def test_self_rag_graph_execution():
    """Verify Self-RAG graph routes through generation and self_eval."""
    app = build_self_rag_graph()
    state = create_initial_state("LangGraph stateful cyclic graphs")
    output = app.invoke(state)

    assert "retrieve" in output["trace"]
    assert "generate" in output["trace"]
    assert any("self_eval" in step for step in output["trace"])
    assert output["hallucination_grade"] == "grounded"
    assert output["answer_grade"] == "useful"


def test_memory_saver_checkpointing_and_thread_isolation():
    """Verify MemorySaver maintains persistent state per thread_id without cross-talk."""
    checkpointer = MemorySaver()
    app = build_crag_graph(checkpointer=checkpointer)

    config_1 = {"configurable": {"thread_id": "user-session-1"}}
    config_2 = {"configurable": {"thread_id": "user-session-2"}}

    state_1 = create_initial_state("Explain chunking strategies")
    state_2 = create_initial_state("Cosmological redshift", max_retries=1)

    app.invoke(state_1, config=config_1)
    app.invoke(state_2, config=config_2)

    persisted_1 = app.get_state(config_1)
    persisted_2 = app.get_state(config_2)

    assert persisted_1.values["original_question"] == "Explain chunking strategies"
    assert "generate" in persisted_1.values["trace"]

    assert persisted_2.values["original_question"] == "Cosmological redshift"
    assert "fallback" in persisted_2.values["trace"]


def test_human_in_the_loop_interrupt_and_resume():
    """Verify HITL breakpoint pauses execution before human_review and resumes after approval."""
    checkpointer = MemorySaver()
    app = build_adaptive_rag_graph(checkpointer=checkpointer, enable_hitl=True)
    config = {"configurable": {"thread_id": "hitl-approval-session"}}

    state = create_initial_state("LangGraph cyclic checkpointing")
    app.invoke(state, config=config)

    # Assert execution paused at breakpoint
    checkpoint_state = app.get_state(config)
    assert checkpoint_state.next == ("human_review",)
    assert "human_review" not in checkpoint_state.values["trace"]

    # Human reviews and updates state with approval
    app.update_state(config, {"human_approval": True})

    # Resume graph from breakpoint
    final_output = app.invoke(None, config=config)
    assert "human_review" in final_output["trace"]
    assert final_output["human_approval"] is True
    assert app.get_state(config).next == ()
