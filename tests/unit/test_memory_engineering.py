"""Unit tests for Track 08 — Memory Engineering.

Verifies:
1. Short-Term Buffers: SlidingWindowMemory, TokenBufferMemory, SummaryBufferMemory, WorkingMemoryScratchpad.
2. PrivacyFilter: automated PII redaction (email, phone, credit card, SSN, API key).
3. Long-Term Semantic Episodic Store: vector retrieval, Ebbinghaus decay, spaced repetition, multi-tenant isolation.
4. EntityFactStore: entity-attribute-value triples, conflict detection, superseded revisions.
5. FactExtractor: rule-based preference and entity extraction.
6. MemoryConsolidator: semantic deduplication and decay-based pruning.
7. MemoryManager: unified multi-tier memory coordination and augmented prompt synthesis.
8. MemoryEvaluator: diagnostic metrics.
"""

import time
import pytest
import numpy as np

from common.memory_engineering import (
    ChatMessage,
    SlidingWindowMemory,
    TokenBufferMemory,
    SummaryBufferMemory,
    WorkingMemoryScratchpad,
    PrivacyFilter,
    EpisodicMemoryRecord,
    SemanticEpisodicStore,
    EntityFact,
    EntityFactStore,
    FactExtractor,
    MemoryConsolidator,
    MemoryManager,
    MemoryEvaluator,
    estimate_tokens,
)


# ---------------------------------------------------------------------------
# 1. Token Estimation & ChatMessage Tests
# ---------------------------------------------------------------------------

def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world") == 3
    assert estimate_tokens("this is a longer sample sentence for estimation") >= 8


def test_chat_message_initialization_and_formatting():
    msg = ChatMessage(role="user", content="Deploy the model to production.")
    assert msg.role == "user"
    assert msg.token_count > 0
    assert msg.format_line() == "User: Deploy the model to production."
    assert "token_count" in msg.to_dict()


# ---------------------------------------------------------------------------
# 2. Short-Term Buffers & Working Memory Tests
# ---------------------------------------------------------------------------

def test_sliding_window_memory_preserves_system_prompt():
    mem = SlidingWindowMemory(k_turns=2)
    mem.add_message("system", "You are an autonomous engineering assistant.")
    mem.add_message("user", "Turn 1")
    mem.add_message("assistant", "Response 1")
    mem.add_message("user", "Turn 2")
    mem.add_message("assistant", "Response 2")

    messages = mem.get_messages()
    # Should contain system message + last 2 turns
    assert len(messages) == 3
    assert messages[0].role == "system"
    assert messages[1].content == "Turn 2"
    assert messages[2].content == "Response 2"


def test_token_buffer_memory_budget_enforcement():
    # Budget 30 tokens
    mem = TokenBufferMemory(max_tokens=30)
    mem.add_message("system", "System instruction.")
    # Add several user/asst turns
    for i in range(10):
        mem.add_message("user", f"User query message number {i} with several words.")
        mem.add_message("assistant", f"Assistant response {i} explaining details.")

    assert mem.total_tokens() <= 30
    messages = mem.get_messages()
    assert messages[0].role == "system"


def test_summary_buffer_memory_compression():
    mem = SummaryBufferMemory(recent_window_size=2)
    mem.add_message("system", "You are a concise assistant.")
    mem.add_message("user", "I need help with Docker containerization.")
    mem.add_message("assistant", "Docker packages applications into portable containers.")
    mem.add_message("user", "How do I configure volumes?")
    mem.add_message("assistant", "Volumes persist data outside the container lifecycle.")
    mem.add_message("user", "What about networking?")
    mem.add_message("assistant", "Bridge networks connect containers on the same host.")

    context = mem.get_context_for_prompt()
    assert "[System Memory Summary]:" in context
    assert "networking" in context.lower()
    # Oldest turns should have been compressed into running summary
    assert "docker" in mem.running_summary.lower()


def test_working_memory_scratchpad():
    scratchpad = WorkingMemoryScratchpad()
    scratchpad.set_goal("Migrate ChromaDB to production cluster")
    scratchpad.add_constraint("Zero downtime")
    scratchpad.add_constraint("Preserve vector embeddings")
    scratchpad.set_variable("current_node", "node-alpha-01")

    formatted = scratchpad.format_scratchpad()
    assert "Migrate ChromaDB" in formatted
    assert "Zero downtime" in formatted
    assert "node-alpha-01" in formatted


# ---------------------------------------------------------------------------
# 3. Privacy Filter Tests
# ---------------------------------------------------------------------------

def test_privacy_filter_redacts_sensitive_identifiers():
    filter_engine = PrivacyFilter()
    raw = (
        "Contact me at alice.engineer@enterprise.io or call 415-555-2671. "
        "My SSN is 123-45-6789 and my Groq key is gsk_abcdef1234567890abcdef1234567890."
    )
    sanitized, redactions = filter_engine.sanitize(raw)
    assert "[REDACTED_EMAIL]" in sanitized
    assert "alice.engineer@enterprise.io" not in sanitized
    assert "[REDACTED_PHONE]" in sanitized
    assert "415-555-2671" not in sanitized
    assert "[REDACTED_SSN]" in sanitized
    assert "123-45-6789" not in sanitized
    assert "[REDACTED_KEY]" in sanitized
    assert "gsk_" not in sanitized
    assert set(redactions) == {"EMAIL", "PHONE", "SSN", "API_KEY"}


# ---------------------------------------------------------------------------
# 4. Long-Term Semantic Episodic Store & Ebbinghaus Curve Tests
# ---------------------------------------------------------------------------

def test_semantic_episodic_store_retrieval_and_similarity():
    store = SemanticEpisodicStore()
    store.add_memory("user_101", "User prefers functional programming in Python and Scala", category="preference")
    store.add_memory("user_101", "User deployed Kubernetes Helm charts to staging", category="event")
    store.add_memory("user_101", "User is learning vector retrieval algorithms and HNSW indexing", category="event")

    results = store.retrieve("What programming paradigms does the user like?", user_id="user_101", top_k=2)
    assert len(results) > 0
    top_hit = results[0]
    assert "functional programming" in top_hit.record.content.lower()
    assert top_hit.similarity_score > 0.0


def test_ebbinghaus_decay_and_spaced_repetition():
    # 1-day half-life
    store = SemanticEpisodicStore(half_life_seconds=86400.0)
    base_time = 1000000.0

    # Add memory at base_time
    mem = store.add_memory("user_1", "User loves PyTorch", timestamp=base_time)

    # 1. Immediate recency score (delta_t = 0)
    score_t0 = store.compute_recency_score(mem, current_time=base_time)
    assert pytest.approx(score_t0, 0.01) == 1.0

    # 2. Recency score after 1 half-life (24 hours = 86400 seconds)
    score_t1 = store.compute_recency_score(mem, current_time=base_time + 86400.0)
    # Retention should drop to ~0.50
    assert pytest.approx(score_t1, 0.05) == 0.50

    # 3. Spaced repetition effect: Repeated access reinforces memory retention
    mem.access_count = 5  # Accessed 5 times
    score_reinforced = store.compute_recency_score(mem, current_time=base_time + 86400.0)
    # Reinforced memory decays much slower than unaccessed memory
    assert score_reinforced > score_t1
    assert score_reinforced > 0.70


def test_multi_tenant_isolation():
    store = SemanticEpisodicStore()
    store.add_memory("user_alice", "Alice secret API project codename: Horizon")
    store.add_memory("user_bob", "Bob project codename: Pegasus")

    # Alice queries
    results_alice = store.retrieve("codename", user_id="user_alice", top_k=5)
    assert len(results_alice) == 1
    assert "Horizon" in results_alice[0].record.content
    assert "Pegasus" not in results_alice[0].record.content

    # Evaluator check
    assert MemoryEvaluator.evaluate_multi_tenant_isolation(store, "user_alice", "user_bob")


# ---------------------------------------------------------------------------
# 5. EntityFactStore & FactExtractor Tests
# ---------------------------------------------------------------------------

def test_fact_extractor_heuristics():
    extractor = FactExtractor()
    text = "I prefer PostgreSQL over MySQL. I live in Tokyo and I work as a Machine Learning Engineer."
    extracted = extractor.extract_facts(text)
    assert len(extracted) >= 3

    facts_dict = {f"{attr}": val for _, attr, val in extracted}
    assert "postgresql" in facts_dict.get("preference", "").lower()
    assert "tokyo" in facts_dict.get("location", "").lower()


def test_entity_fact_store_conflict_resolution():
    store = EntityFactStore()
    # Initial location
    f1, sup1 = store.upsert_fact("user_1", "User", "location", "London", timestamp=100.0)
    assert sup1 is None
    assert f1.is_active

    # Updated location -> London should be superseded by Berlin
    f2, sup2 = store.upsert_fact("user_1", "User", "location", "Berlin", timestamp=200.0)
    assert sup2 is not None
    assert sup2.fact_id == f1.fact_id
    assert not f1.is_active
    assert f1.superseded_by == f2.fact_id

    active_facts = store.get_active_facts("user_1")
    assert len(active_facts) == 1
    assert active_facts[0].value == "Berlin"


# ---------------------------------------------------------------------------
# 6. Memory Consolidation & Pruning Tests
# ---------------------------------------------------------------------------

def test_memory_consolidator_deduplication():
    consolidator = MemoryConsolidator()
    m1 = EpisodicMemoryRecord(
        memory_id="m1",
        user_id="u1",
        session_id="s1",
        content="User enjoys hiking in the Swiss Alps on weekends",
        timestamp=100.0,
        importance_score=0.4,
    )
    m2 = EpisodicMemoryRecord(
        memory_id="m2",
        user_id="u1",
        session_id="s1",
        content="User enjoys hiking in the Swiss Alps on weekends",
        timestamp=200.0,
        importance_score=0.8,
    )

    deduped = consolidator.deduplicate([m1, m2], similarity_threshold=0.90)
    assert len(deduped) == 1
    # Merged record should inherit higher importance
    assert deduped[0].importance_score == 0.8


def test_memory_consolidator_pruning():
    store = SemanticEpisodicStore(half_life_seconds=100.0)
    base = 1000.0
    # Add old low-importance memory
    store.add_memory("u1", "Obsolete transient note", importance_score=0.1, timestamp=base)
    # Add fresh high-importance memory
    store.add_memory("u1", "Critical production database password policy", importance_score=0.95, timestamp=base + 1000.0)

    consolidator = MemoryConsolidator()
    pruned = consolidator.prune_decayed_memories(
        store=store,
        user_id="u1",
        retention_threshold=0.15,
        max_capacity=5,
        current_time=base + 1000.0,
    )
    assert len(pruned) == 1
    assert len(store.get_all_user_memories("u1")) == 1
    remaining = store.get_all_user_memories("u1")[0]
    assert "Critical production" in remaining.content


# ---------------------------------------------------------------------------
# 7. Unified MemoryManager & Evaluator Tests
# ---------------------------------------------------------------------------

def test_memory_manager_end_to_end():
    mgr = MemoryManager(max_buffer_tokens=200)

    # Process first turn with PII and preference
    res1 = mgr.process_turn(
        user_id="user_dev",
        session_id="sess_1",
        user_message="I prefer FastAPI for Python microservices. My phone is 555-123-4567.",
        assistant_message="FastAPI offers async performance and automatic OpenAPI documentation.",
    )
    assert "PHONE" in res1["redactions"]
    assert len(res1["new_facts"]) >= 1

    # Ingest another turn
    mgr.process_turn(
        user_id="user_dev",
        session_id="sess_1",
        user_message="How do I setup dependency injection in FastAPI?",
        assistant_message="Use the Depends() function in route parameters.",
    )

    # Set working scratchpad
    mgr.scratchpad.set_goal("Build production REST microservice")

    # Build augmented prompt
    prompt = mgr.build_augmented_prompt(
        user_id="user_dev",
        session_id="sess_1",
        query="Recommend an API framework for our team",
    )

    assert "Known User Facts:" in prompt
    assert "FastAPI" in prompt
    assert "Working Context:" in prompt
    assert "Build production REST" in prompt
    assert "Current Conversation:" in prompt


def test_memory_evaluator_metrics():
    # Contradiction rate evaluation
    f1 = EntityFact(fact_id="1", user_id="u", entity="User", attribute="role", value="Dev", is_active=True)
    f2 = EntityFact(fact_id="2", user_id="u", entity="User", attribute="role", value="Architect", is_active=True)
    # Both active => contradiction!
    contradiction_rate = MemoryEvaluator.evaluate_contradiction_rate([f1, f2])
    assert contradiction_rate > 0.0

    # If f1 is inactive (superseded)
    f1.is_active = False
    assert MemoryEvaluator.evaluate_contradiction_rate([f1, f2]) == 0.0
