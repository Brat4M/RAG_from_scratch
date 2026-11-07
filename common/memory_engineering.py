"""Memory Engineering for Production RAG Systems.

Provides high-performance, 100% offline-executable, CPU-friendly memory architectures:
1. Short-Term & Working Memory: TokenBufferMemory, SummaryBufferMemory, WorkingMemoryScratchpad.
2. Long-Term Semantic Episodic Memory: Multi-tenant vector store, cosine retrieval, and
   Ebbinghaus forgetting curve decay with spaced-repetition reinforcement.
3. Entity Knowledge Graph / Fact Store: Structured entity-attribute-value triples, conflict detection.
4. Memory Consolidation & Pruning: Semantic deduplication, contradiction resolution, decay pruning.
5. Privacy Filter & PII Guardrails: Automated redaction of sensitive identifiers (API keys, PII).
6. MemoryEvaluator: Diagnostics for context window efficiency, contradiction rate, and tenant isolation.
"""

from __future__ import annotations

import math
import os
import re
import time
import uuid
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from common.query_transformations import DeterministicSemanticEmbedder


# ---------------------------------------------------------------------------
# 0. Helper Functions & Token Accounting
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Heuristic token estimation (~1.33 tokens per whitespace-separated word).
    
    Guarantees consistent, deterministic offline token budgeting without external BPE tokenizers.
    """
    if not text:
        return 0
    words = text.split()
    return max(1, math.ceil(len(words) * 1.33))


# ---------------------------------------------------------------------------
# 1. Dialogue & Short-Term Memory Buffers
# ---------------------------------------------------------------------------

@dataclass
class ChatMessage:
    """Represents a single conversational turn in dialogue state."""
    role: str  # 'system', 'user', 'assistant', 'tool'
    content: str
    timestamp: float = field(default_factory=time.time)
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __post_init__(self):
        if self.token_count <= 0:
            self.token_count = estimate_tokens(self.content)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }

    def format_line(self) -> str:
        return f"{self.role.capitalize()}: {self.content}"


class SlidingWindowMemory:
    """Sliding-window conversational buffer maintaining the last K interaction turns."""

    def __init__(self, k_turns: int = 5):
        if k_turns <= 0:
            raise ValueError("k_turns must be positive")
        self.k_turns = k_turns
        self.messages: List[ChatMessage] = []

    def add_message(self, role: str, content: str, **kwargs) -> ChatMessage:
        msg = ChatMessage(role=role, content=content, **kwargs)
        self.messages.append(msg)
        return msg

    def get_messages(self) -> List[ChatMessage]:
        """Return the most recent K turns while preserving any initial system message."""
        if not self.messages:
            return []
        system_msgs = [m for m in self.messages if m.role == "system"]
        non_system_msgs = [m for m in self.messages if m.role != "system"]
        recent_non_system = non_system_msgs[-self.k_turns:] if self.k_turns > 0 else []
        return system_msgs + recent_non_system

    def get_formatted_history(self) -> str:
        return "\n".join(m.format_line() for m in self.get_messages())

    def clear(self) -> None:
        self.messages.clear()


class TokenBufferMemory:
    """Strict token-bounded conversation buffer with system prompt pinning."""

    def __init__(self, max_tokens: int = 256):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = max_tokens
        self.messages: List[ChatMessage] = []

    def add_message(self, role: str, content: str, **kwargs) -> ChatMessage:
        msg = ChatMessage(role=role, content=content, **kwargs)
        self.messages.append(msg)
        self._enforce_budget()
        return msg

    def _enforce_budget(self) -> None:
        system_msgs = [m for m in self.messages if m.role == "system"]
        non_system = [m for m in self.messages if m.role != "system"]

        sys_tokens = sum(m.token_count for m in system_msgs)
        budget = max(0, self.max_tokens - sys_tokens)

        # Evict oldest non-system turns until remaining fit into budget
        retained: List[ChatMessage] = []
        accum = 0
        for m in reversed(non_system):
            if accum + m.token_count <= budget:
                retained.append(m)
                accum += m.token_count
            else:
                break
        retained.reverse()
        self.messages = system_msgs + retained

    def get_messages(self) -> List[ChatMessage]:
        return list(self.messages)

    def total_tokens(self) -> int:
        return sum(m.token_count for m in self.messages)

    def get_formatted_history(self) -> str:
        return "\n".join(m.format_line() for m in self.messages)

    def clear(self) -> None:
        self.messages.clear()


class SummaryBufferMemory:
    """Progressive summary buffer memory.
    
    Retains the most recent K turns verbatim, and compresses older evicted dialogue
    turns into an evolving running summary to prevent catastrophic context drift.
    """

    def __init__(self, recent_window_size: int = 3, max_tokens: int = 300):
        self.recent_window_size = recent_window_size
        self.max_tokens = max_tokens
        self.raw_history: List[ChatMessage] = []
        self.running_summary: str = ""

    def add_message(self, role: str, content: str, **kwargs) -> ChatMessage:
        msg = ChatMessage(role=role, content=content, **kwargs)
        self.raw_history.append(msg)
        self._consolidate()
        return msg

    def _consolidate(self) -> None:
        """When history exceeds window size or token budget, summarize oldest turns."""
        non_system = [m for m in self.raw_history if m.role != "system"]
        if len(non_system) > self.recent_window_size:
            to_summarize = non_system[:-self.recent_window_size]
            recent = non_system[-self.recent_window_size:]

            # Generate synthetic summary extraction offline
            new_notes: List[str] = []
            for m in to_summarize:
                # Extract key phrases/intent from evicted message
                clean = re.sub(r"[^\w\s]", "", m.content).strip()
                if clean:
                    new_notes.append(f"{m.role} mentioned '{m.content[:50]}'")

            delta_summary = "; ".join(new_notes)
            if self.running_summary:
                self.running_summary = f"{self.running_summary} | Then {delta_summary}"
            else:
                self.running_summary = f"Prior dialogue summary: {delta_summary}"

            # Preserve system messages plus recent window
            sys_msgs = [m for m in self.raw_history if m.role == "system"]
            self.raw_history = sys_msgs + recent

    def get_context_for_prompt(self) -> str:
        parts: List[str] = []
        if self.running_summary:
            parts.append(f"[System Memory Summary]: {self.running_summary}")
        for m in self.raw_history:
            parts.append(m.format_line())
        return "\n".join(parts)

    def clear(self) -> None:
        self.raw_history.clear()
        self.running_summary = ""


class WorkingMemoryScratchpad:
    """Active scratchpad for tracking execution goals, active entities, and constraints."""

    def __init__(self):
        self.goal: str = ""
        self.constraints: List[str] = []
        self.variables: Dict[str, Any] = {}
        self.unresolved_questions: List[str] = []

    def set_goal(self, goal: str) -> None:
        self.goal = goal

    def add_constraint(self, constraint: str) -> None:
        if constraint not in self.constraints:
            self.constraints.append(constraint)

    def set_variable(self, key: str, value: Any) -> None:
        self.variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        return self.variables.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "constraints": list(self.constraints),
            "variables": dict(self.variables),
            "unresolved_questions": list(self.unresolved_questions),
        }

    def format_scratchpad(self) -> str:
        lines = [f"[Current Goal]: {self.goal or 'None'}"]
        if self.constraints:
            lines.append(f"[Active Constraints]: {', '.join(self.constraints)}")
        if self.variables:
            vars_str = ", ".join(f"{k}={v}" for k, v in self.variables.items())
            lines.append(f"[Active State]: {vars_str}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Privacy Filter & PII Guardrails
# ---------------------------------------------------------------------------

class PrivacyFilter:
    """Automated sanitization engine to prevent PII leakage into long-term memories."""

    PATTERNS: List[Tuple[str, str, re.Pattern]] = [
        ("EMAIL", "[REDACTED_EMAIL]", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")),
        ("PHONE", "[REDACTED_PHONE]", re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
        ("CREDIT_CARD", "[REDACTED_CC]", re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")),
        ("SSN", "[REDACTED_SSN]", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
        ("API_KEY", "[REDACTED_KEY]", re.compile(r"\b(?:gsk_[a-zA-Z0-9]{20,}|sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,})\b")),
    ]

    def sanitize(self, text: str) -> Tuple[str, List[str]]:
        """Sanitize text by replacing sensitive PII with type tokens.
        
        Returns:
            Tuple of (sanitized_text, list_of_redaction_types)
        """
        redacted_types: List[str] = []
        sanitized = text
        for ptype, token, pattern in self.PATTERNS:
            if pattern.search(sanitized):
                redacted_types.append(ptype)
                sanitized = pattern.sub(token, sanitized)
        return sanitized, redacted_types


# ---------------------------------------------------------------------------
# 3. Long-Term Semantic Episodic Memory Store & Ebbinghaus Forgetting Curve
# ---------------------------------------------------------------------------

@dataclass
class EpisodicMemoryRecord:
    """Long-term episodic and semantic memory record."""
    memory_id: str
    user_id: str
    session_id: str
    content: str
    timestamp: float
    importance_score: float = 0.5  # [0.0, 1.0]
    last_accessed_at: float = 0.0
    access_count: int = 0
    embedding: Optional[np.ndarray] = None
    entities: List[str] = field(default_factory=list)
    category: str = "fact"  # 'preference', 'fact', 'event', 'instruction'
    is_active: bool = True
    superseded_by: Optional[str] = None

    def __post_init__(self):
        if self.last_accessed_at == 0.0:
            self.last_accessed_at = self.timestamp


@dataclass
class MemoryRetrievalResult:
    """Scored memory retrieval outcome."""
    record: EpisodicMemoryRecord
    score: float
    similarity_score: float
    recency_score: float
    importance_score: float


class SemanticEpisodicStore:
    """Multi-tenant semantic episodic memory store with Ebbinghaus forgetting decay.
    
    Computes hybrid memory retrieval combining:
    1. Cosine similarity between query and memory embedding.
    2. Intrinsic importance score.
    3. Temporal decay according to the Ebbinghaus Forgetting Curve with
       spaced-repetition reinforcement:
         lambda_eff = decay_rate / (1.0 + ln(1 + access_count))
         S_recency = exp(- lambda_eff * delta_t / half_life)
    """

    def __init__(
        self,
        embedder: Optional[DeterministicSemanticEmbedder] = None,
        half_life_seconds: float = 86400.0,  # 24 hours
        default_decay_rate: float = 0.693,   # ln(2)
    ):
        self.embedder = embedder or DeterministicSemanticEmbedder(dim=128)
        self.half_life_seconds = half_life_seconds
        self.default_decay_rate = default_decay_rate
        # Multi-tenant storage: {user_id: {memory_id: EpisodicMemoryRecord}}
        self._stores: Dict[str, Dict[str, EpisodicMemoryRecord]] = {}

    def add_memory(
        self,
        user_id: str,
        content: str,
        session_id: str = "default",
        importance_score: float = 0.5,
        timestamp: Optional[float] = None,
        category: str = "fact",
        entities: Optional[List[str]] = None,
    ) -> EpisodicMemoryRecord:
        """Add a memory entry with automatic vector embedding."""
        if user_id not in self._stores:
            self._stores[user_id] = {}

        now = timestamp if timestamp is not None else time.time()
        mem_id = str(uuid.uuid4())[:8]
        emb = self.embedder.embed(content)

        record = EpisodicMemoryRecord(
            memory_id=mem_id,
            user_id=user_id,
            session_id=session_id,
            content=content,
            timestamp=now,
            importance_score=float(np.clip(importance_score, 0.0, 1.0)),
            last_accessed_at=now,
            access_count=0,
            embedding=emb,
            entities=entities or [],
            category=category,
            is_active=True,
        )
        self._stores[user_id][mem_id] = record
        return record

    def compute_recency_score(
        self,
        record: EpisodicMemoryRecord,
        current_time: float,
    ) -> float:
        """Compute Ebbinghaus retention score with spaced repetition scaling.
        
        Formula:
            delta_t = max(0, current_time - record.last_accessed_at)
            lambda_eff = default_decay_rate / (1.0 + math.log1p(record.access_count))
            retention = math.exp(- lambda_eff * (delta_t / self.half_life_seconds))
        """
        delta_t = max(0.0, current_time - record.last_accessed_at)
        # Spaced repetition: more accesses reduce forgetting rate
        lambda_eff = self.default_decay_rate / (1.0 + math.log1p(record.access_count))
        ratio = delta_t / max(1.0, self.half_life_seconds)
        return float(math.exp(-lambda_eff * ratio))

    def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: int = 3,
        alpha_sim: float = 0.50,
        beta_imp: float = 0.25,
        gamma_rec: float = 0.25,
        current_time: Optional[float] = None,
        include_superseded: bool = False,
    ) -> List[MemoryRetrievalResult]:
        """Retrieve memories for a user ranked by hybrid similarity, importance, and recency.
        
        Strict tenant isolation: Only inspects memories indexed under `user_id`.
        """
        if user_id not in self._stores:
            return []

        user_memories = [
            m for m in self._stores[user_id].values()
            if include_superseded or m.is_active
        ]
        if not user_memories:
            return []

        now = current_time if current_time is not None else time.time()
        q_emb = self.embedder.embed(query)

        results: List[MemoryRetrievalResult] = []
        for mem in user_memories:
            # Cosine similarity
            sim = 0.0
            if mem.embedding is not None:
                sim = float(np.dot(q_emb, mem.embedding))
                sim = max(0.0, sim)  # Clamp to [0, 1]

            # Recency
            rec = self.compute_recency_score(mem, now)

            # Importance
            imp = mem.importance_score

            # Weighted composite score
            score = (alpha_sim * sim) + (beta_imp * imp) + (gamma_rec * rec)

            results.append(
                MemoryRetrievalResult(
                    record=mem,
                    score=score,
                    similarity_score=sim,
                    recency_score=rec,
                    importance_score=imp,
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def record_access(self, user_id: str, memory_id: str, current_time: Optional[float] = None) -> None:
        """Register an access event, updating recency timestamp and incrementing count."""
        if user_id in self._stores and memory_id in self._stores[user_id]:
            mem = self._stores[user_id][memory_id]
            mem.access_count += 1
            mem.last_accessed_at = current_time if current_time is not None else time.time()

    def get_all_user_memories(self, user_id: str) -> List[EpisodicMemoryRecord]:
        if user_id not in self._stores:
            return []
        return list(self._stores[user_id].values())


# ---------------------------------------------------------------------------
# 4. Entity Knowledge Graph & Structured Fact Store
# ---------------------------------------------------------------------------

@dataclass
class EntityFact:
    """Structured knowledge tuple extracted from conversational discourse."""
    fact_id: str
    user_id: str
    entity: str
    attribute: str
    value: str
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    source_turn_id: Optional[str] = None
    is_active: bool = True
    superseded_by: Optional[str] = None

    def key(self) -> Tuple[str, str]:
        return (self.entity.strip().lower(), self.attribute.strip().lower())

    def format_triple(self) -> str:
        return f"({self.entity}, {self.attribute}, {self.value})"


class EntityFactStore:
    """Key-value entity and attribute memory store with automatic conflict detection."""

    def __init__(self):
        # Multi-tenant: {user_id: { (entity, attribute): List[EntityFact] }}
        self._stores: Dict[str, Dict[Tuple[str, str], List[EntityFact]]] = {}

    def upsert_fact(
        self,
        user_id: str,
        entity: str,
        attribute: str,
        value: str,
        confidence: float = 1.0,
        timestamp: Optional[float] = None,
        source_turn_id: Optional[str] = None,
    ) -> Tuple[EntityFact, Optional[EntityFact]]:
        """Insert a fact. If an active fact exists for (entity, attribute), supersede it.
        
        Returns:
            Tuple of (new_fact, superseded_fact_or_None)
        """
        if user_id not in self._stores:
            self._stores[user_id] = {}

        now = timestamp if timestamp is not None else time.time()
        fact_id = str(uuid.uuid4())[:8]
        key = (entity.strip().lower(), attribute.strip().lower())

        new_fact = EntityFact(
            fact_id=fact_id,
            user_id=user_id,
            entity=entity.strip(),
            attribute=attribute.strip(),
            value=value.strip(),
            confidence=confidence,
            timestamp=now,
            source_turn_id=source_turn_id,
            is_active=True,
        )

        superseded: Optional[EntityFact] = None
        if key in self._stores[user_id]:
            # Look for existing active fact
            for old_fact in self._stores[user_id][key]:
                if old_fact.is_active:
                    old_fact.is_active = False
                    old_fact.superseded_by = new_fact.fact_id
                    superseded = old_fact
            self._stores[user_id][key].append(new_fact)
        else:
            self._stores[user_id][key] = [new_fact]

        return new_fact, superseded

    def get_active_facts(self, user_id: str, entity: Optional[str] = None) -> List[EntityFact]:
        if user_id not in self._stores:
            return []
        active = []
        for (ent, _), facts in self._stores[user_id].items():
            if entity is not None and ent != entity.strip().lower():
                continue
            for f in facts:
                if f.is_active:
                    active.append(f)
        return active

    def format_knowledge_graph(self, user_id: str) -> str:
        facts = self.get_active_facts(user_id)
        if not facts:
            return "No structured entities recorded."
        return "\n".join(f"- {f.entity}'s {f.attribute} is {f.value}" for f in facts)


# ---------------------------------------------------------------------------
# 5. Fact Extractor (Heuristic & Offline)
# ---------------------------------------------------------------------------

class FactExtractor:
    """Rule-based and pattern-driven entity/preference extraction engine."""

    # Heuristic regex patterns for extracting user preferences and biographical attributes
    PATTERNS = [
        # Preferences: "I prefer X", "I like X", "My favorite X is Y"
        (re.compile(r"i (?:prefer|love|like)\s+([^.,;!?]+)", re.I), "User", "preference"),
        (re.compile(r"my favorite\s+([\w\s]+?)\s+is\s+([^.,;!?]+)", re.I), "User", "favorite_{0}"),
        # Location: "I live in X", "I moved to X"
        (re.compile(r"i (?:live in|reside in|moved to)\s+([^.,;!?]+)", re.I), "User", "location"),
        # Profession: "I work as an? X", "I am a Y"
        (re.compile(r"i work as an?\s+([^.,;!?]+)", re.I), "User", "occupation"),
        (re.compile(r"i am an?\s+([\w\s]+?)(?:at|in|\.|$)", re.I), "User", "role"),
        # Constraints: "I am allergic to X", "I don't eat Y"
        (re.compile(r"i am allergic to\s+([^.,;!?]+)", re.I), "User", "allergy"),
        (re.compile(r"i (?:don't|do not) (?:eat|consume)\s+([^.,;!?]+)", re.I), "User", "dietary_restriction"),
    ]

    def extract_facts(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract candidate (entity, attribute, value) triples from text."""
        extracted: List[Tuple[str, str, str]] = []
        for pattern, entity, attr_template in self.PATTERNS:
            match = pattern.search(text)
            if match:
                groups = match.groups()
                if len(groups) == 1:
                    attr = attr_template
                    val = groups[0].strip()
                    extracted.append((entity, attr, val))
                elif len(groups) == 2:
                    attr = attr_template.format(groups[0].strip().replace(" ", "_"))
                    val = groups[1].strip()
                    extracted.append((entity, attr, val))
        return extracted


# ---------------------------------------------------------------------------
# 6. Memory Consolidation & Pruning Algorithms
# ---------------------------------------------------------------------------

class MemoryConsolidator:
    """Manages memory lifecycle: deduplication, contradiction resolution, and pruning."""

    def __init__(self, embedder: Optional[DeterministicSemanticEmbedder] = None):
        self.embedder = embedder or DeterministicSemanticEmbedder(dim=128)

    def deduplicate(
        self,
        memories: List[EpisodicMemoryRecord],
        similarity_threshold: float = 0.88,
    ) -> List[EpisodicMemoryRecord]:
        """Detect and deduplicate near-identical memories, retaining the highest importance."""
        if not memories:
            return []

        active_memories = [m for m in memories if m.is_active]
        retained: List[EpisodicMemoryRecord] = []

        for candidate in active_memories:
            c_emb = candidate.embedding
            if c_emb is None:
                c_emb = self.embedder.embed(candidate.content)
                candidate.embedding = c_emb

            is_duplicate = False
            for existing in retained:
                e_emb = existing.embedding
                if e_emb is None:
                    e_emb = self.embedder.embed(existing.content)
                    existing.embedding = e_emb
                sim = float(np.dot(c_emb, e_emb))
                if sim >= similarity_threshold:
                    is_duplicate = True
                    # If candidate has higher importance, merge/upgrade
                    if candidate.importance_score > existing.importance_score:
                        existing.importance_score = candidate.importance_score
                        existing.content = candidate.content
                        existing.last_accessed_at = max(existing.last_accessed_at, candidate.last_accessed_at)
                    existing.access_count += candidate.access_count + 1
                    break
            if not is_duplicate:
                retained.append(candidate)

        return retained

    def prune_decayed_memories(
        self,
        store: SemanticEpisodicStore,
        user_id: str,
        retention_threshold: float = 0.15,
        max_capacity: int = 50,
        current_time: Optional[float] = None,
    ) -> List[str]:
        """Evict memories whose retention score drops below threshold when capacity is exceeded.
        
        Returns:
            List of pruned memory IDs.
        """
        all_mems = store.get_all_user_memories(user_id)
        if not all_mems:
            return []

        now = current_time if current_time is not None else time.time()
        scored_mems: List[Tuple[float, EpisodicMemoryRecord]] = []

        for m in all_mems:
            retention = store.compute_recency_score(m, now)
            # Permanent memories with high importance are protected from low retention eviction
            effective_score = retention * 0.7 + m.importance_score * 0.3
            scored_mems.append((effective_score, m))

        pruned_ids: List[str] = []
        # Sort by score ascending (lowest retained first)
        scored_mems.sort(key=lambda x: x[0])

        for score, mem in scored_mems:
            # Prune if below threshold OR if store size exceeds capacity
            remaining = len(all_mems) - len(pruned_ids)
            if (score < retention_threshold and mem.importance_score < 0.8) or (remaining > max_capacity):
                if user_id in store._stores and mem.memory_id in store._stores[user_id]:
                    del store._stores[user_id][mem.memory_id]
                    pruned_ids.append(mem.memory_id)

        return pruned_ids


# ---------------------------------------------------------------------------
# 7. Unified Memory Manager
# ---------------------------------------------------------------------------

class MemoryManager:
    """Unified memory coordinator integrating short-term buffers, episodic vectors, and entity store."""

    def __init__(
        self,
        max_buffer_tokens: int = 300,
        half_life_seconds: float = 86400.0,
    ):
        self.privacy_filter = PrivacyFilter()
        self.fact_extractor = FactExtractor()
        self.consolidator = MemoryConsolidator()
        self.episodic_store = SemanticEpisodicStore(half_life_seconds=half_life_seconds)
        self.fact_store = EntityFactStore()
        self.scratchpad = WorkingMemoryScratchpad()

        # Session-scoped short-term token buffers: {session_id: TokenBufferMemory}
        self.session_buffers: Dict[str, TokenBufferMemory] = {}
        self.max_buffer_tokens = max_buffer_tokens

    def get_session_buffer(self, session_id: str) -> TokenBufferMemory:
        if session_id not in self.session_buffers:
            self.session_buffers[session_id] = TokenBufferMemory(max_tokens=self.max_buffer_tokens)
        return self.session_buffers[session_id]

    def process_turn(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
        importance_score: float = 0.5,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Ingest a conversational turn through privacy filtering, buffer update, and memory extraction."""
        now = timestamp if timestamp is not None else time.time()

        # 1. Privacy Sanitization
        sanitized_user, user_redactions = self.privacy_filter.sanitize(user_message)
        sanitized_asst, asst_redactions = self.privacy_filter.sanitize(assistant_message)

        # 2. Update Short-Term Buffer
        buffer = self.get_session_buffer(session_id)
        u_msg = buffer.add_message("user", sanitized_user, timestamp=now)
        a_msg = buffer.add_message("assistant", sanitized_asst, timestamp=now)

        # 3. Extract Structured Facts
        extracted = self.fact_extractor.extract_facts(sanitized_user)
        new_facts = []
        for ent, attr, val in extracted:
            nf, _ = self.fact_store.upsert_fact(
                user_id=user_id,
                entity=ent,
                attribute=attr,
                value=val,
                timestamp=now,
                source_turn_id=u_msg.message_id,
            )
            new_facts.append(nf)

        # 4. Ingest Long-Term Episodic Memory if meaningful
        episodic_record = None
        if len(sanitized_user.split()) >= 3:
            episodic_record = self.episodic_store.add_memory(
                user_id=user_id,
                content=f"User: {sanitized_user} | Assistant: {sanitized_asst}",
                session_id=session_id,
                importance_score=importance_score,
                timestamp=now,
                entities=[e for e, _, _ in extracted],
            )

        return {
            "sanitized_user": sanitized_user,
            "redactions": user_redactions + asst_redactions,
            "new_facts": [f.format_triple() for f in new_facts],
            "episodic_memory_id": episodic_record.memory_id if episodic_record else None,
        }

    def build_augmented_prompt(
        self,
        user_id: str,
        session_id: str,
        query: str,
        top_k_episodic: int = 2,
        current_time: Optional[float] = None,
    ) -> str:
        """Construct prompt context combining:
        1. Entity Knowledge Graph facts
        2. Relevant episodic memories (with recency/importance ranking)
        3. Working memory scratchpad
        4. Short-term dialogue history
        """
        now = current_time if current_time is not None else time.time()
        sections: List[str] = []

        # 1. Facts
        facts = self.fact_store.get_active_facts(user_id)
        if facts:
            fact_lines = [f"- {f.entity}'s {f.attribute} is {f.value}" for f in facts]
            sections.append("### Known User Facts:\n" + "\n".join(fact_lines))

        # 2. Episodic Retrieval
        retrieved = self.episodic_store.retrieve(
            query=query,
            user_id=user_id,
            top_k=top_k_episodic,
            current_time=now,
        )
        if retrieved:
            ep_lines = []
            for r in retrieved:
                ep_lines.append(f"- [Relevance: {r.score:.2f}]: {r.record.content}")
                self.episodic_store.record_access(user_id, r.record.memory_id, now)
            sections.append("### Relevant Prior Interactions:\n" + "\n".join(ep_lines))

        # 3. Scratchpad
        if self.scratchpad.goal or self.scratchpad.constraints:
            sections.append("### Working Context:\n" + self.scratchpad.format_scratchpad())

        # 4. Short-term buffer
        buffer = self.get_session_buffer(session_id)
        history = buffer.get_formatted_history()
        if history:
            sections.append("### Current Conversation:\n" + history)

        sections.append(f"User: {query}\nAssistant:")
        return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# 8. Memory Evaluation & Diagnostic Harness
# ---------------------------------------------------------------------------

class MemoryEvaluator:
    """Diagnostic evaluator for testing memory precision, context efficiency, and isolation."""

    @staticmethod
    def evaluate_retrieval_precision(
        retrieved: List[MemoryRetrievalResult],
        expected_substrings: List[str],
    ) -> float:
        """Measure what fraction of expected ground-truth substrings appear in retrieved memories."""
        if not expected_substrings:
            return 1.0
        hits = 0
        retrieved_texts = [r.record.content.lower() for r in retrieved]
        for expected in expected_substrings:
            exp_clean = expected.lower()
            if any(exp_clean in t for t in retrieved_texts):
                hits += 1
        return hits / len(expected_substrings)

    @staticmethod
    def evaluate_multi_tenant_isolation(
        store: SemanticEpisodicStore,
        user_a: str,
        user_b: str,
    ) -> bool:
        """Verify absolute tenant isolation: querying as user_a can NEVER return user_b memories."""
        # Query with user_a id
        results_a = store.retrieve(query="anything", user_id=user_a, top_k=100)
        leaked = [r for r in results_a if r.record.user_id != user_a]
        return len(leaked) == 0

    @staticmethod
    def evaluate_contradiction_rate(facts: List[EntityFact]) -> float:
        """Calculate fraction of entity attributes that contain unresolved conflicting values."""
        if not facts:
            return 0.0
        attr_values: Dict[Tuple[str, str], Set[str]] = {}
        for f in facts:
            if not f.is_active:
                continue
            k = f.key()
            if k not in attr_values:
                attr_values[k] = set()
            attr_values[k].add(f.value.lower())

        conflicts = sum(1 for vals in attr_values.values() if len(vals) > 1)
        return conflicts / max(1, len(attr_values))
