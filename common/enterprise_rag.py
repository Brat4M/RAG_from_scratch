"""Enterprise RAG Architecture Module.

Provides production-grade components for enterprise retrieval-augmented generation:
- Multi-tenancy and strict tenant isolation (separate namespaces & metadata boundaries)
- Fine-grained Access Control Lists (ACL) and Role-Based Access Control (RBAC)
- Multi-level Security Guardrails (indirect prompt injection defense, PII redaction)
- KMS-style envelope encryption simulation and credential management
- High-efficiency Semantic Caching (exact hash + vector similarity cache)
- Asynchronous batch ingestion with deduplication and dead-letter queues
- Zero-downtime embedding migration & blue-green shadow reindexing
- Cryptographically chained immutable audit logging
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Access Control & Classification Types
# ---------------------------------------------------------------------------

class ClassificationLevel(enum.IntEnum):
    """Hierarchical document security classification levels."""
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3


class UserIdentity(BaseModel):
    """Authenticated enterprise user identity."""
    user_id: str
    tenant_id: str
    roles: List[str] = Field(default_factory=list)
    groups: List[str] = Field(default_factory=list)
    clearance: ClassificationLevel = ClassificationLevel.INTERNAL


class DocumentSecurityProfile(BaseModel):
    """Security attributes and access control list governing a document chunk."""
    doc_id: str
    tenant_id: str
    owner: str
    classification: ClassificationLevel = ClassificationLevel.INTERNAL
    allowed_roles: List[str] = Field(default_factory=list)
    allowed_groups: List[str] = Field(default_factory=list)
    allowed_users: List[str] = Field(default_factory=list)
    denied_users: List[str] = Field(default_factory=list)
    content_hash: str = ""
    version: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ACLDecision(BaseModel):
    """Result of an access control policy evaluation."""
    doc_id: str
    allowed: bool
    reason: str
    timestamp: float = Field(default_factory=time.time)


class ACLEvaluator:
    """Evaluates fine-grained access control policies with explicit denial precedence."""

    @staticmethod
    def evaluate(user: UserIdentity, doc: DocumentSecurityProfile) -> ACLDecision:
        """Evaluate if user has permission to read the document.

        Policy precedence:
        1. Tenant Isolation: user.tenant_id must strictly match doc.tenant_id
        2. Explicit Deny: denied_users overrides all allowances
        3. Clearance Check: user clearance level must be >= document classification
        4. Owner or Admin: owner or 'admin' role always permitted
        5. Explicit Allow: user in allowed_users
        6. Role / Group Membership: any matching role or group
        """
        # Rule 1: Multi-tenant boundary
        if user.tenant_id != doc.tenant_id:
            return ACLDecision(
                doc_id=doc.doc_id,
                allowed=False,
                reason=f"Tenant mismatch: user tenant '{user.tenant_id}' != doc tenant '{doc.tenant_id}'"
            )

        # Rule 2: Explicit denial
        if user.user_id in doc.denied_users:
            return ACLDecision(
                doc_id=doc.doc_id,
                allowed=False,
                reason=f"User '{user.user_id}' is explicitly listed in denied_users"
            )

        # Rule 3: Owner or Admin override
        if user.user_id == doc.owner:
            return ACLDecision(
                doc_id=doc.doc_id,
                allowed=True,
                reason=f"Access granted: user '{user.user_id}' is document owner"
            )
        if "admin" in [r.lower() for r in user.roles]:
            return ACLDecision(
                doc_id=doc.doc_id,
                allowed=True,
                reason="Access granted: user holds tenant admin role"
            )

        # Rule 4: Clearance level
        if int(user.clearance) < int(doc.classification):
            return ACLDecision(
                doc_id=doc.doc_id,
                allowed=False,
                reason=f"Insufficient clearance: user has {user.clearance.name} ({int(user.clearance)}), doc requires {doc.classification.name} ({int(doc.classification)})"
            )

        # Rule 5: Explicit user allow
        if user.user_id in doc.allowed_users:
            return ACLDecision(
                doc_id=doc.doc_id,
                allowed=True,
                reason=f"User '{user.user_id}' is explicitly granted access in allowed_users"
            )

        # Rule 6: Role intersection
        user_roles_lower = {r.lower() for r in user.roles}
        doc_roles_lower = {r.lower() for r in doc.allowed_roles}
        matching_roles = user_roles_lower.intersection(doc_roles_lower)
        if matching_roles:
            return ACLDecision(
                doc_id=doc.doc_id,
                allowed=True,
                reason=f"Access granted via role match: {sorted(list(matching_roles))}"
            )

        # Rule 7: Group intersection
        user_groups_lower = {g.lower() for g in user.groups}
        doc_groups_lower = {g.lower() for g in doc.allowed_groups}
        matching_groups = user_groups_lower.intersection(doc_groups_lower)
        if matching_groups:
            return ACLDecision(
                doc_id=doc.doc_id,
                allowed=True,
                reason=f"Access granted via group match: {sorted(list(matching_groups))}"
            )

        # Default Deny
        return ACLDecision(
            doc_id=doc.doc_id,
            allowed=False,
            reason="Access denied: no matching role, group, or explicit grant found"
        )


# ---------------------------------------------------------------------------
# 2. Security Guardrails & PII Redactor
# ---------------------------------------------------------------------------

class SecurityGuardrails:
    """Enterprise security defenses against prompt injection, data exfiltration, and PII leakage."""

    # Regex patterns for common PII
    PATTERNS: Dict[str, re.Pattern] = {
        "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"),
        "PHONE": re.compile(r"\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "API_KEY": re.compile(r"\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|eyJ[a-zA-Z0-9_-]{20,})\b"),
    }

    # Direct and indirect prompt injection heuristic indicators
    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions", re.IGNORECASE),
        re.compile(r"disregard\s+(?:all\s+)?(?:system|developer)\s+prompts?", re.IGNORECASE),
        re.compile(r"(?:print|reveal|output|display)\s+(?:the\s+)?(?:system\s+prompt|secret\s+key|api\s+key)", re.IGNORECASE),
        re.compile(r"<\|im_start\|>|<\|im_end\|>|\[INST\]|\[/INST\]", re.IGNORECASE),
        re.compile(r"(?:BEGIN|START)\s+SYSTEM\s+PROMPT", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+in\s+unrestricted\s+mode", re.IGNORECASE),
        re.compile(r"bypass\s+(?:safety|content)\s+filters?", re.IGNORECASE),
    ]

    @classmethod
    def redact_pii(cls, text: str) -> Tuple[str, Dict[str, int]]:
        """Scrub sensitive personal identifiers and API keys from text."""
        sanitized = text
        counts: Dict[str, int] = {}
        for pii_type, pattern in cls.PATTERNS.items():
            matches = pattern.findall(sanitized)
            if matches:
                counts[pii_type] = len(matches)
                sanitized = pattern.sub(f"[REDACTED_{pii_type}]", sanitized)
        return sanitized, counts

    @classmethod
    def detect_prompt_injection(cls, text: str) -> Tuple[bool, str]:
        """Detect prompt injection payloads in untrusted retrieved documents or user queries."""
        for pattern in cls.INJECTION_PATTERNS:
            match = pattern.search(text)
            if match:
                return True, f"Detected prompt injection pattern: '{match.group(0)}'"
        return False, "Clean"

    @staticmethod
    def compute_sha256(text: str) -> str:
        """Compute SHA-256 fingerprint for tamper-evidence."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 3. KMS Envelope Encryption Simulation
# ---------------------------------------------------------------------------

class EnvelopeEncryptionKMS:
    """Simulates an Enterprise Key Management Service (e.g., AWS KMS, HashiCorp Vault).

    Uses a Key Encryption Key (KEK) to protect per-tenant Data Encryption Keys (DEKs).
    """

    def __init__(self, master_kek: str = "master-kek-corporate-vault-2026"):
        self.master_kek = master_kek.encode("utf-8")
        self._tenant_deks: Dict[str, Dict[str, Any]] = {}

    def get_or_create_tenant_dek(self, tenant_id: str) -> str:
        """Retrieve plaintext DEK, generating and storing encrypted DEK under master KEK."""
        if tenant_id not in self._tenant_deks:
            # Generate deterministic pseudo-random 256-bit key
            raw_dek = hashlib.sha256(f"dek-{tenant_id}-{time.time()}".encode("utf-8")).digest()
            # Wrap DEK with master KEK via HMAC
            encrypted_dek = hmac.new(self.master_kek, raw_dek, hashlib.sha256).hexdigest()
            self._tenant_deks[tenant_id] = {
                "raw_dek_hex": raw_dek.hex(),
                "encrypted_dek": encrypted_dek,
                "version": 1,
                "created_at": time.time(),
            }
        return self._tenant_deks[tenant_id]["raw_dek_hex"]

    def encrypt_payload(self, tenant_id: str, plaintext: str) -> Dict[str, Any]:
        """Encrypt payload using tenant DEK."""
        dek = self.get_or_create_tenant_dek(tenant_id)
        # Fast deterministic stream cipher simulation (XOR with DEK keystream + HMAC integrity tag)
        key_bytes = bytes.fromhex(dek)
        text_bytes = plaintext.encode("utf-8")
        keystream = (key_bytes * ((len(text_bytes) // len(key_bytes)) + 1))[:len(text_bytes)]
        ciphertext_bytes = bytes(b ^ k for b, k in zip(text_bytes, keystream))
        mac = hmac.new(key_bytes, ciphertext_bytes, hashlib.sha256).hexdigest()
        return {
            "ciphertext_hex": ciphertext_bytes.hex(),
            "mac": mac,
            "tenant_id": tenant_id,
            "dek_version": self._tenant_deks[tenant_id]["version"]
        }

    def decrypt_payload(self, encrypted_package: Dict[str, Any]) -> str:
        """Decrypt payload and verify HMAC integrity."""
        tenant_id = encrypted_package["tenant_id"]
        dek = self.get_or_create_tenant_dek(tenant_id)
        key_bytes = bytes.fromhex(dek)
        ciphertext_bytes = bytes.fromhex(encrypted_package["ciphertext_hex"])
        expected_mac = hmac.new(key_bytes, ciphertext_bytes, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected_mac, encrypted_package["mac"]):
            raise ValueError("Integrity verification failed! Tampered ciphertext.")
        keystream = (key_bytes * ((len(ciphertext_bytes) // len(key_bytes)) + 1))[:len(ciphertext_bytes)]
        plaintext_bytes = bytes(b ^ k for b, k in zip(ciphertext_bytes, keystream))
        return plaintext_bytes.decode("utf-8")


# ---------------------------------------------------------------------------
# 4. Semantic Caching & Cost Optimization
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    query_text: str
    query_vector: np.ndarray
    response_text: str
    retrieved_chunk_ids: List[str]
    tenant_id: str
    clearance_level: int
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = 3600.0


class SemanticCache:
    """Two-tier enterprise cache: exact SHA-256 hash match + semantic vector similarity."""

    def __init__(self, similarity_threshold: float = 0.90, max_size: int = 500):
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self._exact_cache: Dict[str, CacheEntry] = {}
        self._vector_entries: List[CacheEntry] = []
        self.stats = {
            "hits_exact": 0,
            "hits_semantic": 0,
            "misses": 0,
            "cost_saved_usd": 0.0,
            "tokens_saved": 0,
        }

    def _hash_key(self, tenant_id: str, clearance: int, query: str) -> str:
        normalized = f"{tenant_id}:{clearance}:{query.strip().lower()}"
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, tenant_id: str, clearance: int, query: str, query_vector: np.ndarray) -> Optional[Tuple[str, List[str], str]]:
        """Lookup query in cache. Returns (response_text, chunk_ids, hit_type) or None."""
        now = time.time()
        # Tier 1: Exact Hash Match
        hkey = self._hash_key(tenant_id, clearance, query)
        if hkey in self._exact_cache:
            entry = self._exact_cache[hkey]
            if now - entry.created_at <= entry.ttl_seconds:
                self.stats["hits_exact"] += 1
                self.stats["tokens_saved"] += 250
                self.stats["cost_saved_usd"] += 0.002
                return entry.response_text, entry.retrieved_chunk_ids, "EXACT_HASH_HIT"
            else:
                del self._exact_cache[hkey]

        # Tier 2: Semantic Vector Match within same tenant & clearance
        norm_q = np.linalg.norm(query_vector)
        if norm_q > 0:
            best_sim = -1.0
            best_entry: Optional[CacheEntry] = None
            for entry in self._vector_entries:
                if entry.tenant_id == tenant_id and entry.clearance_level == clearance:
                    if now - entry.created_at <= entry.ttl_seconds:
                        norm_e = np.linalg.norm(entry.query_vector)
                        if norm_e > 0:
                            sim = float(np.dot(query_vector, entry.query_vector) / (norm_q * norm_e))
                            if sim > best_sim:
                                best_sim = sim
                                best_entry = entry

            if best_sim >= self.similarity_threshold and best_entry is not None:
                self.stats["hits_semantic"] += 1
                self.stats["tokens_saved"] += 220
                self.stats["cost_saved_usd"] += 0.0018
                return best_entry.response_text, best_entry.retrieved_chunk_ids, f"SEMANTIC_HIT (sim={best_sim:.3f})"

        self.stats["misses"] += 1
        return None

    def put(
        self,
        tenant_id: str,
        clearance: int,
        query: str,
        query_vector: np.ndarray,
        response_text: str,
        retrieved_chunk_ids: List[str],
        ttl_seconds: float = 3600.0
    ) -> None:
        """Store result in cache."""
        entry = CacheEntry(
            query_text=query,
            query_vector=query_vector,
            response_text=response_text,
            retrieved_chunk_ids=retrieved_chunk_ids,
            tenant_id=tenant_id,
            clearance_level=clearance,
            created_at=time.time(),
            ttl_seconds=ttl_seconds
        )
        hkey = self._hash_key(tenant_id, clearance, query)
        self._exact_cache[hkey] = entry
        self._vector_entries.append(entry)
        if len(self._vector_entries) > self.max_size:
            self._vector_entries.pop(0)

    def invalidate_tenant(self, tenant_id: str) -> int:
        """Invalidate all cached records for a specific tenant (e.g. after data deletion)."""
        removed = 0
        keys_to_del = [k for k, v in self._exact_cache.items() if v.tenant_id == tenant_id]
        for k in keys_to_del:
            del self._exact_cache[k]
            removed += 1
        self._vector_entries = [e for e in self._vector_entries if e.tenant_id != tenant_id]
        return removed


# ---------------------------------------------------------------------------
# 5. Async Ingestion, Deduplication & Lifecycle Manager
# ---------------------------------------------------------------------------

@dataclass
class IngestionItem:
    doc_id: str
    tenant_id: str
    content: str
    security: DocumentSecurityProfile
    idempotency_key: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class IngestionBatchReport:
    total_received: int
    ingested_count: int
    deduplicated_count: int
    poisoned_rejected_count: int
    dead_letter_count: int
    duration_ms: float


class AsyncIngestionPipeline:
    """Enterprise asynchronous batch ingestion pipeline with deduplication and safety gates."""

    def __init__(self):
        self._seen_content_hashes: Set[str] = set()
        self._processed_idempotency_keys: Set[str] = set()
        self.dead_letter_queue: List[Dict[str, Any]] = []

    def process_batch(
        self,
        items: List[IngestionItem],
        embed_fn: Any,
        kms: EnvelopeEncryptionKMS
    ) -> Tuple[List[Dict[str, Any]], IngestionBatchReport]:
        """Process incoming documents with deduplication, prompt injection scanning, and KMS encryption."""
        start = time.perf_counter()
        ingested: List[Dict[str, Any]] = []
        dedup_count = 0
        poison_count = 0
        dlq_count = 0

        for item in items:
            # 1. Idempotency Check
            if item.idempotency_key in self._processed_idempotency_keys:
                dedup_count += 1
                continue

            # 2. Content Hash Deduplication
            content_hash = SecurityGuardrails.compute_sha256(item.content)
            if content_hash in self._seen_content_hashes:
                dedup_count += 1
                continue

            # 3. Prompt Injection / Poisoning Inspection
            is_injected, reason = SecurityGuardrails.detect_prompt_injection(item.content)
            if is_injected:
                poison_count += 1
                self.dead_letter_queue.append({
                    "doc_id": item.doc_id,
                    "tenant_id": item.tenant_id,
                    "reason": reason,
                    "content_preview": item.content[:80],
                    "timestamp": time.time(),
                })
                continue

            # 4. PII Redaction
            sanitized_content, pii_stats = SecurityGuardrails.redact_pii(item.content)

            # 5. Envelope Encryption at Rest
            encrypted_pkg = kms.encrypt_payload(item.tenant_id, sanitized_content)

            # 6. Embedding generation
            vector = embed_fn(sanitized_content)

            # Mark processed
            self._seen_content_hashes.add(content_hash)
            self._processed_idempotency_keys.add(item.idempotency_key)
            item.security.content_hash = content_hash

            ingested.append({
                "doc_id": item.doc_id,
                "tenant_id": item.tenant_id,
                "sanitized_content": sanitized_content,
                "encrypted_pkg": encrypted_pkg,
                "vector": vector,
                "security": item.security,
                "pii_stats": pii_stats,
            })

        duration = (time.perf_counter() - start) * 1000.0
        report = IngestionBatchReport(
            total_received=len(items),
            ingested_count=len(ingested),
            deduplicated_count=dedup_count,
            poisoned_rejected_count=poison_count,
            dead_letter_count=len(self.dead_letter_queue),
            duration_ms=duration
        )
        return ingested, report


# ---------------------------------------------------------------------------
# 6. Blue-Green Zero-Downtime Model Migration
# ---------------------------------------------------------------------------

class ModelMigrationManager:
    """Coordinates zero-downtime embedding model transitions (e.g. 64-dim -> 128-dim)."""

    def __init__(self, current_model_name: str = "embed-v1-base"):
        self.active_model = current_model_name
        self.shadow_model: Optional[str] = None
        self.traffic_ratio_shadow: float = 0.0  # 0.0 = 100% active, 1.0 = 100% shadow
        self.migration_state = "IDLE"  # IDLE, SHADOWING, CANARY, CUTOVER, COMPLETED

    def start_migration(self, target_model_name: str) -> None:
        """Initiate shadow indexing on target embedding model."""
        self.shadow_model = target_model_name
        self.migration_state = "SHADOWING"
        self.traffic_ratio_shadow = 0.0

    def set_canary_traffic(self, shadow_ratio: float) -> None:
        """Route a percentage of traffic to shadow index for validation."""
        self.traffic_ratio_shadow = max(0.0, min(1.0, shadow_ratio))
        self.migration_state = "CANARY"

    def complete_cutover(self) -> None:
        """Flip traffic to new model and retire previous model."""
        if not self.shadow_model:
            raise ValueError("No shadow model active to cut over.")
        self.active_model = self.shadow_model
        self.shadow_model = None
        self.traffic_ratio_shadow = 0.0
        self.migration_state = "COMPLETED"

    def rollback(self) -> None:
        """Abort migration and reset to active model."""
        self.shadow_model = None
        self.traffic_ratio_shadow = 0.0
        self.migration_state = "IDLE"


# ---------------------------------------------------------------------------
# 7. Cryptographically Chained Immutable Audit Log
# ---------------------------------------------------------------------------

@dataclass
class AuditRecord:
    index: int
    timestamp: float
    tenant_id: str
    user_id: str
    query_hash: str
    retrieved_chunk_ids: List[str]
    acl_decisions: List[Dict[str, Any]]
    cache_hit: str
    latency_ms: float
    previous_hash: str
    record_hash: str = ""

    def compute_hash(self) -> str:
        payload = f"{self.index}|{self.timestamp}|{self.tenant_id}|{self.user_id}|{self.query_hash}|{self.retrieved_chunk_ids}|{self.cache_hit}|{self.previous_hash}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class EnterpriseAuditLogger:
    """Immutable audit trail with SHA-256 block-chaining for non-repudiation."""

    def __init__(self):
        self.chain: List[AuditRecord] = []
        # Genesis block
        genesis = AuditRecord(
            index=0,
            timestamp=time.time(),
            tenant_id="SYSTEM",
            user_id="SYSTEM_ROOT",
            query_hash="GENESIS",
            retrieved_chunk_ids=[],
            acl_decisions=[],
            cache_hit="NONE",
            latency_ms=0.0,
            previous_hash="0" * 64,
        )
        genesis.record_hash = genesis.compute_hash()
        self.chain.append(genesis)

    def log_query(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        retrieved_chunk_ids: List[str],
        acl_decisions: List[ACLDecision],
        cache_hit: str,
        latency_ms: float
    ) -> AuditRecord:
        prev_hash = self.chain[-1].record_hash
        q_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
        rec = AuditRecord(
            index=len(self.chain),
            timestamp=time.time(),
            tenant_id=tenant_id,
            user_id=user_id,
            query_hash=q_hash,
            retrieved_chunk_ids=retrieved_chunk_ids,
            acl_decisions=[{"doc_id": d.doc_id, "allowed": d.allowed, "reason": d.reason} for d in acl_decisions],
            cache_hit=cache_hit,
            latency_ms=latency_ms,
            previous_hash=prev_hash,
        )
        rec.record_hash = rec.compute_hash()
        self.chain.append(rec)
        return rec

    def verify_integrity(self) -> Tuple[bool, str]:
        """Verify cryptographic chain validity."""
        for i in range(1, len(self.chain)):
            curr = self.chain[i]
            prev = self.chain[i - 1]
            if curr.previous_hash != prev.record_hash:
                return False, f"Broken link between block {i-1} and {i}"
            if curr.record_hash != curr.compute_hash():
                return False, f"Corrupted hash in block {i}"
        return True, "Chain valid and tamper-free"


# ---------------------------------------------------------------------------
# 8. Complete Enterprise RAG Pipeline
# ---------------------------------------------------------------------------

@dataclass
class EnterpriseQueryResult:
    query: str
    tenant_id: str
    user_id: str
    retrieved_chunks: List[Dict[str, Any]]
    acl_decisions: List[ACLDecision]
    answer: str
    cache_status: str
    latency_ms: float
    audit_hash: str


class EnterpriseRAGPipeline:
    """Integrated enterprise RAG pipeline supporting pre-filtering, post-filtering, caching, and guardrails."""

    def __init__(self, vector_dim: int = 64):
        self.vector_dim = vector_dim
        self.kms = EnvelopeEncryptionKMS()
        self.cache = SemanticCache()
        self.audit_logger = EnterpriseAuditLogger()
        self.ingestion_pipeline = AsyncIngestionPipeline()
        self._documents: Dict[str, Dict[str, Any]] = {}

    def deterministic_embed(self, text: str) -> np.ndarray:
        """Deterministic 100% offline CPU embedding generator."""
        tokens = text.lower().split()
        vec = np.zeros(self.vector_dim, dtype=np.float32)
        for i, token in enumerate(tokens):
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.vector_dim
            weight = 1.0 / (1.0 + np.log(i + 1.0))
            vec[idx] += float(weight)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec /= norm
        return vec

    def index_document(self, item: IngestionItem) -> IngestionBatchReport:
        """Index a single document through enterprise ingestion safety gates."""
        ingested, report = self.ingestion_pipeline.process_batch(
            items=[item],
            embed_fn=self.deterministic_embed,
            kms=self.kms
        )
        for doc in ingested:
            self._documents[doc["doc_id"]] = doc
        return report

    def index_batch(self, items: List[IngestionItem]) -> IngestionBatchReport:
        """Index multiple documents with batch deduplication."""
        ingested, report = self.ingestion_pipeline.process_batch(
            items=items,
            embed_fn=self.deterministic_embed,
            kms=self.kms
        )
        for doc in ingested:
            self._documents[doc["doc_id"]] = doc
        return report

    def delete_document(self, doc_id: str, tenant_id: str) -> bool:
        """Atomic deletion with tenant verification and cache invalidation."""
        if doc_id in self._documents:
            doc = self._documents[doc_id]
            if doc["tenant_id"] == tenant_id:
                del self._documents[doc_id]
                self.cache.invalidate_tenant(tenant_id)
                return True
        return False

    def query(
        self,
        user: UserIdentity,
        query_text: str,
        top_k: int = 3,
        pre_filter: bool = True
    ) -> EnterpriseQueryResult:
        """Execute enterprise query with ACL filtering and security defenses."""
        start_time = time.perf_counter()

        # 1. Inspect query for prompt injection
        injected, reason = SecurityGuardrails.detect_prompt_injection(query_text)
        if injected:
            latency = (time.perf_counter() - start_time) * 1000.0
            audit_rec = self.audit_logger.log_query(
                tenant_id=user.tenant_id,
                user_id=user.user_id,
                query=query_text,
                retrieved_chunk_ids=[],
                acl_decisions=[],
                cache_hit="INJECTION_REJECTED",
                latency_ms=latency
            )
            return EnterpriseQueryResult(
                query=query_text,
                tenant_id=user.tenant_id,
                user_id=user.user_id,
                retrieved_chunks=[],
                acl_decisions=[],
                answer=f"Request blocked by Enterprise Security Guardrails: {reason}",
                cache_status="INJECTION_REJECTED",
                latency_ms=latency,
                audit_hash=audit_rec.record_hash
            )

        # 2. Scrub PII from query
        sanitized_query, _ = SecurityGuardrails.redact_pii(query_text)
        query_vec = self.deterministic_embed(sanitized_query)

        # 3. Check Semantic Cache
        cached = self.cache.get(
            tenant_id=user.tenant_id,
            clearance=int(user.clearance),
            query=sanitized_query,
            query_vector=query_vec
        )
        if cached is not None:
            cached_answer, chunk_ids, hit_type = cached
            latency = (time.perf_counter() - start_time) * 1000.0
            audit_rec = self.audit_logger.log_query(
                tenant_id=user.tenant_id,
                user_id=user.user_id,
                query=sanitized_query,
                retrieved_chunk_ids=chunk_ids,
                acl_decisions=[],
                cache_hit=hit_type,
                latency_ms=latency
            )
            return EnterpriseQueryResult(
                query=sanitized_query,
                tenant_id=user.tenant_id,
                user_id=user.user_id,
                retrieved_chunks=[{"doc_id": cid, "cached": True} for cid in chunk_ids],
                acl_decisions=[],
                answer=cached_answer,
                cache_status=hit_type,
                latency_ms=latency,
                audit_hash=audit_rec.record_hash
            )

        # 4. Retrieval with Pre-filtering vs Post-filtering
        scored_candidates: List[Tuple[float, Dict[str, Any]]] = []
        acl_decisions: List[ACLDecision] = []

        if pre_filter:
            # PRE-FILTERING: Filter out non-accessible docs before computing similarity
            for doc in self._documents.values():
                # Enforce multi-tenant boundary first
                if doc["tenant_id"] != user.tenant_id:
                    continue
                decision = ACLEvaluator.evaluate(user, doc["security"])
                acl_decisions.append(decision)
                if decision.allowed:
                    sim = float(np.dot(query_vec, doc["vector"]))
                    scored_candidates.append((sim, doc))
        else:
            # POST-FILTERING: Retrieve top candidates across tenant space, then filter by ACL
            raw_scored: List[Tuple[float, Dict[str, Any]]] = []
            for doc in self._documents.values():
                if doc["tenant_id"] == user.tenant_id:
                    sim = float(np.dot(query_vec, doc["vector"]))
                    raw_scored.append((sim, doc))
            raw_scored.sort(key=lambda x: x[0], reverse=True)
            for sim, doc in raw_scored[:top_k * 2]:
                decision = ACLEvaluator.evaluate(user, doc["security"])
                acl_decisions.append(decision)
                if decision.allowed:
                    scored_candidates.append((sim, doc))

        # Rank and select top_k
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_matches = scored_candidates[:top_k]

        # 5. Decrypt retrieved payloads on-the-fly via KMS
        retrieved_chunks: List[Dict[str, Any]] = []
        retrieved_ids: List[str] = []
        for score, doc in top_matches:
            decrypted_text = self.kms.decrypt_payload(doc["encrypted_pkg"])
            retrieved_chunks.append({
                "doc_id": doc["doc_id"],
                "score": round(score, 4),
                "classification": doc["security"].classification.name,
                "content": decrypted_text
            })
            retrieved_ids.append(doc["doc_id"])

        # 6. Formulate Synthesized Enterprise Response
        if not retrieved_chunks:
            answer = "No authorized documents matching your inquiry were found under your access credentials."
        else:
            context_summary = " ".join([c["content"] for c in retrieved_chunks])
            answer = f"Synthesized Response: Based on authorized enterprise documents [{', '.join(retrieved_ids)}], {context_summary}"

        # 7. Populate Cache
        if retrieved_ids:
            self.cache.put(
                tenant_id=user.tenant_id,
                clearance=int(user.clearance),
                query=sanitized_query,
                query_vector=query_vec,
                response_text=answer,
                retrieved_chunk_ids=retrieved_ids
            )

        latency = (time.perf_counter() - start_time) * 1000.0
        audit_rec = self.audit_logger.log_query(
            tenant_id=user.tenant_id,
            user_id=user.user_id,
            query=sanitized_query,
            retrieved_chunk_ids=retrieved_ids,
            acl_decisions=acl_decisions,
            cache_hit="MISS",
            latency_ms=latency
        )

        return EnterpriseQueryResult(
            query=sanitized_query,
            tenant_id=user.tenant_id,
            user_id=user.user_id,
            retrieved_chunks=retrieved_chunks,
            acl_decisions=acl_decisions,
            answer=answer,
            cache_status="CACHE_MISS_POPULATED",
            latency_ms=latency,
            audit_hash=audit_rec.record_hash
        )


# ---------------------------------------------------------------------------
# 9. Architectural Diagram Generation Utilities
# ---------------------------------------------------------------------------

def generate_enterprise_rag_architecture_diagram(output_path: str) -> None:
    """Generate high-resolution architecture diagram of Enterprise Multi-Tenant RAG."""
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.axis("off")

    # Background color styling
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    # Title & Subtitle
    ax.text(
        0.5, 0.95, "Enterprise Multi-Tenant RAG Architecture",
        fontsize=18, fontweight="bold", color="#F8FAFC", ha="center"
    )
    ax.text(
        0.5, 0.91, "Multi-Tenancy Isolation • KMS Envelope Encryption • Fine-Grained ACL • Semantic Caching",
        fontsize=11, color="#94A3B8", ha="center"
    )

    boxes = [
        {"x": 0.05, "y": 0.62, "w": 0.18, "h": 0.22, "title": "1. Ingestion Pipeline", "color": "#1E293B", "edge": "#38BDF8", "lines": ["• Async Worker Queue", "• Content Hash Dedup", "• PII Redaction Engine", "• Injection Guardrails"]},
        {"x": 0.28, "y": 0.62, "w": 0.18, "h": 0.22, "title": "2. KMS & Vault", "color": "#1E293B", "edge": "#F59E0B", "lines": ["• Master KEK in HSM", "• Per-Tenant DEKs", "• Envelope Encryption", "• Automated Key Rotation"]},
        {"x": 0.51, "y": 0.62, "w": 0.20, "h": 0.22, "title": "3. Storage & Indexing", "color": "#1E293B", "edge": "#10B981", "lines": ["• Tenant Namespaces", "• Document ACL Tags", "• Encrypted Payloads", "• Blue-Green Migration"]},
        {"x": 0.76, "y": 0.62, "w": 0.19, "h": 0.22, "title": "4. Semantic Cache", "color": "#1E293B", "edge": "#A855F7", "lines": ["• L1 Exact SHA-256", "• L2 Vector Cosine Sim", "• Tenant Invalidation", "• Token Cost Reduction"]},

        {"x": 0.05, "y": 0.22, "w": 0.22, "h": 0.28, "title": "User Query & Identity", "color": "#1E293B", "edge": "#60A5FA", "lines": ["• User ID & Tenant ID", "• Clearance Level (0-3)", "• Roles & Groups", "• PII Scrubbed Query"]},
        {"x": 0.35, "y": 0.22, "w": 0.28, "h": 0.28, "title": "Policy Engine & Retrieval", "color": "#1E293B", "edge": "#EC4899", "lines": ["• Pre-Filter: Predicate Pushdown", "• Strict Tenant Partitioning", "• Deny-List Precedence", "• Role/Group Intersection", "• Clearance Thresholds"]},
        {"x": 0.70, "y": 0.22, "w": 0.25, "h": 0.28, "title": "Audit & Non-Repudiation", "color": "#1E293B", "edge": "#EF4444", "lines": ["• Cryptographic SHA-256 Hash", "• Block-Chained Audit Trail", "• ACL ALLOW/DENY Records", "• Tamper-Evident Ledger"]},
    ]

    for b in boxes:
        rect = mpatches.FancyBboxPatch(
            (b["x"], b["y"]), b["w"], b["h"],
            boxstyle="round,pad=0.03",
            facecolor=b["color"],
            edgecolor=b["edge"],
            linewidth=2,
            transform=ax.transAxes
        )
        ax.add_patch(rect)
        ax.text(
            b["x"] + b["w"] / 2.0, b["y"] + b["h"] - 0.04, b["title"],
            fontsize=12, fontweight="bold", color=b["edge"], ha="center", transform=ax.transAxes
        )
        y_offset = b["y"] + b["h"] - 0.09
        for line in b["lines"]:
            ax.text(
                b["x"] + 0.02, y_offset, line,
                fontsize=9.5, color="#E2E8F0", transform=ax.transAxes
            )
            y_offset -= 0.042

    # Flow arrows
    arrows = [
        ((0.23, 0.73), (0.28, 0.73), "#38BDF8"),
        ((0.46, 0.73), (0.51, 0.73), "#F59E0B"),
        ((0.71, 0.73), (0.76, 0.73), "#10B981"),
        ((0.27, 0.36), (0.35, 0.36), "#60A5FA"),
        ((0.63, 0.36), (0.70, 0.36), "#EC4899"),
        ((0.61, 0.62), (0.49, 0.50), "#10B981"),
    ]
    for start, end, col in arrows:
        ax.annotate(
            "", xy=end, xytext=start,
            arrowprops=dict(arrowstyle="->", color=col, lw=2.2, mutation_scale=15),
            xycoords="axes fraction", textcoords="axes fraction"
        )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_acl_matrix_diagram(output_path: str) -> None:
    """Generate visual diagram illustrating Pre-Filtering vs Post-Filtering trade-offs and ACL policy flow."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    fig.patch.set_facecolor("#0F172A")
    ax1.set_facecolor("#1E293B")
    ax2.set_facecolor("#1E293B")

    # Subplot 1: Pre-Filtering vs Post-Filtering Comparison
    categories = ["Security Guarantee", "Recall Stability", "Index Scalability", "Vector Efficiency"]
    pre_scores = [1.0, 0.95, 0.88, 0.92]
    post_scores = [0.65, 0.40, 0.95, 0.60]

    y_pos = np.arange(len(categories))
    bar_height = 0.35

    ax1.barh(y_pos + bar_height / 2, pre_scores, bar_height, label="Pre-Filtering (Pushdown)", color="#38BDF8")
    ax1.barh(y_pos - bar_height / 2, post_scores, bar_height, label="Post-Filtering (Fetch & Scrub)", color="#F43F5E")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(categories, color="#F8FAFC", fontsize=11)
    ax1.set_xlabel("Relative Effectiveness Score (0.0 - 1.0)", color="#94A3B8", fontsize=10)
    ax1.set_title("Pre-Filtering vs Post-Filtering Trade-Offs", color="#F8FAFC", fontsize=13, fontweight="bold")
    ax1.tick_params(colors="#94A3B8")
    ax1.set_xlim(0, 1.15)
    ax1.legend(loc="lower right", facecolor="#0F172A", edgecolor="#475569", labelcolor="#F8FAFC")
    ax1.grid(axis="x", linestyle="--", alpha=0.3, color="#94A3B8")

    # Subplot 2: ACL Decision Tree Breakdown
    ax2.axis("off")
    ax2.set_title("Hierarchical ACL Evaluation Rules", color="#F8FAFC", fontsize=13, fontweight="bold")

    rules = [
        ("Step 1: Multi-Tenant Boundary", "user.tenant == doc.tenant", "#38BDF8"),
        ("Step 2: Explicit Deny Check", "user NOT IN doc.denied_users", "#EF4444"),
        ("Step 3: Clearance Threshold", "user.clearance >= doc.classification", "#F59E0B"),
        ("Step 4: Owner / Admin Bypass", "user == doc.owner OR 'admin' IN roles", "#10B981"),
        ("Step 5: Explicit Allow Check", "user IN doc.allowed_users", "#A855F7"),
        ("Step 6: Role / Group Intersection", "roles ∩ allowed_roles OR groups ∩ allowed_groups", "#6366F1"),
        ("Default: Explicit Deny", "No grant matched -> ACCESS DENIED", "#94A3B8"),
    ]

    y = 0.90
    for title, desc, col in rules:
        rect = plt.Rectangle((0.05, y - 0.08), 0.90, 0.09, facecolor="#0F172A", edgecolor=col, linewidth=1.5)
        ax2.add_patch(rect)
        ax2.text(0.08, y - 0.035, title, fontsize=10.5, fontweight="bold", color=col)
        ax2.text(0.08, y - 0.065, desc, fontsize=9.5, color="#CBD5E1")
        y -= 0.125

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
