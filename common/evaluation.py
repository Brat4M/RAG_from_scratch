"""Automated RAG Evaluation Framework (RAGAS & TruLens Architecture).

Provides 100% offline-executable CPU evaluation metrics, proposition extraction,
citation validation, grounded synthesis, and optional Groq LLM-as-a-Judge integration:
1. Context Precision (MAP@K over retrieval relevance).
2. Context Recall (ground-truth claim coverage).
3. Faithfulness / Groundedness (hallucination detection via atomic proposition verification).
4. Answer Relevance (query-response alignment and semantic fidelity).
5. RAG Triad composite scoring (TruLens harmonic triad).
6. Grounded RAG Synthesizer with epistemic refusal and citation validation.
7. Diagnostic reporting and radar visualization helpers.
"""

from __future__ import annotations

import json
import math
import os
import re
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# 1. Core Data Structures & Models
# ---------------------------------------------------------------------------

@dataclass
class EvaluationSample:
    """Represents a single query-retrieval-generation evaluation instance."""
    query: str
    contexts: List[str]
    ground_truth: str
    generated_answer: str
    sample_id: str = "sample_01"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaimVerification:
    """Detailed verification audit for an individual factual claim."""
    claim: str
    is_supported: bool
    confidence: float
    supporting_context_indices: List[int] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "is_supported": self.is_supported,
            "confidence": round(float(self.confidence), 4),
            "supporting_context_indices": self.supporting_context_indices,
            "reason": self.reason,
        }


@dataclass
class EvaluationResult:
    """Comprehensive evaluation score card for a RAG test case."""
    sample_id: str
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevance: float
    rag_triad_score: float
    claim_verifications: List[ClaimVerification] = field(default_factory=list)
    retrieval_relevance: List[bool] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "context_precision": round(float(self.context_precision), 4),
            "context_recall": round(float(self.context_recall), 4),
            "faithfulness": round(float(self.faithfulness), 4),
            "answer_relevance": round(float(self.answer_relevance), 4),
            "rag_triad_score": round(float(self.rag_triad_score), 4),
            "retrieval_relevance": self.retrieval_relevance,
            "claim_verifications": [c.to_dict() for c in self.claim_verifications],
            "diagnostics": self.diagnostics,
        }


# ---------------------------------------------------------------------------
# 2. Deterministic Subword & Semantic Embedder (100% Offline CPU)
# ---------------------------------------------------------------------------

class DeterministicSemanticEmbedder:
    """Lightweight deterministic semantic vector embedder using subword hashing.
    
    Generates consistent, deterministic dense vectors without requiring heavyweight
    PyTorch or HuggingFace neural dependencies. Uses multi-scale character n-gram
    hashing with subword frequency weighting and L2 normalization.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        clean = text.lower().strip()
        tokens = re.findall(r"\w+", clean)
        if not tokens:
            vec = np.zeros(self.dim, dtype=np.float32)
            vec[0] = 1.0
            return vec

        vec = np.zeros(self.dim, dtype=np.float32)

        # Word-level hashed contributions
        for token in tokens:
            h = zlib.crc32(token.encode("utf-8"))
            idx = h % self.dim
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            vec[idx] += sign * 1.5

            # Subword character n-grams (3-to-5 grams)
            token_wrapped = f"^{token}$"
            for n in (3, 4, 5):
                for i in range(len(token_wrapped) - n + 1):
                    sub = token_wrapped[i : i + n]
                    sh = zlib.crc32(sub.encode("utf-8"))
                    s_idx = sh % self.dim
                    s_sign = 1.0 if ((sh >> 16) & 1) else -1.0
                    vec[s_idx] += s_sign * 0.5

        norm = float(np.linalg.norm(vec))
        if norm > 1e-9:
            vec = vec / norm
        else:
            vec[0] = 1.0
        return vec

    def similarity(self, text1: str, text2: str) -> float:
        v1 = self.encode(text1)
        v2 = self.encode(text2)
        dot = float(np.dot(v1, v2))
        return max(0.0, min(1.0, dot))


# ---------------------------------------------------------------------------
# 3. Atomic Proposition & Claim Decomposition
# ---------------------------------------------------------------------------

class PropositionExtractor:
    """Decomposes compound sentences and answers into atomic factual claims.
    
    In RAG evaluation (RAGAS / FactScore / TruLens), hallucination detection requires
    evaluating fine-grained atomic claims rather than entire multi-sentence paragraphs.
    """

    REFUSAL_PATTERN = re.compile(
        r"cannot answer|insufficient (?:information|context|evidence)|not provided|not mentioned",
        flags=re.IGNORECASE,
    )

    @classmethod
    def extract_claims(cls, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        # If text is an epistemic refusal, no factual claims are asserted
        if cls.REFUSAL_PATTERN.search(text):
            return []

        # Remove citation annotations like [Doc 1], [1], [doc_2] before claim splitting
        clean_text = re.sub(r"\[(?:Doc\s*\d+|\d+|chunk_\w+)\]", "", text, flags=re.IGNORECASE)

        # Split into preliminary sentences
        raw_sentences = re.split(r"(?<=[.!?])\s+", clean_text.strip())
        claims: List[str] = []

        # Coordinating conjunctions and clause boundary patterns for proposition splitting
        clause_split_pattern = re.compile(
            r",\s*(?:and|but|while|whereas|however)\s+|\s+(?:and also|and furthermore|additionally|moreover)\s+",
            flags=re.IGNORECASE,
        )

        for sent in raw_sentences:
            s = sent.strip()
            # Clean list markers, bullet points, leading dashes
            s = re.sub(r"^[\s*\-•\d.]+", "", s).strip()
            if not s or cls.REFUSAL_PATTERN.search(s):
                continue

            sub_clauses = clause_split_pattern.split(s)
            for clause in sub_clauses:
                c = clause.strip().rstrip(".,;:")
                words = c.split()
                if len(words) >= 3:
                    claims.append(c)
                elif len(words) >= 1 and len(claims) == 0:
                    claims.append(c)

        return claims if claims else ([clean_text.strip()] if clean_text.strip() else [])


# ---------------------------------------------------------------------------
# 4. Factual Entailment & Claim Verification Engine
# ---------------------------------------------------------------------------

class OfflineClaimVerifier:
    """Evaluates whether an atomic claim is entailed / supported by context chunks."""

    def __init__(self, embedder: Optional[DeterministicSemanticEmbedder] = None, threshold: float = 0.45):
        self.embedder = embedder or DeterministicSemanticEmbedder()
        self.threshold = threshold

    def verify_claim(self, claim: str, contexts: List[str]) -> ClaimVerification:
        if not contexts:
            return ClaimVerification(
                claim=claim,
                is_supported=False,
                confidence=0.0,
                supporting_context_indices=[],
                reason="No context passages provided for verification.",
            )

        claim_words = set(re.findall(r"\w+", claim.lower()))
        # Filter stop words to focus on content words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "and", "or", "to", "in",
            "of", "that", "this", "it", "for", "with", "as", "by", "on", "at", "from"
        }
        content_words = {w for w in claim_words if w not in stop_words and len(w) > 2}

        # Check for explicit negation in claim
        has_claim_negation = bool(re.search(r"\b(?:not|no|never|neither|cannot|does not|without)\b", claim.lower()))

        best_score = 0.0
        best_indices: List[int] = []
        best_reason = ""

        for idx, ctx in enumerate(contexts):
            sim = self.embedder.similarity(claim, ctx)
            ctx_words = set(re.findall(r"\w+", ctx.lower()))
            overlap = len(content_words & ctx_words) / max(1, len(content_words))

            has_ctx_negation = bool(re.search(r"\b(?:not|no|never|neither|cannot|does not|without)\b", ctx.lower()))
            # Contradiction penalty if negation polarity mismatches
            polarity_penalty = 0.4 if (has_claim_negation != has_ctx_negation and overlap > 0.4) else 0.0

            # Combined entailment confidence
            combined_conf = max(0.0, (0.55 * sim + 0.45 * overlap) - polarity_penalty)

            if combined_conf > best_score:
                best_score = combined_conf

            if combined_conf >= self.threshold:
                best_indices.append(idx)

        is_supported = len(best_indices) > 0 and best_score >= self.threshold
        reason = (
            f"Supported by chunk(s) {best_indices} with confidence {best_score:.3f}"
            if is_supported
            else f"Unverifiable: max confidence {best_score:.3f} below threshold {self.threshold}"
        )

        return ClaimVerification(
            claim=claim,
            is_supported=is_supported,
            confidence=best_score,
            supporting_context_indices=best_indices,
            reason=reason,
        )


# ---------------------------------------------------------------------------
# 5. Automated Evaluation Metrics (RAGAS & TruLens Formulations)
# ---------------------------------------------------------------------------

def compute_context_precision(
    query: str,
    contexts: List[str],
    ground_truth: str,
    embedder: Optional[DeterministicSemanticEmbedder] = None,
    threshold: float = 0.55,
) -> Tuple[float, List[bool]]:
    """Calculates Context Precision (Signal-to-Noise Ratio and Rank-weighted Relevance).
    
    Context Precision measures whether retrieved chunks are relevant to answering
    the query, and heavily penalizes irrelevant chunks ranked above relevant ones.
    
    Formula:
        Context Precision@K = (1 / |Relevant Chunks|) * sum_{k=1}^K (Precision@k * v_k)
        where v_k in {0, 1} is relevance indicator of chunk k.
    """
    if not contexts:
        return 0.0, []

    emb = embedder or DeterministicSemanticEmbedder()
    relevance_flags: List[bool] = []

    # A chunk is relevant if it exhibits strong semantic alignment with the query or ground truth
    for ctx in contexts:
        q_sim = emb.similarity(query, ctx)
        gt_sim = emb.similarity(ground_truth, ctx) if ground_truth else 0.0
        # Keyword content overlap with query or ground truth
        q_tokens = {w for w in re.findall(r"\w+", query.lower()) if len(w) > 3}
        gt_tokens = {w for w in re.findall(r"\w+", ground_truth.lower()) if len(w) > 3}
        ctx_tokens = set(re.findall(r"\w+", ctx.lower()))

        q_overlap = len(q_tokens & ctx_tokens) / max(1, len(q_tokens))
        gt_overlap = len(gt_tokens & ctx_tokens) / max(1, len(gt_tokens)) if gt_tokens else 0.0

        is_rel = (q_sim >= threshold or gt_sim >= threshold or q_overlap >= 0.40 or gt_overlap >= 0.40)
        relevance_flags.append(is_rel)

    total_relevant = sum(1 for r in relevance_flags if r)
    if total_relevant == 0:
        return 0.0, relevance_flags

    # Calculate cumulative Mean Average Precision (MAP)
    running_relevant = 0
    precision_sum = 0.0

    for k, is_rel in enumerate(relevance_flags, start=1):
        if is_rel:
            running_relevant += 1
            precision_at_k = running_relevant / k
            precision_sum += precision_at_k

    context_precision = precision_sum / total_relevant
    return float(context_precision), relevance_flags


def compute_context_recall(
    ground_truth: str,
    contexts: List[str],
    claim_extractor: Optional[PropositionExtractor] = None,
    verifier: Optional[OfflineClaimVerifier] = None,
) -> Tuple[float, List[ClaimVerification]]:
    """Calculates Context Recall (Ground-Truth Completeness).
    
    Measures the fraction of atomic claims in the ground-truth answer that can
    be directly attributed to the retrieved context passages.
    
    Formula:
        Context Recall = |Attributed Ground Truth Claims| / |Total Ground Truth Claims|
    """
    if not ground_truth or not ground_truth.strip():
        return 1.0, []

    extractor = claim_extractor or PropositionExtractor()
    vf = verifier or OfflineClaimVerifier()

    gt_claims = extractor.extract_claims(ground_truth)
    if not gt_claims:
        return 1.0, []

    verifications: List[ClaimVerification] = []
    supported_count = 0

    for claim in gt_claims:
        v = vf.verify_claim(claim, contexts)
        verifications.append(v)
        if v.is_supported:
            supported_count += 1

    recall = supported_count / len(gt_claims)
    return float(recall), verifications


def compute_faithfulness(
    generated_answer: str,
    contexts: List[str],
    claim_extractor: Optional[PropositionExtractor] = None,
    verifier: Optional[OfflineClaimVerifier] = None,
) -> Tuple[float, List[ClaimVerification]]:
    """Calculates Faithfulness / Groundedness (Hallucination Detection).
    
    Measures the proportion of atomic claims in the generated response that can be
    inferred from and validated against the retrieved context passages.
    
    Formula:
        Faithfulness = |Supported Answer Claims| / |Total Answer Claims|
    """
    if not generated_answer or not generated_answer.strip():
        return 1.0, []

    # If the answer is an epistemic refusal ("I do not have sufficient information"),
    # it is 100% faithful to the context (did not hallucinate facts).
    refusal_pattern = re.compile(
        r"cannot answer|insufficient (?:information|context|evidence)|not provided in the context|not mentioned",
        flags=re.IGNORECASE,
    )
    if refusal_pattern.search(generated_answer):
        return 1.0, [
            ClaimVerification(
                claim=generated_answer,
                is_supported=True,
                confidence=1.0,
                supporting_context_indices=[],
                reason="Epistemic refusal: model correctly acknowledged absent evidence.",
            )
        ]

    extractor = claim_extractor or PropositionExtractor()
    vf = verifier or OfflineClaimVerifier()

    answer_claims = extractor.extract_claims(generated_answer)
    if not answer_claims:
        return 1.0, []

    verifications: List[ClaimVerification] = []
    supported_count = 0

    for claim in answer_claims:
        v = vf.verify_claim(claim, contexts)
        verifications.append(v)
        if v.is_supported:
            supported_count += 1

    faithfulness = supported_count / len(answer_claims)
    return float(faithfulness), verifications


def compute_answer_relevance(
    query: str,
    generated_answer: str,
    embedder: Optional[DeterministicSemanticEmbedder] = None,
) -> float:
    """Calculates Answer Relevance (Semantic Query-Response Alignment).
    
    Evaluates whether the generated response directly answers the user's inquiry,
    penalizing evasive answers, off-topic hallucinations, and generic boilerplate.
    """
    if not generated_answer or not generated_answer.strip():
        return 0.0

    emb = embedder or DeterministicSemanticEmbedder()

    # Epistemic refusal to an unanswerable query has baseline relevance of 0.70
    # (it is a valid pragmatic response acknowledging limitations)
    refusal_pattern = re.compile(
        r"cannot answer|insufficient (?:information|context|evidence)|not provided",
        flags=re.IGNORECASE,
    )
    if refusal_pattern.search(generated_answer):
        return 0.70

    sim = emb.similarity(query, generated_answer)

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "and", "or", "to", "in",
        "of", "that", "this", "it", "for", "with", "as", "by", "on", "at", "from",
        "what", "when", "where", "which", "who", "why", "how", "does", "do", "did"
    }
    q_tokens = {w for w in re.findall(r"\w+", query.lower()) if w not in stop_words and len(w) >= 3}
    ans_tokens = set(re.findall(r"\w+", generated_answer.lower()))

    overlap = len(q_tokens & ans_tokens) / max(1, len(q_tokens))
    if overlap == 0.0:
        return 0.0
    
    # Calibrate scores for unit-normalized hashing vector distributions
    calibrated_sim = min(1.0, sim / 0.25)
    calibrated_overlap = min(1.0, overlap / 0.40)
    
    score = 0.50 * calibrated_sim + 0.50 * calibrated_overlap
    return float(max(0.0, min(1.0, score)))


def compute_rag_triad_score(
    context_relevance: float,
    faithfulness: float,
    answer_relevance: float,
    eps: float = 1e-4,
) -> float:
    """Calculates the composite RAG Triad Harmonic Score (TruLens Paradigm).
    
    Harmonic mean prevents high scores on one dimension from masking critical
    failures in another (e.g. a high answer relevance cannot compensate for a hallucination).
    """
    cr = max(eps, min(1.0, context_relevance))
    ft = max(eps, min(1.0, faithfulness))
    ar = max(eps, min(1.0, answer_relevance))

    harmonic_mean = 3.0 / ((1.0 / cr) + (1.0 / ft) + (1.0 / ar))
    return float(harmonic_mean)


# ---------------------------------------------------------------------------
# 6. Production Grounded Synthesizer with Citations & Refusals
# ---------------------------------------------------------------------------

class RAGSynthesizer:
    """Production prompt assembler, citation extractor, and grounded generator.
    
    Supports 100% offline extractive/extrapolative generation and optional
    live Groq API synthesis.
    """

    SYSTEM_PROMPT = (
        "You are an authoritative, evidence-grounded AI assistant.\n"
        "Your task is to answer the user's question STRICTLY and ONLY based on the provided context passages.\n"
        "Rules:\n"
        "1. Every factual statement must include an inline bracket citation referencing the source chunk, e.g. [Doc 1] or [Doc 2].\n"
        "2. Do not introduce outside knowledge or extrapolate beyond the text.\n"
        "3. If the context does not contain sufficient evidence to answer the question, state: "
        "'Based on the provided context, I cannot answer this question.' Do not speculate."
    )

    def __init__(self, groq_api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model

    def build_prompt(self, query: str, contexts: List[str]) -> str:
        """Assembles context passages with explicit document demarcation."""
        context_blocks = []
        for i, ctx in enumerate(contexts, start=1):
            context_blocks.append(f"[Doc {i}] (Source ID: chunk_{i:02d})\n{ctx.strip()}")

        joined_context = "\n\n".join(context_blocks)
        prompt = (
            f"=== CONTEXT PASSAGES ===\n"
            f"{joined_context}\n"
            f"========================\n\n"
            f"User Question: {query}\n\n"
            f"Grounded Answer (with inline [Doc X] citations):"
        )
        return prompt

    def synthesize_offline(self, query: str, contexts: List[str]) -> str:
        """Deterministic 100% offline grounded synthesis simulator."""
        if not contexts:
            return "Based on the provided context, I cannot answer this question."

        emb = DeterministicSemanticEmbedder()
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "and", "or", "to", "in",
            "of", "that", "this", "it", "for", "with", "as", "by", "on", "at", "from",
            "what", "when", "where", "which", "who", "why", "how", "does", "do", "did"
        }
        q_tokens = {w for w in re.findall(r"\w+", query.lower()) if w not in stop_words and len(w) >= 3}

        # Check if any chunk has relevance to the query
        scored_chunks: List[Tuple[int, str, float]] = []
        for idx, ctx in enumerate(contexts, start=1):
            sim = emb.similarity(query, ctx)
            ctx_tokens = set(re.findall(r"\w+", ctx.lower()))
            overlap = len(q_tokens & ctx_tokens) / max(1, len(q_tokens))
            score = 0.5 * sim + 0.5 * overlap
            scored_chunks.append((idx, ctx, score))

        scored_chunks.sort(key=lambda x: x[2], reverse=True)
        top_idx, top_ctx, top_score = scored_chunks[0]

        # Epistemic refusal if context relevance is too low (unrelated topic)
        if top_score < 0.20:
            return "Based on the provided context, I cannot answer this question as the necessary information is not mentioned in the passages."

        # Extract the most informative sentence from the best chunk
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", top_ctx) if s.strip()]
        best_sentence = sentences[0]
        best_sent_score = 0.0
        for s in sentences:
            s_score = emb.similarity(query, s)
            if s_score > best_sent_score:
                best_sent_score = s_score
                best_sentence = s

        answer = f"{best_sentence} [Doc {top_idx}]."
        return answer

    def synthesize(self, query: str, contexts: List[str]) -> str:
        """Executes synthesis via live Groq API if key is present, else falls back offline."""
        if not self.groq_api_key or self.groq_api_key.startswith("mock_"):
            return self.synthesize_offline(query, contexts)

        try:
            from groq import Groq
            client = Groq(api_key=self.groq_api_key)
            prompt = self.build_prompt(query, contexts)

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return self.synthesize_offline(query, contexts)

    @staticmethod
    def extract_citations(answer: str) -> List[int]:
        """Extracts cited document numbers from generated answer, e.g. [Doc 1] -> [1]."""
        matches = re.findall(r"\[(?:Doc\s*|chunk_)(\d+)\]", answer, flags=re.IGNORECASE)
        citations = sorted(list({int(m) for m in matches}))
        return citations

    @staticmethod
    def validate_citations(answer: str, num_contexts: int) -> Dict[str, Any]:
        """Validates that all citations are within the bounds of retrieved contexts."""
        citations = RAGSynthesizer.extract_citations(answer)
        invalid = [c for c in citations if c < 1 or c > num_contexts]
        return {
            "citations_found": citations,
            "has_citations": len(citations) > 0,
            "all_citations_valid": len(invalid) == 0,
            "invalid_citations": invalid,
        }


# ---------------------------------------------------------------------------
# 7. Comprehensive Evaluation Harness & Diagnostics
# ---------------------------------------------------------------------------

class RAGEvaluationHarness:
    """Unified evaluation test harness orchestrating RAGAS and TruLens metrics."""

    def __init__(self, embedder: Optional[DeterministicSemanticEmbedder] = None):
        self.embedder = embedder or DeterministicSemanticEmbedder()
        self.claim_extractor = PropositionExtractor()
        self.verifier = OfflineClaimVerifier(embedder=self.embedder)

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        """Evaluates a single sample across Context Precision, Recall, Faithfulness, Relevance."""
        # 1. Context Precision
        ctx_precision, rel_flags = compute_context_precision(
            query=sample.query,
            contexts=sample.contexts,
            ground_truth=sample.ground_truth,
            embedder=self.embedder,
        )

        # 2. Context Recall
        ctx_recall, gt_verifications = compute_context_recall(
            ground_truth=sample.ground_truth,
            contexts=sample.contexts,
            claim_extractor=self.claim_extractor,
            verifier=self.verifier,
        )

        # 3. Faithfulness
        faithfulness, ans_verifications = compute_faithfulness(
            generated_answer=sample.generated_answer,
            contexts=sample.contexts,
            claim_extractor=self.claim_extractor,
            verifier=self.verifier,
        )

        # 4. Answer Relevance
        ans_relevance = compute_answer_relevance(
            query=sample.query,
            generated_answer=sample.generated_answer,
            embedder=self.embedder,
        )

        # 5. Composite RAG Triad Score
        triad_score = compute_rag_triad_score(
            context_relevance=ctx_precision,
            faithfulness=faithfulness,
            answer_relevance=ans_relevance,
        )

        # Diagnostic categorization
        diagnostics = self._generate_diagnostics(
            ctx_precision, ctx_recall, faithfulness, ans_relevance, sample.contexts, sample.generated_answer
        )

        return EvaluationResult(
            sample_id=sample.sample_id,
            context_precision=ctx_precision,
            context_recall=ctx_recall,
            faithfulness=faithfulness,
            answer_relevance=ans_relevance,
            rag_triad_score=triad_score,
            claim_verifications=ans_verifications,
            retrieval_relevance=rel_flags,
            diagnostics=diagnostics,
        )

    def evaluate_dataset(self, samples: List[EvaluationSample]) -> Dict[str, Any]:
        """Runs evaluation over a collection of samples and computes mean metrics."""
        results = [self.evaluate_sample(s) for s in samples]

        mean_precision = float(np.mean([r.context_precision for r in results]))
        mean_recall = float(np.mean([r.context_recall for r in results]))
        mean_faithfulness = float(np.mean([r.faithfulness for r in results]))
        mean_relevance = float(np.mean([r.answer_relevance for r in results]))
        mean_triad = float(np.mean([r.rag_triad_score for r in results]))

        return {
            "total_samples": len(samples),
            "mean_context_precision": round(mean_precision, 4),
            "mean_context_recall": round(mean_recall, 4),
            "mean_faithfulness": round(mean_faithfulness, 4),
            "mean_answer_relevance": round(mean_relevance, 4),
            "mean_rag_triad_score": round(mean_triad, 4),
            "results": [r.to_dict() for r in results],
        }

    def _generate_diagnostics(
        self,
        precision: float,
        recall: float,
        faithfulness: float,
        relevance: float,
        contexts: List[str],
        answer: str,
    ) -> Dict[str, Any]:
        """Pinpoints root causes of RAG pipeline failures."""
        failure_modes = []

        if precision < 0.50:
            failure_modes.append("NOISY_RETRIEVAL: Low context precision; irrelevant chunks pollute context window.")

        if recall < 0.50:
            failure_modes.append("INCOMPLETE_RETRIEVAL: Low context recall; context missed key ground-truth facts.")

        if faithfulness < 0.65:
            failure_modes.append("HALLUCINATION_DETECTED: Answer contains claims unsupported by retrieved context.")

        if relevance < 0.50:
            failure_modes.append("TOPIC_DRIFT: Answer failed to address query intent directly.")

        citation_audit = RAGSynthesizer.validate_citations(answer, len(contexts))

        return {
            "pipeline_status": "PASS" if not failure_modes else "WARNING",
            "failure_modes": failure_modes if failure_modes else ["HEALTHY: All metrics meet acceptance thresholds."],
            "citation_audit": citation_audit,
        }
