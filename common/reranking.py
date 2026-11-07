"""Advanced Reranking and Cross-Encoder Architectures.

Provides educational, 100% offline-executable CPU implementations and optional
Groq LLM-as-a-Reranker integrations for:
1. Bi-Encoder candidate generation simulator (independent representation bottleneck).
2. Cross-Encoder joint attention simulator (token-level interaction & negation resolution).
3. Score calibration (sigmoid scaling with temperature, min-max normalization, z-score).
4. Cascaded two-stage retrieval pipeline (coarse candidate generation + precision reranking).
5. Information retrieval evaluation metrics (Precision@k, Recall@k, MRR, NDCG@k).
"""

from __future__ import annotations

import math
import os
import re
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# 1. Data Structures & Scored Document Model
# ---------------------------------------------------------------------------

@dataclass
class ScoredDocument:
    """Represents a document scored during multi-stage retrieval."""
    doc_id: str
    text: str
    bi_encoder_score: float = 0.0
    cross_encoder_logit: float = 0.0
    calibrated_score: float = 0.0
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "bi_encoder_score": round(float(self.bi_encoder_score), 4),
            "cross_encoder_logit": round(float(self.cross_encoder_logit), 4),
            "calibrated_score": round(float(self.calibrated_score), 4),
            "rank": self.rank,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# 2. Bi-Encoder Baseline (Independent Representation Bottleneck)
# ---------------------------------------------------------------------------

class BiEncoderSimulator:
    """Simulates dual-encoder (Bi-Encoder) embeddings and cosine similarity.
    
    Bi-encoders map query and document independently into fixed-dimensional vectors:
        u = f(q), v = f(d)
    and compute relevance via inner product:
        s(q, d) = <u, v> / (||u|| * ||v||)
        
    Complexity for N documents: O(N * D) cosine operations.
    Bottleneck: Tokens in q cannot attend to tokens in d.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        """Compute an L2-normalized dense embedding via subword hashing."""
        vec = np.zeros(self.dim, dtype=np.float32)
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = [t for t in cleaned.split() if t]

        for token in tokens:
            w_idx = zlib.crc32(token.encode("utf-8")) % self.dim
            vec[w_idx] += 1.0

            if len(token) >= 3:
                for i in range(len(token) - 2):
                    sub = token[i:i+3]
                    s_idx = zlib.crc32(sub.encode("utf-8")) % self.dim
                    vec[s_idx] += 0.35

        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    def score(self, query: str, document: str) -> float:
        """Compute cosine similarity between query and document vectors."""
        u = self.encode(query)
        v = self.encode(document)
        cos = float(np.dot(u, v))
        return max(-1.0, min(1.0, cos))

    def retrieve_candidates(
        self,
        query: str,
        corpus: Dict[str, str],
        top_k: int = 10,
    ) -> List[ScoredDocument]:
        """Rank all documents in corpus using bi-encoder cosine similarity."""
        results: List[ScoredDocument] = []
        u = self.encode(query)

        for doc_id, text in corpus.items():
            v = self.encode(text)
            sim = float(np.dot(u, v))
            results.append(
                ScoredDocument(
                    doc_id=doc_id,
                    text=text,
                    bi_encoder_score=sim,
                )
            )

        # Sort descending by bi-encoder score
        results.sort(key=lambda x: x.bi_encoder_score, reverse=True)
        for rank_idx, doc in enumerate(results[:top_k], start=1):
            doc.rank = rank_idx

        return results[:top_k]


# ---------------------------------------------------------------------------
# 3. Cross-Encoder Simulator (Joint Token-Level Cross-Attention)
# ---------------------------------------------------------------------------

class CrossEncoderSimulator:
    """Simulates a full cross-attention transformer reranker.
    
    Given concatenated sequence [CLS] + q + [SEP] + d + [SEP], every token
    in q attends to every token in d across attention heads and layers.
    
    Captures:
    1. Token-level interaction heatmap M[i, j] = <W_q * t_q[i], W_k * t_d[j]>.
    2. Contextual negation and polarity reversal ('without' vs 'with', 'not supported').
    3. Exact phrase alignment and semantic alignment pooling.
    4. Transformer scoring head yielding uncalibrated logit z in (-inf, +inf).
    """

    def __init__(self, token_dim: int = 32, num_heads: int = 4, seed: int = 42):
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.rng = np.random.default_rng(seed)

        # Token embedding projection dictionary for deterministic token representations
        self._token_cache: Dict[str, np.ndarray] = {}

        # Polarities and negation indicators that invert relevance in cross-attention
        self.negation_words = {"not", "no", "never", "without", "unsupported", "cannot", "neither", "nor"}
        self.affirmative_words = {"with", "using", "requires", "supported", "supports", "enables", "can"}

    def _tokenize(self, text: str) -> List[str]:
        """Simple lowercase word tokenizer preserving basic punctuation tokens."""
        cleaned = re.sub(r"([?.!,;])", r" \1 ", text.lower())
        return [t for t in cleaned.split() if t]

    def _get_token_vec(self, token: str) -> np.ndarray:
        """Deterministic pseudo-embedding for individual token."""
        if token in self._token_cache:
            return self._token_cache[token]

        # Use CRC32 hash to seed deterministic vector
        seed = zlib.crc32(token.encode("utf-8"))
        token_rng = np.random.default_rng(seed)
        vec = token_rng.standard_normal(self.token_dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        self._token_cache[token] = vec
        return vec

    def compute_attention_matrix(
        self, query: str, document: str
    ) -> Tuple[List[str], List[str], np.ndarray]:
        """Compute the (L_q, L_d) cross-attention matrix between query and document tokens.
        
        Returns:
            (query_tokens, doc_tokens, attention_weights)
        """
        q_tokens = self._tokenize(query)
        d_tokens = self._tokenize(document)

        if not q_tokens or not d_tokens:
            return q_tokens, d_tokens, np.zeros((len(q_tokens), len(d_tokens)), dtype=np.float32)

        # Build token representation matrices Q: (L_q, D) and K: (L_d, D)
        Q = np.vstack([self._get_token_vec(t) for t in q_tokens])
        K = np.vstack([self._get_token_vec(t) for t in d_tokens])

        # Scaled dot-product: S = Q @ K.T / sqrt(D)
        scale = math.sqrt(self.token_dim)
        raw_scores = np.dot(Q, K.T) / scale

        # Softmax over document tokens for each query token
        exp_scores = np.exp(raw_scores - np.max(raw_scores, axis=1, keepdims=True))
        attention_matrix = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-9)

        return q_tokens, d_tokens, attention_matrix

    def predict_logit(self, query: str, document: str) -> float:
        """Score (query, document) pair using full cross-attention semantics.
        
        Returns raw logit z in (-inf, +inf).
        """
        q_tokens, d_tokens, attn = self.compute_attention_matrix(query, document)
        if not q_tokens or not d_tokens:
            return -5.0

        q_set = set(q_tokens)
        d_set = set(d_tokens)

        # 1. Base semantic alignment via max-pooled cross-attention
        # For each query token, find the strongest matching document token
        max_token_alignments = np.max(attn, axis=1)
        base_alignment_score = float(np.mean(max_token_alignments))

        # 2. Token overlap & lexical alignment
        overlap = q_set.intersection(d_set)
        jaccard = len(overlap) / max(1, len(q_set.union(d_set)))

        # 3. Negation & Conditionality Interaction (The classic Cross-Encoder advantage)
        # Check if the query contains conditional negation (e.g. "without docker", "not supported")
        # and verify whether the document matches or contradicts that exact condition.
        polarity_penalty = 0.0
        condition_bonus = 0.0

        query_has_negation = bool(q_set.intersection(self.negation_words))
        doc_has_negation = bool(d_set.intersection(self.negation_words))

        # Specific bigram / dependency check: e.g. "without <X>" vs "with <X>"
        for i, q_tok in enumerate(q_tokens[:-1]):
            target = q_tokens[i + 1]
            if q_tok in self.negation_words:
                # Query wants NEGATION of target (e.g., "without docker")
                # Look for target in document
                for j, d_tok in enumerate(d_tokens[:-1]):
                    if d_tokens[j + 1] == target or d_tok == target:
                        prev_d_tok = d_tokens[j] if d_tokens[j + 1] == target else (d_tokens[j-1] if j > 0 else "")
                        if prev_d_tok in self.affirmative_words:
                            # Contradiction: Doc uses "with docker" when query wanted "without docker"
                            polarity_penalty += 2.5
                        elif prev_d_tok in self.negation_words:
                            # Agreement: Doc also specifies "without docker"
                            condition_bonus += 2.5
            elif q_tok in self.affirmative_words:
                # Query wants AFFIRMATIVE target (e.g., "with docker")
                for j, d_tok in enumerate(d_tokens[:-1]):
                    if d_tokens[j + 1] == target or d_tok == target:
                        prev_d_tok = d_tokens[j] if d_tokens[j + 1] == target else (d_tokens[j-1] if j > 0 else "")
                        if prev_d_tok in self.negation_words:
                            # Contradiction: Doc says "without docker" when query wanted "with docker"
                            polarity_penalty += 2.5
                        elif prev_d_tok in self.affirmative_words:
                            condition_bonus += 1.5

        # 4. Synthesize final logit
        # Logit formula: bias + w1*jaccard + w2*base_alignment + bonus - penalty
        raw_logit = -2.0 + (6.0 * jaccard) + (4.0 * base_alignment_score) + condition_bonus - polarity_penalty

        return float(raw_logit)

    def rerank(
        self,
        query: str,
        candidates: List[ScoredDocument],
        top_k: Optional[int] = None,
    ) -> List[ScoredDocument]:
        """Rerank candidate documents using cross-attention scoring."""
        reranked = [
            ScoredDocument(
                doc_id=d.doc_id,
                text=d.text,
                bi_encoder_score=d.bi_encoder_score,
                cross_encoder_logit=self.predict_logit(query, d.text),
                metadata=dict(d.metadata),
            )
            for d in candidates
        ]

        # Sort descending by cross-encoder logit
        reranked.sort(key=lambda x: x.cross_encoder_logit, reverse=True)

        k = top_k if top_k is not None else len(reranked)
        result = reranked[:k]

        for rank_idx, doc in enumerate(result, start=1):
            doc.rank = rank_idx
            doc.calibrated_score = ScoreCalibrator.sigmoid(doc.cross_encoder_logit)

        return result


# ---------------------------------------------------------------------------
# 4. Score Calibration & Normalization
# ---------------------------------------------------------------------------

class ScoreCalibrator:
    """Calibrates and normalizes raw model scores and logits into interpretable scales."""

    @staticmethod
    def sigmoid(logit: float, temperature: float = 1.0) -> float:
        """Sigmoid calibration: maps unconstrained logit z in (-inf, +inf) to [0, 1].
        
        Represents estimated posterior probability P(Relevant | q, d).
        T > 1 softens probabilities; T < 1 sharpens confident predictions.
        """
        temp = max(1e-5, float(temperature))
        scaled_z = logit / temp
        # Numerical stability guard against overflow
        if scaled_z >= 40.0:
            return 1.0
        if scaled_z <= -40.0:
            return 0.0
        return float(1.0 / (1.0 + math.exp(-scaled_z)))

    @staticmethod
    def calibrate_sigmoid(
        documents: List[ScoredDocument], temperature: float = 1.0
    ) -> List[ScoredDocument]:
        """Apply sigmoid calibration in-place to documents."""
        for doc in documents:
            doc.calibrated_score = ScoreCalibrator.sigmoid(doc.cross_encoder_logit, temperature)
        return documents

    @staticmethod
    def min_max_normalize(
        scores: Sequence[float], epsilon: float = 1e-9
    ) -> List[float]:
        """Min-max scale a score list into [0, 1]."""
        if not scores:
            return []
        arr = np.array(scores, dtype=np.float64)
        min_v = float(np.min(arr))
        max_v = float(np.max(arr))
        spread = max_v - min_v

        if spread < epsilon:
            return [1.0 for _ in scores]
        return [float((s - min_v) / spread) for s in scores]

    @staticmethod
    def calibrate_min_max(documents: List[ScoredDocument]) -> List[ScoredDocument]:
        """Apply min-max normalization to document cross-encoder logits."""
        if not documents:
            return documents
        raw_scores = [d.cross_encoder_logit for d in documents]
        scaled = ScoreCalibrator.min_max_normalize(raw_scores)
        for doc, s in zip(documents, scaled):
            doc.calibrated_score = s
        return documents

    @staticmethod
    def truncate_with_threshold(
        documents: List[ScoredDocument],
        top_k: int = 5,
        relevance_threshold: float = 0.4,
    ) -> List[ScoredDocument]:
        """Truncate documents to top_k while dropping items below threshold.
        
        Safeguards downstream LLM context window against low-relevance hallucination hazards.
        """
        filtered = [d for d in documents if d.calibrated_score >= relevance_threshold]
        truncated = filtered[:top_k]
        for idx, doc in enumerate(truncated, start=1):
            doc.rank = idx
        return truncated


# ---------------------------------------------------------------------------
# 5. Cascaded Two-Stage Retrieval Pipeline
# ---------------------------------------------------------------------------

class CascadedRetrievalPipeline:
    """Production Two-Stage Retrieval Pipeline:
    
    Stage 1: Coarse Bi-Encoder Candidate Generation (O(N * D)) -> retrieves top-K1 candidates.
    Stage 2: Fine Cross-Encoder Joint Reranking (O(K1 * M * L^2)) -> scores top-K1.
    Stage 3: Score Calibration & Top-K2 Truncation with confidence gating.
    """

    def __init__(
        self,
        bi_encoder: Optional[BiEncoderSimulator] = None,
        cross_encoder: Optional[CrossEncoderSimulator] = None,
    ):
        self.bi_encoder = bi_encoder or BiEncoderSimulator()
        self.cross_encoder = cross_encoder or CrossEncoderSimulator()

    def execute(
        self,
        query: str,
        corpus: Dict[str, str],
        candidate_k: int = 10,
        final_k: int = 3,
        temperature: float = 1.0,
        threshold: float = 0.35,
    ) -> Tuple[List[ScoredDocument], List[ScoredDocument]]:
        """Run the end-to-end two-stage retrieval pipeline.
        
        Returns:
            (stage1_candidates, stage2_reranked_and_calibrated)
        """
        # Stage 1: Coarse candidate generation
        candidates = self.bi_encoder.retrieve_candidates(query, corpus, top_k=candidate_k)

        # Stage 2: Cross-Encoder reranking
        reranked = self.cross_encoder.rerank(query, candidates)

        # Stage 3: Sigmoid calibration
        calibrated = ScoreCalibrator.calibrate_sigmoid(reranked, temperature=temperature)

        # Top-K2 truncation with confidence threshold
        final_selection = ScoreCalibrator.truncate_with_threshold(
            calibrated, top_k=final_k, relevance_threshold=threshold
        )

        return candidates, final_selection


# ---------------------------------------------------------------------------
# 6. Information Retrieval Evaluation Metrics
# ---------------------------------------------------------------------------

class RankingMetrics:
    """Calculates standard Information Retrieval (IR) evaluation metrics."""

    @staticmethod
    def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: set, k: int) -> float:
        """Precision@k = (Relevant docs in top-k) / k."""
        if k <= 0:
            return 0.0
        top_k = retrieved_ids[:k]
        if not top_k:
            return 0.0
        hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return float(hits / k)

    @staticmethod
    def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: set, k: int) -> float:
        """Recall@k = (Relevant docs in top-k) / Total Relevant docs."""
        if not relevant_ids:
            return 0.0
        top_k = retrieved_ids[:k]
        hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return float(hits / len(relevant_ids))

    @staticmethod
    def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: set) -> float:
        """Reciprocal Rank (RR) = 1 / rank_of_first_relevant_doc."""
        for rank_idx, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return float(1.0 / rank_idx)
        return 0.0

    @staticmethod
    def ndcg_at_k(
        retrieved_ids: Sequence[str],
        relevance_grades: Dict[str, float],
        k: int,
    ) -> float:
        """Normalized Discounted Cumulative Gain (NDCG@k).
        
        DCG@k = sum_{i=1}^k (2^{rel_i} - 1) / log2(i + 1)
        """
        if k <= 0:
            return 0.0

        top_k = retrieved_ids[:k]
        dcg = 0.0
        for i, doc_id in enumerate(top_k, start=1):
            rel = relevance_grades.get(doc_id, 0.0)
            dcg += (math.pow(2.0, rel) - 1.0) / math.log2(i + 1.0)

        # Ideal DCG
        ideal_rels = sorted(relevance_grades.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_rels, start=1):
            idcg += (math.pow(2.0, rel) - 1.0) / math.log2(i + 1.0)

        if idcg == 0.0:
            return 0.0
        return float(dcg / idcg)


# ---------------------------------------------------------------------------
# 7. Optional LLM-as-a-Reranker (Groq API Integration)
# ---------------------------------------------------------------------------

class GroqLLMReranker:
    """Optional Groq-powered LLM-as-a-Reranker.
    
    Prompts an LLM to act as a cross-attention judge, scoring passage relevance
    from 0 to 10 with concise reasoning.
    Gracefully falls back to offline CrossEncoderSimulator when API key is missing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        fallback_simulator: Optional[CrossEncoderSimulator] = None,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.fallback_simulator = fallback_simulator or CrossEncoderSimulator()
        self.client = None

        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except Exception:
                self.client = None

    def rerank(
        self,
        query: str,
        candidates: List[ScoredDocument],
        top_k: Optional[int] = None,
    ) -> List[ScoredDocument]:
        """Rerank candidates using Groq LLM or deterministic fallback."""
        if not self.client:
            # Offline deterministic fallback
            return self.fallback_simulator.rerank(query, candidates, top_k=top_k)

        # Format candidates into structured prompt
        doc_prompts = []
        for d in candidates:
            doc_prompts.append(f"[{d.doc_id}]: {d.text}")
        passages_text = "\n\n".join(doc_prompts)

        prompt = (
            f"You are an expert search reranker. Score the relevance of each candidate passage "
            f"to the user query on a strict scale of 0.0 (completely irrelevant) to 10.0 (direct exact answer).\n\n"
            f"User Query: {query}\n\n"
            f"Candidate Passages:\n{passages_text}\n\n"
            f"Respond strictly in the format:\n"
            f"[doc_id]: <score_between_0.0_and_10.0> | <brief_reason>\n"
            f"Do not include preamble."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=600,
            )
            raw_text = response.choices[0].message.content or ""
            
            # Parse scores
            scores_by_id: Dict[str, float] = {}
            for line in raw_text.splitlines():
                match = re.search(r"\[([\w\-]+)\]:\s*([\d.]+)", line)
                if match:
                    d_id = match.group(1)
                    score = float(match.group(2))
                    scores_by_id[d_id] = score

            reranked: List[ScoredDocument] = []
            for d in candidates:
                llm_score = scores_by_id.get(d.doc_id, 0.0)
                reranked.append(
                    ScoredDocument(
                        doc_id=d.doc_id,
                        text=d.text,
                        bi_encoder_score=d.bi_encoder_score,
                        cross_encoder_logit=llm_score - 5.0,  # Center around 0
                        calibrated_score=llm_score / 10.0,
                        metadata={**d.metadata, "llm_score": llm_score},
                    )
                )

            reranked.sort(key=lambda x: x.calibrated_score, reverse=True)
            k = top_k if top_k is not None else len(reranked)
            for idx, doc in enumerate(reranked[:k], start=1):
                doc.rank = idx
            return reranked[:k]

        except Exception:
            # Fall back to local simulator on network or rate limit failure
            return self.fallback_simulator.rerank(query, candidates, top_k=top_k)
