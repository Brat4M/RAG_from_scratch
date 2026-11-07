"""Query Transformation, Expansion, and Rewriting Utilities.

Provides both deterministic offline CPU-only fallbacks and optional Groq LLM-powered
implementations for:
1. Query Rewriting and Expansion (slang resolution, technical term densification).
2. Step-Back Prompting (concept abstraction for foundational context).
3. Sub-Query Decomposition (atomic multi-intent splitting and RRF fusion).
4. Hypothetical Document Embeddings (HyDE - hallucinated passage embedding).
"""

from __future__ import annotations

import os
import re
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# 1. Deterministic Semantic Embedding for Offline CPU Vector Operations
# ---------------------------------------------------------------------------

class DeterministicSemanticEmbedder:
    """Lightweight, deterministic semantic embedder using subword n-gram hashing.
    
    Zero-network, zero-download, 100% offline CPU-executable embedding generator.
    Produces dense, normalized L2 vectors where lexical and morphological similarity
    translate to continuous cosine proximity.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        """Compute an L2-normalized dense vector for the input text."""
        vec = np.zeros(self.dim, dtype=np.float32)
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        words = [w for w in cleaned.split() if w]

        for word in words:
            # Word-level hash bucket
            w_idx = zlib.crc32(word.encode("utf-8")) % self.dim
            vec[w_idx] += 1.0

            # Subword character tri-grams to capture morphology
            if len(word) >= 3:
                for i in range(len(word) - 2):
                    trigram = word[i:i+3]
                    tg_idx = zlib.crc32(trigram.encode("utf-8")) % self.dim
                    vec[tg_idx] += 0.35

        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Batch compute embeddings as an (N, dim) matrix."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.vstack([self.embed(t) for t in texts])


# ---------------------------------------------------------------------------
# 2. Query Rewriting & Expansion
# ---------------------------------------------------------------------------

class QueryRewriter:
    """Transforms raw, colloquial, or ill-formed queries into search-optimized queries."""

    # Domain abbreviation and synonym mapping
    TERM_EXPANSIONS: Dict[str, str] = {
        "oom": "out of memory cuda allocation failure",
        "ann": "approximate nearest neighbor search hnsw ivf",
        "vram": "gpu video ram memory allocation",
        "chroma": "chromadb local vector database storage",
        "vector db": "vector database embeddings index similarity",
        "rrf": "reciprocal rank fusion hybrid rank merge",
        "hyde": "hypothetical document embeddings zero shot dense retrieval",
        "knn": "exact k-nearest neighbors euclidean cosine search",
    }

    STOP_PHRASES: List[str] = [
        "can you tell me",
        "i want to know",
        "please explain",
        "what is the deal with",
        "how does",
        "why is my",
        "help me understand",
    ]

    def __init__(self, groq_api_key: Optional[str] = None):
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY")

    def rewrite_offline(self, query: str) -> str:
        """Deterministic rule-based query rewrite and expansion."""
        q_clean = query.strip()
        lower_q = q_clean.lower()

        # Remove conversational filler
        for phrase in self.STOP_PHRASES:
            if lower_q.startswith(phrase):
                q_clean = q_clean[len(phrase):].strip(" ?:.,!")
                lower_q = q_clean.lower()

        # Replace multi-word and single-word phrases
        rewritten = lower_q
        for term, expansion in self.TERM_EXPANSIONS.items():
            pattern = r"\b" + re.escape(term) + r"\b"
            rewritten = re.sub(pattern, expansion, rewritten)

        # Normalize whitespace
        rewritten = " ".join(rewritten.split())
        return rewritten if rewritten else query

    def rewrite_llm(self, query: str, model: str = "llama-3.3-70b-versatile") -> str:
        """Groq LLM-powered query rewriting."""
        if not self.api_key:
            return self.rewrite_offline(query)

        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)
            prompt = (
                "You are an expert search query engineer. Rewrite the following user query into a concise, "
                "canonical, keyword-rich search query optimized for vector and keyword retrieval. "
                "Preserve technical intent, resolve ambiguities, and output ONLY the rewritten query without commentary.\n\n"
                f"User Query: {query}\n"
                "Optimized Search Query:"
            )
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            content = response.choices[0].message.content
            return content.strip().strip('"') if content else self.rewrite_offline(query)
        except Exception:
            return self.rewrite_offline(query)

    def rewrite(self, query: str, use_llm: bool = False) -> str:
        """Unified rewrite entrypoint."""
        if use_llm and self.api_key:
            return self.rewrite_llm(query)
        return self.rewrite_offline(query)


# ---------------------------------------------------------------------------
# 3. Step-Back Prompting (Concept Abstraction)
# ---------------------------------------------------------------------------

class StepBackPrompter:
    """Generates higher-level, foundational conceptual queries from granular questions."""

    STEP_BACK_RULES: Dict[str, str] = {
        "cuda": "What are the core GPU memory architecture and allocation limits in PyTorch CUDA?",
        "oom": "What causes out of memory errors and memory fragmentation in deep learning frameworks?",
        "hnsw": "What are the foundational graph algorithms and trade-offs for Approximate Nearest Neighbor search?",
        "ivf": "How do inverted file indexes partition continuous vector spaces for nearest neighbor retrieval?",
        "chromadb": "What is the architectural design and persistence mechanism of modern vector databases?",
        "batch size": "How do batch sizes and activation tensors affect GPU memory consumption during neural network training?",
    }

    DEFAULT_STEP_BACK = "What are the fundamental architectural principles and constraints governing this system?"

    def __init__(self, groq_api_key: Optional[str] = None):
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY")

    def step_back_offline(self, query: str) -> str:
        """Deterministic rule-based abstraction."""
        lower_q = query.lower()
        for keyword, abstract_query in self.STEP_BACK_RULES.items():
            if keyword in lower_q:
                return abstract_query
        return self.DEFAULT_STEP_BACK

    def step_back_llm(self, query: str, model: str = "llama-3.3-70b-versatile") -> str:
        """Groq LLM-powered step-back abstraction."""
        if not self.api_key:
            return self.step_back_offline(query)

        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)
            prompt = (
                "You are an expert at Step-Back Prompting for information retrieval. Given a specific, granular question, "
                "step back and formulate a broader, higher-level conceptual question about the underlying principles, "
                "theory, or architecture that govern the problem. Output ONLY the step-back question.\n\n"
                f"Granular Question: {query}\n"
                "Step-Back Question:"
            )
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            content = response.choices[0].message.content
            return content.strip().strip('"') if content else self.step_back_offline(query)
        except Exception:
            return self.step_back_offline(query)

    def generate(self, query: str, use_llm: bool = False) -> str:
        """Unified step-back entrypoint."""
        if use_llm and self.api_key:
            return self.step_back_llm(query)
        return self.step_back_offline(query)


# ---------------------------------------------------------------------------
# 4. Sub-Query Decomposition (Multi-Hop Divide & Conquer)
# ---------------------------------------------------------------------------

class SubQueryDecomposer:
    """Decomposes compound or comparative queries into atomic sub-queries."""

    def __init__(self, groq_api_key: Optional[str] = None):
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY")

    def decompose_offline(self, query: str) -> List[str]:
        """Deterministic rule-based sub-query splitting."""
        lower_q = query.strip()

        # Handle comparative queries: "compare X and Y", "X vs Y", "difference between X and Y"
        comp_match = re.search(
            r"(?:compare|differences? between)\s+(.*?)\s+(?:and|versus|vs\.?)\s+(.*)",
            lower_q,
            re.IGNORECASE,
        )
        if comp_match:
            part_a = comp_match.group(1).strip(" ?:.,")
            part_b = comp_match.group(2).strip(" ?:.,")
            return [
                f"Architectural design and characteristics of {part_a}",
                f"Architectural design and characteristics of {part_b}",
            ]

        # Handle "vs" split directly: "HNSW vs IVF"
        vs_match = re.split(r"\s+(?:vs\.?|versus)\s+", lower_q, flags=re.IGNORECASE)
        if len(vs_match) >= 2:
            return [
                f"Core mechanics and performance of {vs_match[0].strip()}",
                f"Core mechanics and performance of {vs_match[1].strip()}",
            ]

        # Handle conjunction "and": "How does ChromaDB store vectors and how does PyTorch handle CUDA memory?"
        and_parts = re.split(r",?\s+and\s+(?:how\s+does\s+|how\s+to\s+|what\s+is\s+)?", lower_q, flags=re.IGNORECASE)
        if len(and_parts) >= 2 and all(len(p.strip()) > 10 for p in and_parts):
            return [p.strip(" ?:.,") for p in and_parts]

        # Default fallback: return original query as single atomic sub-query
        return [query.strip()]

    def decompose_llm(self, query: str, model: str = "llama-3.3-70b-versatile") -> List[str]:
        """Groq LLM-powered sub-query decomposition."""
        if not self.api_key:
            return self.decompose_offline(query)

        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)
            prompt = (
                "Decompose the following complex user question into 2 to 3 distinct, independent atomic search sub-queries. "
                "Each sub-query must be standalone and suitable for vector search. "
                "Output each sub-query on a new line starting with a dash (-), with no other text.\n\n"
                f"Query: {query}\n"
                "Atomic Sub-Queries:"
            )
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150,
            )
            content = response.choices[0].message.content or ""
            lines = [
                re.sub(r"^[-*\d.\s]+", "", line).strip()
                for line in content.split("\n")
                if line.strip()
            ]
            valid_queries = [l for l in lines if len(l) > 5]
            return valid_queries if valid_queries else self.decompose_offline(query)
        except Exception:
            return self.decompose_offline(query)

    def decompose(self, query: str, use_llm: bool = False) -> List[str]:
        """Unified decomposition entrypoint."""
        if use_llm and self.api_key:
            return self.decompose_llm(query)
        return self.decompose_offline(query)


# ---------------------------------------------------------------------------
# 5. Hypothetical Document Embeddings (HyDE)
# ---------------------------------------------------------------------------

class HyDEGenerator:
    """Generates synthetic hypothetical answer passages to bridge the query-document semantic gap."""

    HYDE_TEMPLATES: Dict[str, str] = {
        "gpu": (
            "When running neural network models with large batch sizes, PyTorch allocates extensive GPU VRAM "
            "for activations and tensor weights. If device memory capacity is exceeded, CUDA throws out-of-memory "
            "errors and memory fragmentation prevents contiguous page allocation."
        ),
        "oom": (
            "PyTorch CUDA out-of-memory errors occur when GPU memory allocations exceed physical device capacity "
            "during tensor operations or large batch sizes. Memory fragmentation further prevents contiguous allocations."
        ),
        "hnsw": (
            "HNSW (Hierarchical Navigable Small World) graphs construct multi-layer proximity networks for fast "
            "approximate nearest neighbor search with high recall and logarithmic search complexity."
        ),
        "ivf": (
            "Inverted File Index (IVF) partitions continuous vector space into Voronoi cells using k-means centroids, "
            "reducing the number of distance calculations during approximate nearest neighbor querying."
        ),
        "chromadb": (
            "ChromaDB uses SQLite for relational metadata storage and DuckDB or Rust vector indices for fast "
            "persistent similarity search with compound metadata filtering."
        ),
        "hybrid": (
            "Hybrid search combines dense vector similarity with sparse BM25 lexical keyword matching using "
            "Reciprocal Rank Fusion (RRF) to maximize both semantic and lexical recall."
        ),
    }

    DEFAULT_PASSAGE = (
        "This technical document provides authoritative architectural details, system mechanisms, "
        "and empirical trade-offs regarding information retrieval, vector search, and computational systems."
    )

    def __init__(self, groq_api_key: Optional[str] = None):
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY")

    def generate_offline(self, query: str) -> str:
        """Deterministic template-based hypothetical passage generation."""
        lower_q = query.lower()
        for keyword, template in self.HYDE_TEMPLATES.items():
            if keyword in lower_q:
                return template
        return f"{self.DEFAULT_PASSAGE} Specifically regarding: {query.strip()}."

    def generate_llm(self, query: str, model: str = "llama-3.3-70b-versatile") -> str:
        """Groq LLM-powered hypothetical passage generation."""
        if not self.api_key:
            return self.generate_offline(query)

        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)
            prompt = (
                "Write an authoritative technical passage that answers the following query directly and factually. "
                "Write in textbook or documentation style. Do not include conversational greetings or preamble; "
                "begin the passage immediately.\n\n"
                f"Query: {query}\n"
                "Passage:"
            )
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            content = response.choices[0].message.content
            return content.strip() if content else self.generate_offline(query)
        except Exception:
            return self.generate_offline(query)

    def generate(self, query: str, use_llm: bool = False) -> str:
        """Unified HyDE passage generator."""
        if use_llm and self.api_key:
            return self.generate_llm(query)
        return self.generate_offline(query)


# ---------------------------------------------------------------------------
# 6. Reciprocal Rank Fusion (RRF) Multi-List Rank Aggregation
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
    top_n: int = 5,
) -> List[Tuple[str, float]]:
    """Merges multiple ranked candidate lists using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_Score(d) = sum_{list L} [ 1 / (k + rank_L(d)) ]

    Args:
        ranked_lists: A list of candidate lists, where each candidate list contains (doc_id, score) tuples.
        k: Smoothing constant preventing high ranks from disproportionately dominating (standard k=60).
        top_n: Number of top results to return.

    Returns:
        Sorted list of (doc_id, fused_rrf_score) tuples in descending score order.
    """
    scores: Dict[str, float] = {}

    for candidates in ranked_lists:
        for rank, (doc_id, _) in enumerate(candidates, start=1):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank)

    # Sort descending by fused RRF score, with stable alphanumeric tie-breaking
    sorted_items = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return sorted_items[:top_n]


# ---------------------------------------------------------------------------
# 7. Unified Query Transformation & Retrieval Engine
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    doc_id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueryTransformationEngine:
    """Unified engine executing naive, rewritten, step-back, decomposed, and HyDE retrieval."""

    def __init__(
        self,
        corpus: Dict[str, str],
        embedder: Optional[DeterministicSemanticEmbedder] = None,
        groq_api_key: Optional[str] = None,
    ):
        self.corpus = corpus
        self.embedder = embedder or DeterministicSemanticEmbedder()
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY")

        self.rewriter = QueryRewriter(self.api_key)
        self.step_back_prompter = StepBackPrompter(self.api_key)
        self.decomposer = SubQueryDecomposer(self.api_key)
        self.hyde_generator = HyDEGenerator(self.api_key)

        # Precompute corpus embeddings
        self.doc_ids = list(corpus.keys())
        self.doc_texts = [corpus[k] for k in self.doc_ids]
        self.doc_embeddings = self.embedder.embed_batch(self.doc_texts)

    def _dense_search(self, query_text: str, top_k: int = 3) -> List[SearchResult]:
        """Performs raw cosine similarity vector search over the corpus."""
        q_vec = self.embedder.embed(query_text)
        # Cosine similarity (vectors are L2-normalized, so dot product = cosine)
        sims = np.dot(self.doc_embeddings, q_vec)
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    score=float(sims[idx]),
                    text=self.corpus[doc_id],
                )
            )
        return results

    def retrieve_naive(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Baseline naive dense retrieval without query transformation."""
        return self._dense_search(query, top_k=top_k)

    def retrieve_rewritten(self, query: str, top_k: int = 3, use_llm: bool = False) -> Tuple[str, List[SearchResult]]:
        """Rewrites the query before dense retrieval."""
        rewritten_q = self.rewriter.rewrite(query, use_llm=use_llm)
        return rewritten_q, self._dense_search(rewritten_q, top_k=top_k)

    def retrieve_step_back(
        self, query: str, top_k: int = 3, use_llm: bool = False
    ) -> Tuple[str, List[SearchResult]]:
        """Performs step-back abstraction, searches for both queries, and fuses with RRF."""
        step_back_q = self.step_back_prompter.generate(query, use_llm=use_llm)

        res_orig = [(r.doc_id, r.score) for r in self._dense_search(query, top_k=top_k * 2)]
        res_step = [(r.doc_id, r.score) for r in self._dense_search(step_back_q, top_k=top_k * 2)]

        fused = reciprocal_rank_fusion([res_orig, res_step], top_n=top_k)
        results = [
            SearchResult(doc_id=doc_id, score=score, text=self.corpus[doc_id])
            for doc_id, score in fused
        ]
        return step_back_q, results

    def retrieve_decomposed(
        self, query: str, top_k: int = 3, use_llm: bool = False
    ) -> Tuple[List[str], List[SearchResult]]:
        """Decomposes compound query into sub-queries, executes searches, and fuses with RRF."""
        sub_queries = self.decomposer.decompose(query, use_llm=use_llm)
        candidate_lists = []

        for sub_q in sub_queries:
            sub_results = [(r.doc_id, r.score) for r in self._dense_search(sub_q, top_k=top_k * 2)]
            candidate_lists.append(sub_results)

        fused = reciprocal_rank_fusion(candidate_lists, top_n=top_k)
        results = [
            SearchResult(doc_id=doc_id, score=score, text=self.corpus[doc_id])
            for doc_id, score in fused
        ]
        return sub_queries, results

    def retrieve_hyde(
        self, query: str, top_k: int = 3, use_llm: bool = False
    ) -> Tuple[str, List[SearchResult]]:
        """Generates a hypothetical passage, embeds it, and retrieves documents."""
        hypo_doc = self.hyde_generator.generate(query, use_llm=use_llm)
        # Search using the hypothetical passage vector
        results = self._dense_search(hypo_doc, top_k=top_k)
        return hypo_doc, results
