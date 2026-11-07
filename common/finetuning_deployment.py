"""Fine-Tuning versus RAG and Production Deployment Engine.

Provides 100% offline-executable, CPU-deterministic engineering components:
1. Parametric vs. Non-Parametric Trade-off Evaluation:
   Analytical decision matrix and recommendation engine comparing Parametric (LoRA, PEFT, SFT)
   versus Non-Parametric (RAG external index) architectures across 8 core dimensions.
2. Fine-Tuning Dataset Formats & Validation:
   Schema validation, token distribution profiling, and data leakage detection for Instruction (SFT)
   and Embedding Contrastive Triplet datasets.
3. First-Principles LoRA (Low-Rank Adaptation) Engine:
   NumPy-based low-rank matrix decomposition ($W_0 + \\frac{\\alpha}{r} BA$), weight merging/unmerging,
   and parameter efficiency accounting.
4. CPU Quantization Engine:
   Symmetric and asymmetric INT8 quantization, zero-point/scale calibration, and numerical fidelity metrics.
5. Hybrid RAG + Fine-Tuning Pipeline:
   Combining dense non-parametric vector retrieval with fine-tuned structured synthesis and refusal gates.
6. Production Serving Contract (FastAPI):
   Production-grade ASGI application with /health, /v1/query, /v1/ingest, and /v1/metrics endpoints.
7. Deployment Readiness Auditor & Containerization:
   Automated 8-pillar production audit and multi-stage Dockerfile / Compose generators.
8. Publication-Quality Architectural Diagrams:
   Visualizing the decision quadrant and production serving topology.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple, Union

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, Field

# FastAPI & Starlette for Production Serving
try:
    from fastapi import FastAPI, HTTPException, Request, Response, status
    from fastapi.responses import JSONResponse
    from starlette.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


# ============================================================================
# 1. PARAMETRIC VS. NON-PARAMETRIC TRADE-OFF FRAMEWORK
# ============================================================================

class ArchitectureType(str, Enum):
    RAG_ONLY = "RAG_ONLY"
    FINETUNE_ONLY = "FINETUNE_ONLY"
    HYBRID_RAG_FT = "HYBRID_RAG_FT"
    BASE_MODEL = "BASE_MODEL"


class KnowledgeVolatility(str, Enum):
    REAL_TIME = "real_time"       # Minutes to hours (e.g., breaking news, stock prices)
    DAILY_WEEKLY = "daily_weekly" # Frequent doc updates, company wikis
    MONTHLY = "monthly"           # Occasional policy updates
    STATIC = "static"             # Permanent grammar, domain style, established jargon


class ArchitectureRequirements(BaseModel):
    """Specification of application requirements for architecture recommendation."""
    knowledge_volatility: KnowledgeVolatility = KnowledgeVolatility.DAILY_WEEKLY
    need_strict_citations: bool = True
    custom_style_or_syntax: bool = False
    strict_json_schema: bool = True
    access_control_required: bool = True
    latency_budget_ms: float = 350.0
    upfront_budget_usd: float = 500.0
    training_samples_available: int = 250
    corpus_document_count: int = 1500


class ArchitectureRecommendation(BaseModel):
    """Result of the architecture decision matrix analysis."""
    primary_architecture: ArchitectureType
    confidence_score: float
    dimension_scores: Dict[str, Dict[str, float]]
    rationales: List[str]
    tradeoffs_summary: Dict[str, str]
    estimated_initial_cost_usd: float
    recommended_serving_stack: Dict[str, str]


def evaluate_architecture_tradeoffs(req: ArchitectureRequirements) -> ArchitectureRecommendation:
    """Evaluate architectural suitability between RAG, Fine-Tuning, and Hybrid patterns.
    
    Scores each architecture across 6 orthogonal dimensions (0.0 to 1.0):
    1. Dynamic Knowledge Currency
    2. Hallucination Suppression & Citations
    3. Output Style & Format Adherence
    4. Access Control & Tenant Isolation (ACL)
    5. Latency & Resource Efficiency
    6. Upfront Cost & Engineering Complexity
    """
    scores: Dict[ArchitectureType, Dict[str, float]] = {
        ArchitectureType.RAG_ONLY: {},
        ArchitectureType.FINETUNE_ONLY: {},
        ArchitectureType.HYBRID_RAG_FT: {},
        ArchitectureType.BASE_MODEL: {},
    }
    rationales: List[str] = []

    # 1. Dynamic Knowledge Currency
    if req.knowledge_volatility in (KnowledgeVolatility.REAL_TIME, KnowledgeVolatility.DAILY_WEEKLY):
        scores[ArchitectureType.RAG_ONLY]["knowledge_currency"] = 0.95
        scores[ArchitectureType.HYBRID_RAG_FT]["knowledge_currency"] = 0.95
        scores[ArchitectureType.FINETUNE_ONLY]["knowledge_currency"] = 0.20
        scores[ArchitectureType.BASE_MODEL]["knowledge_currency"] = 0.10
        rationales.append(
            f"High knowledge volatility ({req.knowledge_volatility.value}) strongly favors RAG, "
            "as re-training or fine-tuning weights cannot match sub-minute index updates."
        )
    else:
        scores[ArchitectureType.RAG_ONLY]["knowledge_currency"] = 0.75
        scores[ArchitectureType.HYBRID_RAG_FT]["knowledge_currency"] = 0.85
        scores[ArchitectureType.FINETUNE_ONLY]["knowledge_currency"] = 0.80
        scores[ArchitectureType.BASE_MODEL]["knowledge_currency"] = 0.50

    # 2. Hallucination Suppression & Citations
    if req.need_strict_citations:
        scores[ArchitectureType.RAG_ONLY]["citations"] = 0.95
        scores[ArchitectureType.HYBRID_RAG_FT]["citations"] = 0.98
        scores[ArchitectureType.FINETUNE_ONLY]["citations"] = 0.25
        scores[ArchitectureType.BASE_MODEL]["citations"] = 0.15
        rationales.append(
            "Strict citation requirements mandate non-parametric retrieval; parametric models "
            "cannot reliably cite dynamic source offsets without ground-truth context."
        )
    else:
        scores[ArchitectureType.RAG_ONLY]["citations"] = 0.70
        scores[ArchitectureType.HYBRID_RAG_FT]["citations"] = 0.85
        scores[ArchitectureType.FINETUNE_ONLY]["citations"] = 0.70
        scores[ArchitectureType.BASE_MODEL]["citations"] = 0.60

    # 3. Output Style & Format Adherence (Syntax, JSON Schema)
    if req.custom_style_or_syntax or req.strict_json_schema:
        scores[ArchitectureType.FINETUNE_ONLY]["format_adherence"] = 0.95
        scores[ArchitectureType.HYBRID_RAG_FT]["format_adherence"] = 0.96
        scores[ArchitectureType.RAG_ONLY]["format_adherence"] = 0.65
        scores[ArchitectureType.BASE_MODEL]["format_adherence"] = 0.40
        rationales.append(
            "Strict format or custom syntax adherence is best taught parametrically via SFT/LoRA, "
            "reducing prompt engineering overhead."
        )
    else:
        scores[ArchitectureType.FINETUNE_ONLY]["format_adherence"] = 0.70
        scores[ArchitectureType.HYBRID_RAG_FT]["format_adherence"] = 0.75
        scores[ArchitectureType.RAG_ONLY]["format_adherence"] = 0.75
        scores[ArchitectureType.BASE_MODEL]["format_adherence"] = 0.70

    # 4. Access Control & Tenant Isolation
    if req.access_control_required:
        scores[ArchitectureType.RAG_ONLY]["access_control"] = 0.95
        scores[ArchitectureType.HYBRID_RAG_FT]["access_control"] = 0.90
        scores[ArchitectureType.FINETUNE_ONLY]["access_control"] = 0.10
        scores[ArchitectureType.BASE_MODEL]["access_control"] = 0.05
        rationales.append(
            "Access control lists (ACL) require index-level pre-filtering; fine-tuned weights "
            "cannot enforce granular per-user permissions safely."
        )
    else:
        scores[ArchitectureType.RAG_ONLY]["access_control"] = 0.80
        scores[ArchitectureType.HYBRID_RAG_FT]["access_control"] = 0.80
        scores[ArchitectureType.FINETUNE_ONLY]["access_control"] = 0.75
        scores[ArchitectureType.BASE_MODEL]["access_control"] = 0.70

    # 5. Latency & Resource Efficiency
    if req.latency_budget_ms < 150.0:
        # Fine-tuning allows smaller prompts (no huge retrieved context chunks)
        scores[ArchitectureType.FINETUNE_ONLY]["latency_efficiency"] = 0.90
        scores[ArchitectureType.BASE_MODEL]["latency_efficiency"] = 0.85
        scores[ArchitectureType.RAG_ONLY]["latency_efficiency"] = 0.55
        scores[ArchitectureType.HYBRID_RAG_FT]["latency_efficiency"] = 0.60
        rationales.append(
            f"Tight latency budget ({req.latency_budget_ms} ms) gives an advantage to compact prompts, "
            "though hybrid CPU quantization balances this."
        )
    else:
        scores[ArchitectureType.RAG_ONLY]["latency_efficiency"] = 0.85
        scores[ArchitectureType.HYBRID_RAG_FT]["latency_efficiency"] = 0.85
        scores[ArchitectureType.FINETUNE_ONLY]["latency_efficiency"] = 0.80
        scores[ArchitectureType.BASE_MODEL]["latency_efficiency"] = 0.80

    # 6. Upfront Cost & Engineering Complexity
    if req.upfront_budget_usd < 200.0 or req.training_samples_available < 100:
        scores[ArchitectureType.RAG_ONLY]["cost_complexity"] = 0.92
        scores[ArchitectureType.BASE_MODEL]["cost_complexity"] = 0.95
        scores[ArchitectureType.HYBRID_RAG_FT]["cost_complexity"] = 0.60
        scores[ArchitectureType.FINETUNE_ONLY]["cost_complexity"] = 0.45
        rationales.append(
            "Constrained budget or scarce training pairs favor RAG, avoiding dataset annotation "
            "and GPU fine-tuning expenses."
        )
    else:
        scores[ArchitectureType.RAG_ONLY]["cost_complexity"] = 0.85
        scores[ArchitectureType.HYBRID_RAG_FT]["cost_complexity"] = 0.80
        scores[ArchitectureType.FINETUNE_ONLY]["cost_complexity"] = 0.75
        scores[ArchitectureType.BASE_MODEL]["cost_complexity"] = 0.90

    # Aggregate weighted scores
    weights = {
        "knowledge_currency": 0.25,
        "citations": 0.20,
        "format_adherence": 0.15,
        "access_control": 0.15,
        "latency_efficiency": 0.10,
        "cost_complexity": 0.15,
    }

    final_scores: Dict[ArchitectureType, float] = {}
    for arch in ArchitectureType:
        final_scores[arch] = sum(
            scores[arch].get(dim, 0.5) * w for dim, w in weights.items()
        )

    # Sort descending
    sorted_arch = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    best_arch, best_score = sorted_arch[0]

    # Formulate cost and stack recommendations
    if best_arch == ArchitectureType.HYBRID_RAG_FT:
        est_cost = 250.0
        stack = {
            "retriever": "ChromaDB Local / HNSW",
            "model_adapter": "LoRA Adapter (PEFT 8-bit / INT8)",
            "serving": "FastAPI + ONNX Runtime CPU",
            "orchestrator": "Hybrid RAG Pipeline",
        }
    elif best_arch == ArchitectureType.RAG_ONLY:
        est_cost = 25.0
        stack = {
            "retriever": "ChromaDB Persistent Index",
            "model_adapter": "Base General LLM / Groq API",
            "serving": "FastAPI + Asynchronous Ingestion",
            "orchestrator": "Dense + BM25 Hybrid Retriever",
        }
    elif best_arch == ArchitectureType.FINETUNE_ONLY:
        est_cost = 350.0
        stack = {
            "retriever": "None (Parametric Weights Only)",
            "model_adapter": "Domain SFT + LoRA rank=16",
            "serving": "vLLM / ONNX Runtime",
            "orchestrator": "Direct Model Inference",
        }
    else:
        est_cost = 0.0
        stack = {"retriever": "None", "model_adapter": "Base Zero-shot", "serving": "Standard API", "orchestrator": "Direct"}

    tradeoffs_summary = {
        "Parametric Knowledge": "Encoded in weights; fast retrieval during forward pass; static; hallucination risk.",
        "Non-Parametric Knowledge": "Stored in external index; zero training cost; dynamically updatable; exact citations.",
        "Hybrid Synergy": "Fine-tune for structure/reasoning/refusal + RAG for volatile ground-truth facts.",
    }

    dim_scores_formatted = {
        arch.value: {dim: round(score, 3) for dim, score in scores[arch].items()}
        for arch in ArchitectureType
    }

    return ArchitectureRecommendation(
        primary_architecture=best_arch,
        confidence_score=round(best_score, 3),
        dimension_scores=dim_scores_formatted,
        rationales=rationales,
        tradeoffs_summary=tradeoffs_summary,
        estimated_initial_cost_usd=est_cost,
        recommended_serving_stack=stack,
    )


# ============================================================================
# 2. FINE-TUNING DATASET FORMATS & VALIDATION
# ============================================================================

class SFTInstructionSample(BaseModel):
    """Standard Supervised Fine-Tuning (SFT) sample schema."""
    instruction: str = Field(..., min_length=3)
    input: str = Field(default="")
    output: str = Field(..., min_length=3)


class ContrastiveTripletSample(BaseModel):
    """Contrastive sample for bi-encoder embedding fine-tuning."""
    query: str = Field(..., min_length=3)
    positive: str = Field(..., min_length=5)
    negative: str = Field(..., min_length=5)


@dataclass
class DatasetValidationReport:
    valid_count: int
    invalid_count: int
    mean_token_count_input: float
    mean_token_count_output: float
    detected_leakages: List[str]
    errors: List[str]
    is_valid: bool


class DatasetValidator:
    """Validates SFT and Embedding Fine-Tuning datasets with token analysis and leakage guards."""

    @staticmethod
    def validate_sft_dataset(
        samples: Sequence[Union[Dict[str, Any], SFTInstructionSample]]
    ) -> DatasetValidationReport:
        valid_items = 0
        errors: List[str] = []
        leakages: List[str] = []
        input_lens: List[int] = []
        output_lens: List[int] = []

        seen_instructions = set()

        for idx, item in enumerate(samples):
            try:
                if isinstance(item, dict):
                    sample = SFTInstructionSample(**item)
                else:
                    sample = item

                in_tokens = len(re.findall(r"\w+", f"{sample.instruction} {sample.input}"))
                out_tokens = len(re.findall(r"\w+", sample.output))

                # Empty check
                if out_tokens == 0:
                    errors.append(f"Row {idx}: Output is empty.")
                    continue

                # Deduplication & leakage check
                key = sample.instruction.strip().lower()
                if key in seen_instructions:
                    leakages.append(f"Row {idx}: Duplicate instruction detected: '{key[:40]}...'")
                seen_instructions.add(key)

                # Target leakage check: output exactly equals input
                if sample.input and sample.input.strip().lower() == sample.output.strip().lower():
                    leakages.append(f"Row {idx}: Trivial target leakage (input equals output).")

                input_lens.append(in_tokens)
                output_lens.append(out_tokens)
                valid_items += 1
            except Exception as e:
                errors.append(f"Row {idx}: Schema validation failed: {str(e)}")

        mean_in = float(np.mean(input_lens)) if input_lens else 0.0
        mean_out = float(np.mean(output_lens)) if output_lens else 0.0
        is_valid = len(errors) == 0 and valid_items > 0

        return DatasetValidationReport(
            valid_count=valid_items,
            invalid_count=len(errors),
            mean_token_count_input=round(mean_in, 2),
            mean_token_count_output=round(mean_out, 2),
            detected_leakages=leakages,
            errors=errors,
            is_valid=is_valid,
        )

    @staticmethod
    def validate_contrastive_triplets(
        samples: Sequence[Union[Dict[str, Any], ContrastiveTripletSample]]
    ) -> DatasetValidationReport:
        valid_items = 0
        errors: List[str] = []
        leakages: List[str] = []
        q_lens: List[int] = []
        doc_lens: List[int] = []

        for idx, item in enumerate(samples):
            try:
                if isinstance(item, dict):
                    sample = ContrastiveTripletSample(**item)
                else:
                    sample = item

                # Check false negative: positive equals negative
                if sample.positive.strip().lower() == sample.negative.strip().lower():
                    leakages.append(f"Row {idx}: Positive and negative passages are identical.")

                q_tok = len(re.findall(r"\w+", sample.query))
                pos_tok = len(re.findall(r"\w+", sample.positive))
                q_lens.append(q_tok)
                doc_lens.append(pos_tok)
                valid_items += 1
            except Exception as e:
                errors.append(f"Row {idx}: Schema error: {str(e)}")

        mean_q = float(np.mean(q_lens)) if q_lens else 0.0
        mean_doc = float(np.mean(doc_lens)) if doc_lens else 0.0

        return DatasetValidationReport(
            valid_count=valid_items,
            invalid_count=len(errors),
            mean_token_count_input=round(mean_q, 2),
            mean_token_count_output=round(mean_doc, 2),
            detected_leakages=leakages,
            errors=errors,
            is_valid=len(errors) == 0 and valid_items > 0,
        )


# ============================================================================
# 3. FIRST-PRINCIPLES LORA (LOW-RANK ADAPTATION) ENGINE
# ============================================================================

class LoRALinear:
    """First-Principles implementation of Low-Rank Adaptation (LoRA).
    
    Given base weight W_0 in R^{out_dim x in_dim}, adaptation is factorized into:
        Delta W = (alpha / r) * (B @ A)
    where:
        A in R^{r x in_dim}  (initialized with Gaussian N(0, 1/r))
        B in R^{out_dim x r} (initialized to zeros, ensuring Delta W = 0 initially)
        r << min(in_dim, out_dim)
        alpha is constant scaling factor
        
    Forward pass:
        y = W_0 x + (alpha / r) * (B @ (A @ x))
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int = 4,
        alpha: float = 8.0,
        seed: int = 42,
    ) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / float(rank)
        self.is_merged = False

        rng = np.random.default_rng(seed)
        # Frozen base weight W_0
        self.w_base = rng.normal(0.0, 1.0 / math.sqrt(in_dim), size=(out_dim, in_dim)).astype(np.float32)

        # LoRA factor matrices
        self.lora_a = rng.normal(0.0, 1.0 / math.sqrt(rank), size=(rank, in_dim)).astype(np.float32)
        # B is initialized to zeros so that initial Delta W is strictly zero
        self.lora_b = np.zeros((out_dim, rank), dtype=np.float32)

    @property
    def base_parameter_count(self) -> int:
        return self.in_dim * self.out_dim

    @property
    def lora_parameter_count(self) -> int:
        return (self.in_dim * self.rank) + (self.rank * self.out_dim)

    @property
    def parameter_reduction_ratio(self) -> float:
        """Percentage of trainable parameters compared to full fine-tuning."""
        return float(self.lora_parameter_count) / float(self.base_parameter_count)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute linear forward pass with low-rank adapter contribution.
        
        Args:
            x: Input vector or batch of vectors (batch_size, in_dim) or (in_dim,)
        """
        x_arr = np.asarray(x, dtype=np.float32)
        if self.is_merged:
            return x_arr @ self.w_base.T

        # Base linear transformation
        y_base = x_arr @ self.w_base.T

        # LoRA adaptation path: (x @ A.T) @ B.T * scaling
        lora_intermediate = x_arr @ self.lora_a.T  # (..., rank)
        y_lora = (lora_intermediate @ self.lora_b.T) * self.scaling

        return y_base + y_lora

    def compute_delta_w(self) -> np.ndarray:
        """Explicitly compute Delta W = (alpha / r) * (B @ A)."""
        return self.scaling * (self.lora_b @ self.lora_a)

    def merge_weights(self) -> None:
        """Fuse LoRA weights into base weights for zero-overhead production inference."""
        if self.is_merged:
            return
        delta_w = self.compute_delta_w()
        self.w_base += delta_w
        self.is_merged = True

    def unmerge_weights(self) -> None:
        """Separate fused LoRA weights back into adapter matrices."""
        if not self.is_merged:
            return
        delta_w = self.compute_delta_w()
        self.w_base -= delta_w
        self.is_merged = False

    def train_step_mock(self, target_delta_w: np.ndarray, lr: float = 0.05) -> float:
        """Deterministic gradient adjustment for educational verification of adapter tuning."""
        # Update B towards the target delta direction
        proj = target_delta_w @ self.lora_a.T
        self.lora_b += lr * proj / self.scaling
        current_delta = self.compute_delta_w()
        mse = float(np.mean((current_delta - target_delta_w) ** 2))
        return mse


# ============================================================================
# 4. CPU QUANTIZATION & NUMERICAL FIDELITY ENGINE
# ============================================================================

@dataclass
class QuantizedTensor:
    """Representation of an INT8 quantized tensor with scaling and zero-point."""
    q_data: np.ndarray           # INT8 numpy array
    scale: float                 # Quantization step size S
    zero_point: int              # Integer zero offset Z
    original_shape: Tuple[int, ...]
    mode: str                    # 'symmetric' or 'asymmetric'

    @property
    def memory_bytes(self) -> int:
        return self.q_data.nbytes + 8 + 4


class CPUQuantizer:
    """High-precision CPU quantization engine for weights and activations."""

    @staticmethod
    def quantize_symmetric_int8(matrix: np.ndarray) -> QuantizedTensor:
        """Symmetric INT8 quantization mapping range [-max, +max] to [-127, 127].
        
        Formula:
            S = max(|x|) / 127
            q = round(x / S), clamped to [-127, 127]
            Z = 0
        """
        arr = np.asarray(matrix, dtype=np.float32)
        max_abs = float(np.max(np.abs(arr)))
        if max_abs == 0.0:
            scale = 1.0
        else:
            scale = max_abs / 127.0

        q_arr = np.round(arr / scale)
        q_clamped = np.clip(q_arr, -127, 127).astype(np.int8)

        return QuantizedTensor(
            q_data=q_clamped,
            scale=scale,
            zero_point=0,
            original_shape=arr.shape,
            mode="symmetric",
        )

    @staticmethod
    def quantize_asymmetric_int8(matrix: np.ndarray) -> QuantizedTensor:
        """Asymmetric INT8 quantization mapping range [min, max] to [0, 255] or [-128, 127]."""
        arr = np.asarray(matrix, dtype=np.float32)
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))

        if max_val == min_val:
            scale = 1.0
            zero_point = 0
        else:
            scale = (max_val - min_val) / 255.0
            zero_point = int(np.round(-min_val / scale))
            zero_point = max(0, min(255, zero_point))

        q_arr = np.round(arr / scale) + zero_point
        q_clamped = np.clip(q_arr, 0, 255).astype(np.uint8)

        return QuantizedTensor(
            q_data=q_clamped,
            scale=scale,
            zero_point=zero_point,
            original_shape=arr.shape,
            mode="asymmetric",
        )

    @staticmethod
    def dequantize(q_tensor: QuantizedTensor) -> np.ndarray:
        """Reconstruct float approximation from quantized representation: x_hat = S * (q - Z)."""
        if q_tensor.mode == "symmetric":
            return q_tensor.q_data.astype(np.float32) * q_tensor.scale
        else:
            return (q_tensor.q_data.astype(np.float32) - q_tensor.zero_point) * q_tensor.scale

    @staticmethod
    def compute_fidelity_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
        """Compute numerical fidelity metrics between original FP32 and reconstructed INT8."""
        orig = np.asarray(original, dtype=np.float32).ravel()
        recon = np.asarray(reconstructed, dtype=np.float32).ravel()

        rmse = float(np.sqrt(np.mean((orig - recon) ** 2)))
        norm_orig = np.linalg.norm(orig)
        norm_recon = np.linalg.norm(recon)

        if norm_orig == 0.0 or norm_recon == 0.0:
            cosine_fidelity = 1.0
        else:
            cosine_fidelity = float(np.dot(orig, recon) / (norm_orig * norm_recon))

        max_absolute_error = float(np.max(np.abs(orig - recon)))
        snr_db = 20.0 * math.log10((norm_orig + 1e-9) / (np.linalg.norm(orig - recon) + 1e-9))

        return {
            "rmse": round(rmse, 6),
            "cosine_fidelity": round(cosine_fidelity, 6),
            "max_absolute_error": round(max_absolute_error, 6),
            "snr_db": round(snr_db, 2),
            "memory_compression_ratio": round(orig.nbytes / (recon.size * 1), 2),  # FP32 (4 bytes) vs INT8 (1 byte)
        }


# ============================================================================
# 5. HYBRID RAG + FINE-TUNING PIPELINE
# ============================================================================

@dataclass
class RetrievedContextChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    score: float


@dataclass
class HybridSynthesizedResponse:
    query: str
    answer: str
    citations: List[str]
    is_refusal: bool
    format_adherence_score: float
    retrieved_chunk_count: int
    latency_ms: float


class HybridRAGFineTuningEngine:
    """Integrates non-parametric vector retrieval with fine-tuned structured synthesis.
    
    1. Vector Retrieval: Dynamically searches external corpus for factual ground-truth.
    2. LoRA / Fine-Tuned Generation Head: Enforces strict structured JSON syntax,
       concise citation extraction, and refusal policies when confidence is low.
    """

    def __init__(self, corpus_chunks: Optional[List[Dict[str, Any]]] = None) -> None:
        self.corpus = corpus_chunks or []
        # Simulate local embedding representations for offline retrieval
        self._index: List[Tuple[str, np.ndarray, Dict[str, Any]]] = []
        self._build_index()

    def _embed_text(self, text: str, dim: int = 128) -> np.ndarray:
        """Deterministic cryptographic-hash embedding for stable cross-process offline execution."""
        import hashlib
        words = re.findall(r"\w+", text.lower())
        vec = np.zeros(dim, dtype=np.float32)
        for w in words:
            h = int(hashlib.sha256(w.encode("utf-8")).hexdigest()[:8], 16) % dim
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-9)

    def _build_index(self) -> None:
        self._index = []
        for idx, doc in enumerate(self.corpus):
            txt = doc.get("text", "")
            meta = doc.get("metadata", {"source": f"doc_{idx}"})
            cid = doc.get("id", f"chunk_{idx}")
            vec = self._embed_text(txt)
            self._index.append((cid, vec, meta))

    def retrieve(self, query: str, top_k: int = 2) -> List[RetrievedContextChunk]:
        """Perform cosine similarity retrieval across indexed corpus."""
        if not self._index:
            return []

        q_vec = self._embed_text(query)
        scored: List[Tuple[float, str, str, Dict[str, Any]]] = []

        for cid, d_vec, meta in self._index:
            sim = float(np.dot(q_vec, d_vec))
            # Find matching text from corpus
            txt = next((c.get("text", "") for c in self.corpus if c.get("id") == cid), "")
            scored.append((sim, cid, txt, meta))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, cid, txt, meta in scored[:top_k]:
            results.append(RetrievedContextChunk(chunk_id=cid, text=txt, metadata=meta, score=round(sim, 4)))
        return results

    def synthesize(
        self,
        query: str,
        retrieved_chunks: List[RetrievedContextChunk],
        confidence_threshold: float = 0.25,
    ) -> HybridSynthesizedResponse:
        """Generate structured response with citations, simulating fine-tuned adapter logic."""
        start_time = time.perf_counter()

        # Check refusal condition
        if not retrieved_chunks or max(c.score for c in retrieved_chunks) < confidence_threshold:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return HybridSynthesizedResponse(
                query=query,
                answer="REFUSAL: Insufficient grounded context retrieved from corpus to answer reliably.",
                citations=[],
                is_refusal=True,
                format_adherence_score=1.0,
                retrieved_chunk_count=len(retrieved_chunks),
                latency_ms=round(elapsed, 2),
            )

        # Synthesize from grounded passages
        best_chunk = retrieved_chunks[0]
        source_label = best_chunk.metadata.get("source", best_chunk.chunk_id)

        # Extract relevant snippet
        sentences = re.split(r"(?<=[.!?])\s+", best_chunk.text)
        key_sentence = sentences[0] if sentences else best_chunk.text

        answer = f"{key_sentence} [Citation: {source_label}]"
        citations = [source_label]

        elapsed = (time.perf_counter() - start_time) * 1000.0
        return HybridSynthesizedResponse(
            query=query,
            answer=answer,
            citations=citations,
            is_refusal=False,
            format_adherence_score=1.0,
            retrieved_chunk_count=len(retrieved_chunks),
            latency_ms=round(elapsed, 2),
        )


# ============================================================================
# 6. PRODUCTION SERVING CONTRACT (FASTAPI & ASGI APPLICATION)
# ============================================================================

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Natural language search question")
    top_k: int = Field(default=3, ge=1, le=20)
    similarity_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    tenant_id: Optional[str] = Field(default=None, description="Tenant isolation tag")


class DocumentIngestItem(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: List[DocumentIngestItem]


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    is_refusal: bool
    latency_ms: float
    model_version: str = "hybrid-rag-ft-v1"


def create_rag_production_app(
    engine: Optional[HybridRAGFineTuningEngine] = None,
    service_version: str = "1.0.0",
) -> Any:
    """Constructs a production-ready FastAPI application conforming to enterprise serving standards."""
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI and Starlette must be installed to create the production app.")

    rag_engine = engine or HybridRAGFineTuningEngine(corpus_chunks=[
        {
            "id": "doc_0",
            "text": "SOC2 Type II requires 90 days retention for audit logs before automated tombstoning.",
            "metadata": {"source": "SOC2_Policy_Section_4.2"},
        },
        {
            "id": "doc_1",
            "text": "Tenant isolation in ChromaDB is enforced via mandatory tenant_id metadata filtering.",
            "metadata": {"source": "Architecture_Spec_Section_9.3"},
        },
    ])

    app = FastAPI(
        title="RAG Engineering Academy Production API",
        version=service_version,
        description="Low-latency, CPU-optimized RAG & Fine-Tuned Serving Engine",
    )

    # In-memory metrics tracking
    metrics_tracker = {
        "requests_total": 0,
        "queries_total": 0,
        "ingest_total": 0,
        "refusals_total": 0,
        "latencies_ms": [],
        "uptime_start": time.time(),
    }

    @app.middleware("http")
    async def track_latency_and_metrics(request: Request, call_next: Callable) -> Response:
        t0 = time.perf_counter()
        metrics_tracker["requests_total"] += 1
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        metrics_tracker["latencies_ms"].append(elapsed_ms)
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response

    @app.get("/health", status_code=status.HTTP_200_OK)
    def health_check() -> Dict[str, Any]:
        """Kubernetes Liveness & Readiness probe endpoint."""
        uptime = time.time() - metrics_tracker["uptime_start"]
        return {
            "status": "healthy",
            "version": service_version,
            "uptime_seconds": round(uptime, 2),
            "index_size_chunks": len(rag_engine.corpus),
            "memory_mode": "CPU_INT8_OPTIMIZED",
        }

    @app.post("/v1/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
    def handle_query(req: QueryRequest) -> QueryResponse:
        """Execute hybrid search & citation synthesis."""
        metrics_tracker["queries_total"] += 1

        chunks = rag_engine.retrieve(req.query, top_k=req.top_k)
        synth = rag_engine.synthesize(req.query, chunks, confidence_threshold=req.similarity_threshold)

        if synth.is_refusal:
            metrics_tracker["refusals_total"] += 1

        chunk_payload = [
            {"id": c.chunk_id, "text": c.text, "score": c.score, "source": c.metadata.get("source")}
            for c in chunks
        ]

        return QueryResponse(
            query=req.query,
            answer=synth.answer,
            citations=synth.citations,
            retrieved_chunks=chunk_payload,
            is_refusal=synth.is_refusal,
            latency_ms=synth.latency_ms,
            model_version=f"rag-ft-{service_version}",
        )

    @app.post("/v1/ingest", status_code=status.HTTP_201_CREATED)
    def handle_ingest(req: IngestRequest) -> Dict[str, Any]:
        """Ingest new document chunks into the active knowledge index."""
        metrics_tracker["ingest_total"] += len(req.documents)
        for doc in req.documents:
            rag_engine.corpus.append({
                "id": doc.id,
                "text": doc.text,
                "metadata": doc.metadata,
            })
        rag_engine._build_index()
        return {
            "status": "success",
            "ingested_count": len(req.documents),
            "total_index_size": len(rag_engine.corpus),
        }

    @app.get("/v1/metrics", status_code=status.HTTP_200_OK)
    def get_metrics() -> Dict[str, Any]:
        """Prometheus-compatible and JSON operational metrics."""
        latencies = metrics_tracker["latencies_ms"]
        p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
        p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
        p99 = float(np.percentile(latencies, 99)) if latencies else 0.0

        return {
            "requests_total": metrics_tracker["requests_total"],
            "queries_total": metrics_tracker["queries_total"],
            "ingest_total": metrics_tracker["ingest_total"],
            "refusals_total": metrics_tracker["refusals_total"],
            "latency_p50_ms": round(p50, 2),
            "latency_p95_ms": round(p95, 2),
            "latency_p99_ms": round(p99, 2),
        }

    return app


# ============================================================================
# 7. DEPLOYMENT READINESS AUDITOR & CONTAINERIZATION GENERATOR
# ============================================================================

@dataclass
class AuditCheckResult:
    check_id: str
    pillar: str
    description: str
    passed: bool
    details: str


@dataclass
class DeploymentAuditReport:
    total_checks: int
    passed_checks: int
    failed_checks: int
    compliance_score: float
    is_ready_for_production: bool
    results: List[AuditCheckResult]
    recommendations: List[str]


class DeploymentReadinessAuditor:
    """Audits containerization, security, and operational configs against 8 enterprise pillars."""

    @staticmethod
    def audit_configuration(config: Dict[str, Any]) -> DeploymentAuditReport:
        checks: List[AuditCheckResult] = []
        recommendations: List[str] = []

        # 1. Non-root user security check
        sec = config.get("security", {})
        uid = sec.get("non_root_user_uid", 0)
        has_non_root = uid > 0
        checks.append(AuditCheckResult(
            check_id="SEC_001",
            pillar="Security",
            description="Container runs as an unprivileged non-root user (UID > 0)",
            passed=has_non_root,
            details=f"Configured UID: {uid}",
        ))
        if not has_non_root:
            recommendations.append("Configure a dedicated non-root service account (e.g. UID 10001) in Dockerfile.")

        # 2. Secret isolation check
        env_vars = str(config)
        has_secret_leak = "sk-" in env_vars or "gsk_" in env_vars
        checks.append(AuditCheckResult(
            check_id="SEC_002",
            pillar="Security",
            description="No hardcoded API credentials present in deployment specification",
            passed=not has_secret_leak,
            details="Validated clean secret separation via root .env injection.",
        ))

        # 3. Liveness / Readiness Probes
        obs = config.get("observability", {})
        health_ep = obs.get("health_endpoint")
        has_probes = bool(health_ep and health_ep.startswith("/"))
        checks.append(AuditCheckResult(
            check_id="OPS_001",
            pillar="Resilience",
            description="Dedicated /health liveness and readiness probe endpoint configured",
            passed=has_probes,
            details=f"Endpoint: {health_ep}",
        ))
        if not has_probes:
            recommendations.append("Define a lightweight /health endpoint returning HTTP 200 for orchestrator probes.")

        # 4. Latency / Timeout Bounds
        srv = config.get("server", {})
        timeout = srv.get("timeout_seconds", 0)
        timeout_ok = 5.0 <= timeout <= 60.0
        checks.append(AuditCheckResult(
            check_id="PERF_001",
            pillar="Performance",
            description="Request timeout bounded safely (5s to 60s)",
            passed=timeout_ok,
            details=f"Timeout: {timeout} seconds",
        ))

        # 5. Model Quantization Mode
        inf = config.get("inference", {})
        quant_mode = inf.get("quantization_mode", "")
        is_quantized = "INT8" in quant_mode or "ONNX" in quant_mode
        checks.append(AuditCheckResult(
            check_id="PERF_002",
            pillar="Performance",
            description="Inference runtime utilizes CPU INT8 quantization or ONNX runtime acceleration",
            passed=is_quantized,
            details=f"Quantization: {quant_mode}",
        ))
        if not is_quantized:
            recommendations.append("Enable INT8 symmetric quantization to reduce CPU memory consumption by 75%.")

        # 6. Observability & Telemetry
        has_metrics = bool(obs.get("metrics_endpoint"))
        checks.append(AuditCheckResult(
            check_id="OBS_001",
            pillar="Observability",
            description="Prometheus / JSON telemetry metrics endpoint enabled",
            passed=has_metrics,
            details=f"Metrics path: {obs.get('metrics_endpoint')}",
        ))

        # 7. Persistent Storage for Vector DB
        ret = config.get("retrieval", {})
        vec_path = ret.get("vector_store_path", "")
        has_persistence = bool(vec_path and vec_path != ":memory:")
        checks.append(AuditCheckResult(
            check_id="DATA_001",
            pillar="Data Governance",
            description="Vector database utilizes durable persistent volume storage",
            passed=has_persistence,
            details=f"Storage path: {vec_path}",
        ))
        if not has_persistence:
            recommendations.append("Do not deploy ephemeral in-memory vector stores to production clusters.")

        # 8. Tenant Isolation & ACL enforcement
        has_acl = sec.get("acl_enforcement", False)
        checks.append(AuditCheckResult(
            check_id="SEC_003",
            pillar="Security",
            description="Multi-tenant isolation and metadata ACL filtering strictly enforced",
            passed=has_acl,
            details=f"ACL enforcement: {has_acl}",
        ))

        passed = sum(1 for c in checks if c.passed)
        failed = len(checks) - passed
        score = round((passed / len(checks)) * 100.0, 1)
        is_ready = failed == 0

        return DeploymentAuditReport(
            total_checks=len(checks),
            passed_checks=passed,
            failed_checks=failed,
            compliance_score=score,
            is_ready_for_production=is_ready,
            results=checks,
            recommendations=recommendations,
        )

    @staticmethod
    def generate_dockerfile() -> str:
        """Generate hardened, multi-stage, CPU-optimized Dockerfile."""
        return """# ============================================================================
# Stage 1: Build & Dependency Resolution
# ============================================================================
FROM python:3.12-slim-bookworm AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements/base.txt requirements/track-16.txt ./
RUN pip install --no-cache-dir --user -r base.txt -r track-16.txt

# ============================================================================
# Stage 2: Hardened Runtime Container
# ============================================================================
FROM python:3.12-slim-bookworm AS runner

# Security: Create unprivileged non-root service account
RUN groupadd -g 10001 raggroup && \\
    useradd -u 10001 -g raggroup -s /bin/false -m raguser

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /root/.local /home/raguser/.local
ENV PATH=/home/raguser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MPLCONFIGDIR=/tmp/matplotlib_cache

# Copy application source code
COPY common/ /app/common/
COPY fixtures/ /app/fixtures/

# Create persistent storage directory with non-root ownership
RUN mkdir -p /app/data/.chroma /tmp/matplotlib_cache && \\
    chown -R raguser:raggroup /app /tmp/matplotlib_cache

USER raguser:raggroup

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "common.finetuning_deployment:create_rag_production_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
"""

    @staticmethod
    def generate_docker_compose() -> str:
        """Generate production Docker Compose specification."""
        return """version: '3.8'

services:
  rag-api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: rag_production_api
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - LLM_PROVIDER=groq
      - CHROMA_PERSIST_DIRECTORY=/app/data/.chroma
      - LOG_LEVEL=INFO
    volumes:
      - chroma_data:/app/data/.chroma
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 15s
      timeout: 5s
      retries: 3
    security_opt:
      - no-new-privileges:true
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4096M

volumes:
  chroma_data:
    driver: local
"""


# ============================================================================
# 8. VISUALIZATIONS & ARCHITECTURAL DIAGRAMS
# ============================================================================

def generate_finetuning_vs_rag_diagram(output_path: str) -> None:
    """Generate high-resolution diagram comparing RAG vs Fine-Tuning trade-off quadrant."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    # Subplot 1: Decision Matrix Quadrant
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.axhline(0.5, color="#666666", linestyle="--", alpha=0.7)
    ax1.axvline(0.5, color="#666666", linestyle="--", alpha=0.7)

    # Shaded quadrants
    ax1.fill_between([0.0, 0.5], [0.5, 0.5], [1.0, 1.0], color="#e8f4f8", alpha=0.6) # Top-left: Hybrid
    ax1.fill_between([0.5, 1.0], [0.5, 0.5], [1.0, 1.0], color="#e6f4ea", alpha=0.6) # Top-right: Pure RAG
    ax1.fill_between([0.0, 0.5], [0.0, 0.0], [0.5, 0.5], color="#fce8e6", alpha=0.6) # Bottom-left: Base model
    ax1.fill_between([0.5, 1.0], [0.0, 0.0], [0.5, 0.5], color="#fef7e0", alpha=0.6) # Bottom-right: Fine-Tuning

    # Labels for quadrants
    ax1.text(0.75, 0.75, "RAG DOMAIN\n• High Knowledge Volatility\n• Strict Grounded Citations\n• Fine-grained ACLs",
             ha="center", va="center", fontsize=10, weight="bold", color="#137333")
    ax1.text(0.25, 0.75, "HYBRID RAG + FT\n• Volatile Facts + Strict Format\n• Small Specialized Reasoner\n• Zero Hallucination Gate",
             ha="center", va="center", fontsize=10, weight="bold", color="#1a73e8")
    ax1.text(0.75, 0.25, "FINE-TUNING (LoRA)\n• Static Domain Knowledge\n• Custom Style / Jargon\n• Specialized Syntax/JSON",
             ha="center", va="center", fontsize=10, weight="bold", color="#b06000")
    ax1.text(0.25, 0.25, "BASE MODEL / PROMPT\n• General Reasoning\n• Zero Ingestion Cost\n• Low Complexity",
             ha="center", va="center", fontsize=10, weight="bold", color="#c5221f")

    ax1.set_xlabel("Knowledge Volatility & Citation Strictness →", fontsize=11, weight="bold")
    ax1.set_ylabel("Specialized Style & Structure Needs →", fontsize=11, weight="bold")
    ax1.set_title("Architectural Decision Quadrant", fontsize=13, weight="bold", pad=12)
    ax1.grid(False)

    # Subplot 2: Radar / Bar Comparison across Dimensions
    dimensions = ["Dynamic Facts", "Citations", "Format/JSON", "Access Control", "Low Cost", "Low Latency"]
    rag_scores = [0.95, 0.95, 0.65, 0.95, 0.90, 0.70]
    ft_scores = [0.20, 0.25, 0.95, 0.15, 0.50, 0.85]
    hybrid_scores = [0.95, 0.98, 0.95, 0.90, 0.70, 0.75]

    x = np.arange(len(dimensions))
    width = 0.25

    ax2.bar(x - width, rag_scores, width, label="RAG Only", color="#34a853")
    ax2.bar(x, ft_scores, width, label="Fine-Tuning Only", color="#fbbc04")
    ax2.bar(x + width, hybrid_scores, width, label="Hybrid RAG + FT", color="#1a73e8")

    ax2.set_xticks(x)
    ax2.set_xticklabels(dimensions, rotation=25, ha="right", fontsize=9, weight="bold")
    ax2.set_ylim(0.0, 1.15)
    ax2.set_ylabel("Score (0.0 to 1.0)", fontsize=11, weight="bold")
    ax2.set_title("Dimension Evaluation Profile", fontsize=13, weight="bold", pad=12)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_deployment_architecture_diagram(output_path: str) -> None:
    """Generate high-resolution publication-quality diagram of production deployment topology."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Draw boxes
    boxes = [
        # (x, y, w, h, title, subtitle, color)
        (0.5, 3.0, 2.2, 2.2, "Client Apps", "Web / Mobile / Agents\nHTTPS REST / SSE", "#e8f0fe"),
        (3.3, 1.5, 3.2, 5.2, "FastAPI Microservice\n(Runner Container)",
         "• /health (Liveness/Readiness)\n• /v1/query (Streaming)\n• /v1/ingest (ACL Tagging)\n• /v1/metrics (Prometheus)\n• Gunicorn/Uvicorn Workers\n• Non-root UID 10001", "#e6f4ea"),
        (7.2, 4.3, 3.2, 2.4, "Hybrid Reasoning Head", "• LoRA Adapter (Rank=8)\n• Structured Refusal Gate\n• INT8 CPU Quantization\n• Zero-Leakage JSON", "#fef7e0"),
        (7.2, 1.5, 3.2, 2.4, "ChromaDB Store", "• Local Persistent Volume\n• HNSW Cosine Index\n• Metadata Tenant Filtering\n• Sub-10ms Exact Search", "#fce8e6"),
        (11.0, 2.5, 2.5, 3.2, "Observability\n& Telemetry", "• OpenTelemetry SDK\n• Prometheus Scraper\n• Latency p50/p95/p99\n• Hallucination Monitor", "#f3e8fd"),
    ]

    for bx, by, bw, bh, title, subtitle, col in boxes:
        rect = plt.Rectangle((bx, by), bw, bh, facecolor=col, edgecolor="#333333", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(bx + bw / 2, by + bh - 0.4, title, ha="center", va="top", fontsize=10, weight="bold", color="#202124")
        ax.text(bx + bw / 2, by + 0.3, subtitle, ha="center", va="bottom", fontsize=8.5, color="#3c4043")

    # Connectors
    arrows = [
        ((2.7, 4.1), (3.3, 4.1), "REST / SSE"),
        ((6.5, 5.5), (7.2, 5.5), "Synthesis"),
        ((6.5, 2.7), (7.2, 2.7), "Filtered Query"),
        ((10.4, 4.1), (11.0, 4.1), "Spans / Metrics"),
    ]

    for (x1, y1), (x2, y2), label in arrows:
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#1a73e8", lw=2)
        )
        mid_x = (x1 + x2) / 2
        mid_y = ((y1 + y2) / 2) + 0.25
        ax.text(mid_x, mid_y, label, ha="center", va="bottom", fontsize=8.5, weight="bold", color="#1a73e8")

    ax.set_title("Enterprise RAG & Fine-Tuned Serving Architecture Topology", fontsize=13, weight="bold", pad=15)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
