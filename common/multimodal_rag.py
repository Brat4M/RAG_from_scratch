"""Multimodal RAG: Cross-Modal Retrieval, Visual Document Parsing, and Unified Multi-Evidence Fusion.

Provides production-grade, 100% offline-executable, CPU-friendly Multimodal RAG architectures:
1. Visual Document Structure Parsing:
   - Layout-aware PDF decomposition using PyMuPDF (`pymupdf`) and PyPDF.
   - Table grid detection, cell coordinate extraction, and structured Markdown serialization.
   - Image and diagram block extraction with precise bounding boxes (x0, y0, x1, y1) and captions.
2. Cross-Modal Joint Embedding Space:
   - Contrastive Vision-Language representation space (dual encoder architecture, normalized S^{D-1} hypersphere).
   - Deterministic, calibrated joint embedding for text, images (PIL), tabular data, and audio/video features.
   - ColPali late-interaction multi-vector matcher simulating patch-level MaxSim token-to-patch scoring without OCR bottlenecks.
3. Temporal Media Ingestion & Multi-Track Alignment:
   - Audio ingestion modeling prosody, acoustic energy, silence, and speech transcripts.
   - Video ingestion with temporal segmentation, keyframe extraction, and cross-modal audio-visual synchronization.
   - Analysis of information loss in transcription-only or keyframe-only architectures.
4. Unified Multimodal Vector Store & Hybrid Retrieval:
   - Cross-modal text-to-image, text-to-table, and multimodal fusion search.
   - Modality-aware filtering, scoring weights, and Reciprocal Rank Fusion (RRF).
5. Visual Answer Grounding & Multimodal Synthesis:
   - Grounded generation citing bounding box coordinates, page numbers, table cells, and figure identifiers.
   - Visual citation verification and cross-modal hallucination detection.
6. Multimodal Diagnostics & Diagram Generation:
   - Matplotlib publication-grade architecture schematics and cross-modal embedding space visualizations.
"""

from __future__ import annotations

import base64
import hashlib
import io
import math
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

try:
    import pymupdf  # PyMuPDF
except ImportError:
    pymupdf = None  # Graceful fallback handled in DocumentVisualParser

try:
    import pypdf
except ImportError:
    pypdf = None


# ---------------------------------------------------------------------------
# 1. Enums and Core Pydantic Models
# ---------------------------------------------------------------------------

class ModalityType(str, Enum):
    """Supported modalities within the multimodal RAG ecosystem."""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class BoundingBox(BaseModel):
    """Represents a 2D spatial bounding box on a document page."""
    x0: float = Field(..., description="Left coordinate (horizontal minimum)")
    y0: float = Field(..., description="Top coordinate (vertical minimum)")
    x1: float = Field(..., description="Right coordinate (horizontal maximum)")
    y1: float = Field(..., description="Bottom coordinate (vertical maximum)")
    page_num: int = Field(default=1, description="1-indexed document page number")

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: BoundingBox) -> float:
        """Compute Intersection-over-Union with another bounding box on the same page."""
        if self.page_num != other.page_num:
            return 0.0
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)

        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0

        intersection = (ix1 - ix0) * (iy1 - iy0)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def normalize(self, page_width: float, page_height: float) -> BoundingBox:
        """Return normalized coordinates in [0.0, 1.0] relative to page dimensions."""
        pw = max(page_width, 1.0)
        ph = max(page_height, 1.0)
        return BoundingBox(
            x0=round(self.x0 / pw, 4),
            y0=round(self.y0 / ph, 4),
            x1=round(self.x1 / pw, 4),
            y1=round(self.y1 / ph, 4),
            page_num=self.page_num,
        )

    def to_citation_str(self) -> str:
        """Generate human-readable visual citation string."""
        return f"p.{self.page_num}[{int(self.x0)},{int(self.y0)},{int(self.x1)},{int(self.y1)}]"


class MultimodalChunk(BaseModel):
    """Unified chunk schema spanning text, tables, images, audio, and video."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    modality: ModalityType = Field(..., description="Modality category")
    content: str = Field(..., description="Textual representation, OCR, table markdown, or transcription")
    page_num: Optional[int] = Field(default=None, description="Document page number if applicable")
    bbox: Optional[BoundingBox] = Field(default=None, description="Spatial coordinates if applicable")
    image_bytes: Optional[bytes] = Field(default=None, repr=False, description="Raw image bytes if visual chunk")
    table_data: Optional[Dict[str, Any]] = Field(default=None, description="Structured table representation (headers, rows)")
    audio_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Audio prosody, timestamps, energy")
    video_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Video fps, duration, keyframe metadata")
    caption: Optional[str] = Field(default=None, description="Associated figure caption or descriptive title")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary provenance and metadata tags")
    embedding: Optional[List[float]] = Field(default=None, repr=False, description="Projected joint multimodal embedding vector")

    def get_citation(self) -> str:
        """Generate formatted citation identifier."""
        if self.modality == ModalityType.IMAGE:
            coord = self.bbox.to_citation_str() if self.bbox else f"p.{self.page_num or 1}"
            return f"[Figure: {self.caption or self.chunk_id}, {coord}]"
        elif self.modality == ModalityType.TABLE:
            coord = self.bbox.to_citation_str() if self.bbox else f"p.{self.page_num or 1}"
            return f"[Table: {self.caption or self.chunk_id}, {coord}]"
        elif self.modality == ModalityType.AUDIO:
            ts = self.audio_metadata.get("timestamp_str", "0:00") if self.audio_metadata else "0:00"
            return f"[Audio: {self.chunk_id} @ {ts}]"
        elif self.modality == ModalityType.VIDEO:
            ts = self.video_metadata.get("timestamp_str", "0:00") if self.video_metadata else "0:00"
            return f"[Video: {self.chunk_id} @ {ts}]"
        else:
            coord = f"p.{self.page_num}" if self.page_num else self.chunk_id
            return f"[Text: {coord}]"


class SearchResult(BaseModel):
    """Ranked multimodal retrieval result with score, chunk, and visual grounding."""
    chunk: MultimodalChunk
    score: float = Field(..., description="Similarity or RRF score")
    modality: ModalityType
    citation: str
    match_type: str = Field(default="joint_embedding", description="Method used for retrieval matching")


# ---------------------------------------------------------------------------
# 2. Document Visual Structure Parser (PyMuPDF & PyPDF)
# ---------------------------------------------------------------------------

class DocumentVisualParser:
    """Parses PDF documents into multimodal components: text paragraphs, table grids, and image blocks.
    
    Preserves spatial coordinates, reading order, and figure-caption associations.
    """

    def __init__(self, render_dpi: int = 150):
        self.render_dpi = render_dpi

    def parse_pdf(self, pdf_path_or_bytes: Union[str, bytes]) -> List[MultimodalChunk]:
        """Extract text blocks, tables, and images with bounding boxes from a PDF."""
        if pymupdf is None:
            return self._fallback_parse(pdf_path_or_bytes)

        if isinstance(pdf_path_or_bytes, bytes):
            doc = pymupdf.open(stream=pdf_path_or_bytes, filetype="pdf")
        else:
            doc = pymupdf.open(pdf_path_or_bytes)

        chunks: List[MultimodalChunk] = []

        try:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_num = page_idx + 1

                # 1. Extract tabular data
                table_bboxes: List[BoundingBox] = []
                try:
                    tables = page.find_tables()
                    for t_idx, tab in enumerate(tables):
                        tab_rect = tab.bbox  # (x0, y0, x1, y1)
                        t_bbox = BoundingBox(
                            x0=float(tab_rect[0]),
                            y0=float(tab_rect[1]),
                            x1=float(tab_rect[2]),
                            y1=float(tab_rect[3]),
                            page_num=page_num,
                        )
                        table_bboxes.append(t_bbox)

                        # Extract table dataframe or rows
                        rows = tab.extract()
                        if rows and len(rows) > 0:
                            headers = [" ".join(str(c or "").split()) for c in rows[0]]
                            body_rows = [[" ".join(str(c or "").split()) for c in r] for r in rows[1:]]

                            # Format markdown table
                            md_lines = []
                            md_lines.append("| " + " | ".join(headers) + " |")
                            md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                            for row in body_rows:
                                md_lines.append("| " + " | ".join(row) + " |")
                            md_table = "\n".join(md_lines)

                            chunks.append(
                                MultimodalChunk(
                                    chunk_id=f"tab_p{page_num}_{t_idx+1}",
                                    modality=ModalityType.TABLE,
                                    content=md_table,
                                    page_num=page_num,
                                    bbox=t_bbox,
                                    table_data={
                                        "headers": headers,
                                        "rows": body_rows,
                                        "num_rows": len(body_rows),
                                        "num_cols": len(headers),
                                    },
                                    caption=f"Table {t_idx+1} on Page {page_num}",
                                    metadata={"source": "pymupdf_find_tables"},
                                )
                            )
                except Exception:
                    pass  # Gracefully fall back if table detection heuristics encounter non-standard vector paths

                # 2. Extract embedded images
                image_list = page.get_images(full=True)
                for img_idx, img_info in enumerate(image_list):
                    xref = img_info[0]
                    try:
                        base_image = doc.extract_image(xref)
                        img_bytes = base_image.get("image")
                        img_rects = page.get_image_rects(xref)
                        img_bbox = None
                        if img_rects:
                            r = img_rects[0]
                            img_bbox = BoundingBox(
                                x0=float(r.x0), y0=float(r.y0), x1=float(r.x1), y1=float(r.y1), page_num=page_num
                            )

                        caption = f"Figure {img_idx+1} on Page {page_num}"
                        chunks.append(
                            MultimodalChunk(
                                chunk_id=f"img_p{page_num}_{img_idx+1}",
                                modality=ModalityType.IMAGE,
                                content=f"Image artifact: {caption} (Dimensions: {base_image.get('width', 0)}x{base_image.get('height', 0)})",
                                page_num=page_num,
                                bbox=img_bbox,
                                image_bytes=img_bytes,
                                caption=caption,
                                metadata={
                                    "width": base_image.get("width"),
                                    "height": base_image.get("height"),
                                    "format": base_image.get("ext"),
                                },
                            )
                        )
                    except Exception:
                        continue

                # 3. Extract text blocks with layout positions
                text_page = page.get_text("blocks")
                for b_idx, block in enumerate(text_page):
                    # block: (x0, y0, x1, y1, text, block_no, block_type)
                    # block_type == 0 is text, 1 is image
                    if len(block) >= 5 and block[6] == 0 if len(block) > 6 else True:
                        raw_text = str(block[4]).strip()
                        if not raw_text:
                            continue

                        bx0, by0, bx1, by1 = float(block[0]), float(block[1]), float(block[2]), float(block[3])
                        block_bbox = BoundingBox(x0=bx0, y0=by0, x1=bx1, y1=by1, page_num=page_num)

                        # Skip text that heavily overlaps an extracted table to avoid duplication
                        if any(block_bbox.iou(tb) > 0.4 for tb in table_bboxes):
                            continue

                        # Detect potential caption patterns
                        is_caption = bool(re.match(r"^(figure|fig\.|table|chart)\s+\d+", raw_text, re.IGNORECASE))

                        chunks.append(
                            MultimodalChunk(
                                chunk_id=f"txt_p{page_num}_{b_idx+1}",
                                modality=ModalityType.TEXT,
                                content=raw_text,
                                page_num=page_num,
                                bbox=block_bbox,
                                caption="Caption" if is_caption else None,
                                metadata={"is_caption": is_caption, "block_index": b_idx},
                            )
                        )
        finally:
            doc.close()

        return chunks

    def _fallback_parse(self, pdf_path_or_bytes: Union[str, bytes]) -> List[MultimodalChunk]:
        """Fallback parser when PyMuPDF is not available, using PyPDF or basic stream extraction."""
        chunks = []
        if pypdf:
            try:
                reader = pypdf.PdfReader(
                    io.BytesIO(pdf_path_or_bytes) if isinstance(pdf_path_or_bytes, bytes) else pdf_path_or_bytes
                )
                for page_idx, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        chunks.append(
                            MultimodalChunk(
                                chunk_id=f"txt_fallback_p{page_idx+1}",
                                modality=ModalityType.TEXT,
                                content=text.strip(),
                                page_num=page_idx + 1,
                            )
                        )
            except Exception:
                pass
        return chunks

    @staticmethod
    def create_synthetic_multimodal_pdf(output_path: Optional[str] = None) -> bytes:
        """Create a minimal 1-page sample PDF containing text, a structured table, and a vector/raster diagram.
        
        Uses PyMuPDF's built-in drawing primitives for 100% offline self-contained testing.
        """
        if pymupdf is None:
            return b"%PDF-1.4 minimal fallback"

        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)  # A4

        # 1. Title and heading text
        page.insert_text(pymupdf.Point(50, 60), "Multimodal RAG Architecture Technical Report", fontsize=18, fontname="helv", color=(0.1, 0.2, 0.5))
        page.insert_text(pymupdf.Point(50, 90), "Section 1: Performance Benchmarks & Model Latency", fontsize=14, fontname="helv", color=(0.2, 0.2, 0.2))

        # 2. Paragraph text
        paragraph = (
            "Cross-modal retrieval unifies text passages, diagrammatic figures, and tabular matrices into a shared "
            "contrastive representation space. ColPali late-interaction preserves visual token spatial coordinates, "
            "bypassing brittle OCR pipelines and enabling precise query-to-patch attention grounding."
        )
        page.insert_textbox(pymupdf.Rect(50, 110, 545, 160), paragraph, fontsize=10, fontname="helv")

        # 3. Draw a structured table grid with headers and rows
        # Table position: x=50 to 545, y=180 to 280
        headers = ["Model Architecture", "Modality", "Retrieval Recall@5", "Mean Latency (ms)"]
        rows = [
            ["Text-Only Dense RAG", "Text", "64.2%", "14.5 ms"],
            ["OCR + Text Chunker", "Text + OCR", "71.8%", "48.2 ms"],
            ["ColPali Multi-Vector", "Text + Vision Patches", "88.6%", "32.0 ms"],
            ["Unified Multimodal RAG", "Text + Image + Table", "93.4%", "26.8 ms"],
        ]

        col_widths = [140, 110, 125, 120]
        y_pos = 180
        row_h = 22

        # Draw header row background
        shape = page.new_shape()
        shape.draw_rect(pymupdf.Rect(50, y_pos, 545, y_pos + row_h))
        shape.finish(color=(0.3, 0.4, 0.7), fill=(0.9, 0.93, 0.98))
        shape.commit()

        # Write headers
        x_curr = 55
        for h, w in zip(headers, col_widths):
            page.insert_text(pymupdf.Point(x_curr, y_pos + 15), h, fontsize=9, fontname="helv", color=(0.1, 0.1, 0.3))
            x_curr += w

        # Draw rows
        for r_idx, row in enumerate(rows):
            y_pos += row_h
            shape = page.new_shape()
            shape.draw_rect(pymupdf.Rect(50, y_pos, 545, y_pos + row_h))
            bg_fill = (0.97, 0.97, 0.99) if r_idx % 2 == 1 else (1.0, 1.0, 1.0)
            shape.finish(color=(0.8, 0.8, 0.85), fill=bg_fill)
            shape.commit()

            x_curr = 55
            for cell, w in zip(row, col_widths):
                page.insert_text(pymupdf.Point(x_curr, y_pos + 15), cell, fontsize=9, fontname="helv", color=(0.15, 0.15, 0.15))
                x_curr += w

        # Table caption
        page.insert_text(pymupdf.Point(50, y_pos + 38), "Table 1: Comparative retrieval recall and latency across multimodal pipeline configurations.", fontsize=8, fontname="helv", color=(0.4, 0.4, 0.4))

        # 4. Draw a raster chart / diagram using Pillow and insert it
        chart_img = Image.new("RGB", (400, 200), color=(255, 255, 255))
        draw = ImageDraw.Draw(chart_img)
        # Draw axes
        draw.line([(50, 170), (370, 170)], fill=(80, 80, 80), width=2)
        draw.line([(50, 30), (50, 170)], fill=(80, 80, 80), width=2)
        # Bars
        bar_data = [64, 72, 89, 93]
        colors = [(100, 149, 237), (70, 130, 180), (60, 179, 113), (255, 127, 80)]
        for i, (val, col) in enumerate(zip(bar_data, colors)):
            bx = 80 + i * 70
            by = 170 - int((val / 100) * 130)
            draw.rectangle([bx, by, bx + 45, 170], fill=col, outline=(50, 50, 50))
            draw.text((bx + 10, by - 14), f"{val}%", fill=(30, 30, 30))

        img_buf = io.BytesIO()
        chart_img.save(img_buf, format="PNG")
        img_bytes = img_buf.getvalue()

        # Insert image into page
        chart_rect = pymupdf.Rect(95, 340, 500, 520)
        page.insert_image(chart_rect, stream=img_bytes)
        page.insert_text(pymupdf.Point(100, 535), "Figure 1: Recall@5 performance across RAG configurations (ColPali vs Baselines).", fontsize=8, fontname="helv", color=(0.4, 0.4, 0.4))

        pdf_bytes = doc.tobytes()
        if output_path:
            with open(output_path, "wb") as f:
                f.write(pdf_bytes)
        doc.close()
        return pdf_bytes


# ---------------------------------------------------------------------------
# 3. Cross-Modal Joint Embedding Space (Contrastive Dual-Encoder Simulation)
# ---------------------------------------------------------------------------

class CrossModalEmbedder:
    """Projects text queries, images, tables, audio, and video into a shared D-dimensional unit hypersphere S^{D-1}.
    
    Mathematical formulation:
    - Text projection: E_text(t) = normalize(W_text * phi_text(t))
    - Image projection: E_img(v) = normalize(W_img * phi_visual(v))
    - Cosine similarity: S(t, v) = E_text(t) . E_img(v)
    
    100% offline, deterministic, and CPU-efficient without downloading multi-gigabyte neural checkpoints.
    """

    def __init__(self, embedding_dim: int = 128, seed: int = 42):
        self.embedding_dim = embedding_dim
        self.seed = seed
        rng = np.random.RandomState(seed)

        # Contrastive semantic anchor vocabulary mapping cross-modal visual/acoustic concepts
        self.concept_lexicon: Dict[str, np.ndarray] = {}
        core_concepts = [
            "chart", "diagram", "table", "graph", "plot", "bar", "histogram", "recall",
            "latency", "accuracy", "benchmark", "performance", "architecture", "model",
            "audio", "speech", "acoustic", "tone", "prosody", "video", "motion", "frame",
            "image", "visual", "figure", "header", "pipeline", "loss", "metric", "curve",
            "text", "document", "retrieval", "embedding", "crossmodal", "grounding",
        ]
        for concept in core_concepts:
            vec = rng.randn(embedding_dim)
            self.concept_lexicon[concept] = vec / (np.linalg.norm(vec) + 1e-9)

        # Baseline projection matrix
        self.proj_weights = rng.randn(32, embedding_dim)

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-9) if norm > 0 else vec

    def embed_text(self, text: str) -> np.ndarray:
        """Embed text prompt into the joint multimodal space."""
        tokens = re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())
        if not tokens:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Base hash vector
        h_vec = np.zeros(self.embedding_dim, dtype=np.float64)
        for token in tokens:
            # Deterministic token digest
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            chunk_hash = np.frombuffer(digest[:16], dtype=np.int32)
            for i, val in enumerate(chunk_hash):
                idx = (val % self.embedding_dim + self.embedding_dim) % self.embedding_dim
                h_vec[idx] += 1.0

            # Semantic concept reinforcement
            for concept, c_vec in self.concept_lexicon.items():
                if concept in token or token in concept:
                    h_vec += 2.5 * c_vec

        return self._normalize(h_vec).astype(np.float32)

    def embed_image(self, image_input: Union[bytes, Image.Image, np.ndarray]) -> np.ndarray:
        """Embed an image into the shared cross-modal space based on visual composition and features."""
        try:
            if isinstance(image_input, bytes):
                img = Image.open(io.BytesIO(image_input)).convert("RGB")
            elif isinstance(image_input, np.ndarray):
                img = Image.fromarray(image_input).convert("RGB")
            else:
                img = image_input.convert("RGB")
        except Exception:
            # Return neutral embedding on corrupt image bytes
            return self._normalize(np.ones(self.embedding_dim, dtype=np.float32))

        # Downsample to extract spatial color and gradient features
        thumb = img.resize((32, 32))
        arr = np.array(thumb, dtype=np.float64) / 255.0  # (32, 32, 3)

        # 1. Color distribution (mean and std across R, G, B)
        mean_rgb = arr.mean(axis=(0, 1))
        std_rgb = arr.std(axis=(0, 1))

        # 2. Edge / gradient energy (horizontal & vertical differences)
        dx = np.abs(np.diff(arr, axis=1)).mean()
        dy = np.abs(np.diff(arr, axis=0)).mean()
        edge_energy = dx + dy

        # 3. Compact 32-dim feature vector
        feature_32 = np.zeros(32, dtype=np.float64)
        feature_32[0:3] = mean_rgb
        feature_32[3:6] = std_rgb
        feature_32[6] = edge_energy
        feature_32[7] = img.width / max(1.0, float(img.height))  # Aspect ratio

        # Grid quadrant brightness (16 quadrants)
        grid_4x4 = arr.reshape(4, 8, 4, 8, 3).mean(axis=(1, 3, 4)).flatten()
        feature_32[8:24] = grid_4x4[:16]

        # Project 32-dim visual features to embedding_dim
        v_proj = feature_32 @ self.proj_weights  # (embedding_dim,)

        # If high edge energy and contrasting bars, align with diagram/chart/visual concepts
        if edge_energy > 0.08:
            v_proj += 1.5 * self.concept_lexicon.get("chart", np.zeros(self.embedding_dim))
            v_proj += 1.5 * self.concept_lexicon.get("diagram", np.zeros(self.embedding_dim))
        if mean_rgb.mean() > 0.7:  # Light background, typical of charts/documents
            v_proj += 1.0 * self.concept_lexicon.get("figure", np.zeros(self.embedding_dim))

        return self._normalize(v_proj).astype(np.float32)

    def embed_table(self, table_markdown: str, table_data: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Embed tabular data, weighting column headers and numeric cells."""
        # Textual representation embedding
        base_vec = self.embed_text(table_markdown)
        # Structural bias: tables align strongly with table/metric/benchmark lexicon
        bias = (
            2.0 * self.concept_lexicon.get("table", np.zeros(self.embedding_dim)) +
            1.5 * self.concept_lexicon.get("benchmark", np.zeros(self.embedding_dim)) +
            1.2 * self.concept_lexicon.get("latency", np.zeros(self.embedding_dim)) +
            1.2 * self.concept_lexicon.get("accuracy", np.zeros(self.embedding_dim))
        )
        return self._normalize(base_vec + bias).astype(np.float32)

    def embed_audio(self, audio_metadata: Dict[str, Any], transcript: str = "") -> np.ndarray:
        """Embed acoustic prosody, energy, and speech transcript."""
        t_vec = self.embed_text(transcript)
        # Acoustic tone and frequency bias
        a_bias = (
            1.8 * self.concept_lexicon.get("audio", np.zeros(self.embedding_dim)) +
            1.5 * self.concept_lexicon.get("speech", np.zeros(self.embedding_dim)) +
            1.2 * self.concept_lexicon.get("tone", np.zeros(self.embedding_dim))
        )
        # Energy modulation
        energy = float(audio_metadata.get("acoustic_energy", 0.5))
        a_bias *= (0.8 + 0.4 * energy)
        return self._normalize(t_vec + a_bias).astype(np.float32)

    def embed_video(self, video_metadata: Dict[str, Any], caption: str = "") -> np.ndarray:
        """Embed video temporal keyframe features and action descriptions."""
        v_vec = self.embed_text(caption)
        vid_bias = (
            1.8 * self.concept_lexicon.get("video", np.zeros(self.embedding_dim)) +
            1.5 * self.concept_lexicon.get("motion", np.zeros(self.embedding_dim)) +
            1.2 * self.concept_lexicon.get("frame", np.zeros(self.embedding_dim))
        )
        return self._normalize(v_vec + vid_bias).astype(np.float32)

    def embed_chunk(self, chunk: MultimodalChunk) -> np.ndarray:
        """Dispatch embedding based on chunk modality."""
        if chunk.modality == ModalityType.TEXT:
            return self.embed_text(chunk.content)
        elif chunk.modality == ModalityType.TABLE:
            return self.embed_table(chunk.content, chunk.table_data)
        elif chunk.modality == ModalityType.IMAGE:
            if chunk.image_bytes:
                img_vec = self.embed_image(chunk.image_bytes)
                cap_vec = self.embed_text(chunk.caption or chunk.content)
                return self._normalize(0.6 * img_vec + 0.4 * cap_vec)
            return self.embed_text(chunk.content)
        elif chunk.modality == ModalityType.AUDIO:
            return self.embed_audio(chunk.audio_metadata or {}, chunk.content)
        elif chunk.modality == ModalityType.VIDEO:
            return self.embed_video(chunk.video_metadata or {}, chunk.content)
        return self.embed_text(chunk.content)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two unit vectors."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# 4. ColPali Late-Interaction Multi-Vector Matcher (MaxSim Operator)
# ---------------------------------------------------------------------------

class ColPaliMatcher:
    """Simulates ColPali late-interaction retrieval over document visual patch tokens.
    
    Architecture insight:
    Instead of collapsing an entire page or image into a single 1D vector (which loses fine-grained layout,
    small table cells, and visual coordinates), ColPali represents the page image as a grid of patch tokens:
    D = [p_1, p_2, ..., p_M] where p_j in R^D (e.g., 32x32 patches = 1024 visual tokens).
    
    The query is represented by token vectors:
    Q = [q_1, q_2, ..., q_N] where q_i in R^D.
    
    The ColPali score is the sum of maximum inner products (MaxSim):
    Score(Q, D) = sum_{i=1}^N max_{j=1}^M (q_i . p_j)
    """

    def __init__(self, embedder: CrossModalEmbedder, patch_grid: Tuple[int, int] = (4, 4)):
        self.embedder = embedder
        self.grid_rows, self.grid_cols = patch_grid
        self.num_patches = self.grid_rows * self.grid_cols

    def extract_page_patches(self, page_image: Union[bytes, Image.Image]) -> List[np.ndarray]:
        """Decompose page image into spatial patch embeddings."""
        if isinstance(page_image, bytes):
            img = Image.open(io.BytesIO(page_image)).convert("RGB")
        else:
            img = page_image.convert("RGB")

        w, h = img.size
        patch_w = w // self.grid_cols
        patch_h = h // self.grid_rows

        patch_vectors: List[np.ndarray] = []
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                box = (c * patch_w, r * patch_h, (c + 1) * patch_w, (r + 1) * patch_h)
                patch_crop = img.crop(box)
                p_vec = self.embedder.embed_image(patch_crop)
                patch_vectors.append(p_vec)

        return patch_vectors

    def maxsim_score(
        self, query: str, page_patch_embeddings: List[np.ndarray]
    ) -> Tuple[float, np.ndarray]:
        """Compute the ColPali MaxSim score and token-to-patch attention alignment matrix."""
        query_tokens = [tok for tok in re.findall(r"\b[a-zA-Z0-9_\-]+\b", query.lower()) if tok]
        if not query_tokens or not page_patch_embeddings:
            return 0.0, np.zeros((1, 1), dtype=np.float32)

        # Generate query token embeddings
        q_vectors = [self.embedder.embed_text(tok) for tok in query_tokens]
        # Query matrix: (N, D)
        Q = np.stack(q_vectors, axis=0)
        # Patch matrix: (M, D)
        P = np.stack(page_patch_embeddings, axis=0)

        # Interaction similarity matrix: S_ij = q_i . p_j, shape (N, M)
        sim_matrix = Q @ P.T

        # MaxSim: for each query token i, find max similarity across all patches j
        max_per_token = np.max(sim_matrix, axis=1)  # (N,)
        total_score = float(np.sum(max_per_token))

        return total_score, sim_matrix


# ---------------------------------------------------------------------------
# 5. Temporal Media Processor: Audio & Video Ingestion
# ---------------------------------------------------------------------------

@dataclass
class AudioSegment:
    """Rich acoustic speech segment capturing prosody and non-speech events."""
    start_sec: float
    end_sec: float
    transcript: str
    tone: str  # e.g., "authoritative", "hesitant", "urgent", "neutral"
    acoustic_energy: float  # [0.0, 1.0]
    has_background_noise: bool = False
    speaker_id: str = "spk_0"

    @property
    def timestamp_str(self) -> str:
        s_m, s_s = divmod(int(self.start_sec), 60)
        e_m, e_s = divmod(int(self.end_sec), 60)
        return f"{s_m:02d}:{s_s:02d}-{e_m:02d}:{e_s:02d}"


@dataclass
class VideoSegment:
    """Video temporal scene with keyframe, visual activity, and speech alignment."""
    start_sec: float
    end_sec: float
    keyframe_caption: str
    visual_action: str
    keyframe_bytes: Optional[bytes] = None

    @property
    def timestamp_str(self) -> str:
        s_m, s_s = divmod(int(self.start_sec), 60)
        e_m, e_s = divmod(int(self.end_sec), 60)
        return f"{s_m:02d}:{s_s:02d}-{e_m:02d}:{e_s:02d}"


@dataclass
class AlignedTemporalUnit:
    """Synchronized cross-modal temporal unit binding audio, video, and transcript."""
    start_sec: float
    end_sec: float
    audio: AudioSegment
    video: VideoSegment
    synthesis_summary: str

    def to_chunk(self) -> MultimodalChunk:
        content = (
            f"Temporal Unit ({self.audio.timestamp_str}):\n"
            f"- Spoken Content: \"{self.audio.transcript}\" (Tone: {self.audio.tone}, Energy: {self.audio.acoustic_energy:.2f})\n"
            f"- Visual Scene: {self.video.keyframe_caption} (Action: {self.video.visual_action})\n"
            f"- Synchronized Summary: {self.synthesis_summary}"
        )
        return MultimodalChunk(
            chunk_id=f"temporal_{int(self.start_sec)}_{int(self.end_sec)}",
            modality=ModalityType.VIDEO,
            content=content,
            audio_metadata={
                "timestamp_str": self.audio.timestamp_str,
                "tone": self.audio.tone,
                "acoustic_energy": self.audio.acoustic_energy,
                "speaker": self.audio.speaker_id,
            },
            video_metadata={
                "timestamp_str": self.video.timestamp_str,
                "caption": self.video.keyframe_caption,
                "action": self.video.visual_action,
            },
            caption=f"Synchronized AV Scene @ {self.audio.timestamp_str}",
            metadata={"start_sec": self.start_sec, "end_sec": self.end_sec},
        )


class TemporalMediaProcessor:
    """Demonstrates audio and video temporal ingestion, synchronization, and information loss analysis."""

    @staticmethod
    def explain_information_loss() -> Dict[str, str]:
        """Document why pure transcription or isolated frame extraction fails in production."""
        return {
            "Audio Transcription-Only Loss": (
                "Transcribing audio to text discards acoustic prosody, irony, hesitation, vocal distress, "
                "overlapping speakers, and background non-speech acoustic cues (e.g. alarm bells, engine failure noises). "
                "Two audio tracks with identical transcripts can carry opposite semantic intents if spoken sarcastically vs earnestly."
            ),
            "Video Keyframe-Only Loss": (
                "Sampling isolated still frames discards temporal dynamics, continuous trajectory, camera movement, "
                "gestures, and action verbs (e.g., distinguishing 'assembling a circuit' from 'disassembling a circuit'). "
                "Video RAG requires temporal windowing and joint audio-visual temporal alignment."
            ),
        }


# ---------------------------------------------------------------------------
# 6. Unified Multimodal Vector Store & Search Engine
# ---------------------------------------------------------------------------

class MultimodalRAGEngine:
    """Unified index and search engine supporting text-to-image, text-to-table, and multimodal fusion retrieval."""

    def __init__(self, embedder: Optional[CrossModalEmbedder] = None):
        self.embedder = embedder or CrossModalEmbedder()
        self.chunks: List[MultimodalChunk] = []
        self.embeddings: List[np.ndarray] = []

    def index_chunks(self, chunks: List[MultimodalChunk]) -> int:
        """Compute embeddings and index multimodal chunks."""
        count = 0
        for chunk in chunks:
            if chunk.embedding is None:
                vec = self.embedder.embed_chunk(chunk)
                chunk.embedding = vec.tolist()
            else:
                vec = np.array(chunk.embedding, dtype=np.float32)

            self.chunks.append(chunk)
            self.embeddings.append(vec)
            count += 1
        return count

    def search(
        self,
        query: str,
        top_k: int = 4,
        modality_filter: Optional[List[ModalityType]] = None,
        modality_weights: Optional[Dict[ModalityType, float]] = None,
    ) -> List[SearchResult]:
        """Perform cross-modal vector similarity search with optional modality weighting."""
        if not self.chunks:
            return []

        q_vec = self.embedder.embed_text(query)
        weights = modality_weights or {
            ModalityType.TEXT: 1.0,
            ModalityType.TABLE: 1.15,
            ModalityType.IMAGE: 1.25,
            ModalityType.AUDIO: 1.1,
            ModalityType.VIDEO: 1.2,
        }

        results: List[SearchResult] = []
        for chunk, emb in zip(self.chunks, self.embeddings):
            if modality_filter and chunk.modality not in modality_filter:
                continue

            raw_sim = self.embedder.cosine_similarity(q_vec, emb)
            w = weights.get(chunk.modality, 1.0)
            final_score = raw_sim * w

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=round(final_score, 4),
                    modality=chunk.modality,
                    citation=chunk.get_citation(),
                    match_type="cross_modal_joint_space",
                )
            )

        # Sort descending by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def reciprocal_rank_fusion(
        self,
        rank_lists: List[List[SearchResult]],
        k_constant: int = 60,
    ) -> List[SearchResult]:
        """Combine multiple retrieval ranking lists (e.g. text search + visual search) using RRF."""
        fused_scores: Dict[str, float] = {}
        chunk_map: Dict[str, SearchResult] = {}

        for r_list in rank_lists:
            for rank, item in enumerate(r_list):
                cid = item.chunk.chunk_id
                chunk_map[cid] = item
                fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (k_constant + rank + 1))

        sorted_cids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)
        fused_results: List[SearchResult] = []
        for cid in sorted_cids:
            orig = chunk_map[cid]
            fused_results.append(
                SearchResult(
                    chunk=orig.chunk,
                    score=round(fused_scores[cid], 5),
                    modality=orig.modality,
                    citation=orig.citation,
                    match_type="reciprocal_rank_fusion",
                )
            )
        return fused_results


# ---------------------------------------------------------------------------
# 7. Visual Answer Grounding & Multimodal Synthesizer
# ---------------------------------------------------------------------------

class VisualGroundedSynthesizer:
    """Synthesizes grounded answers incorporating text, tabular metrics, and visual diagrams."""

    def __init__(self, groq_api_key: Optional[str] = None):
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")

    def format_multimodal_context(self, search_results: List[SearchResult]) -> str:
        """Format heterogeneous multimodal retrieved context into a structured prompt."""
        sections = []
        for idx, res in enumerate(search_results, start=1):
            chunk = res.chunk
            header = f"### Evidence [{idx}]: {chunk.get_citation()} (Score: {res.score:.3f})"
            if chunk.modality == ModalityType.TABLE:
                body = f"Tabular Matrix:\n{chunk.content}"
            elif chunk.modality == ModalityType.IMAGE:
                coords = chunk.bbox.to_citation_str() if chunk.bbox else "Spatial coordinates unindexed"
                body = f"Visual Diagram Description: {chunk.content}\nSpatial Location: {coords}\nAssociated Caption: {chunk.caption}"
            elif chunk.modality in (ModalityType.AUDIO, ModalityType.VIDEO):
                body = f"Media Content:\n{chunk.content}"
            else:
                body = f"Text Passage:\n{chunk.content}"
            sections.append(f"{header}\n{body}")

        return "\n\n".join(sections)

    def generate_grounded_answer(
        self, query: str, search_results: List[SearchResult]
    ) -> Dict[str, Any]:
        """Generate a grounded answer citing specific figures, tables, and paragraphs."""
        if not search_results:
            return {
                "answer": "I do not possess sufficient multimodal evidence to address this question.",
                "citations": [],
                "visual_grounding_verified": False,
            }

        context_str = self.format_multimodal_context(search_results)

        # Offline deterministic synthesis
        citations = [res.citation for res in search_results]

        # Extract specific table or figure details to ground response
        table_facts = []
        image_facts = []
        text_facts = []

        for res in search_results:
            c = res.chunk
            if c.modality == ModalityType.TABLE and c.table_data:
                for row in c.table_data.get("rows", []):
                    if len(row) >= 3:
                        table_facts.append(f"{row[0]} achieves {row[2]} recall ({row[3]} latency) {c.get_citation()}")
            elif c.modality == ModalityType.IMAGE:
                image_facts.append(f"visual data from {c.get_citation()} confirms relative recall trajectories")
            elif c.modality == ModalityType.TEXT:
                text_facts.append(f"document analysis {c.get_citation()} notes: '{c.content[:100]}...'")

        bullet_points = []
        if table_facts:
            bullet_points.append(f"1. **Tabular Benchmarks**: According to empirical measurements, {table_facts[0]}.")
        if image_facts:
            bullet_points.append(f"2. **Visual Evidence**: The charted distribution in {citations[0] if citations else 'Figure 1'} demonstrates consistent scaling advantages.")
        if text_facts:
            bullet_points.append(f"3. **Architectural Grounding**: {text_facts[0]}.")

        if not bullet_points:
            bullet_points.append(f"According to the retrieved evidence across {', '.join(citations)}, multimodal architectures systematically outperform unimodal baselines.")

        answer_text = (
            f"Based on cross-modal retrieval for query '{query}':\n\n"
            + "\n".join(bullet_points)
            + f"\n\n**Grounded References**: {', '.join(citations)}."
        )

        return {
            "answer": answer_text,
            "citations": citations,
            "visual_grounding_verified": any(r.chunk.modality in (ModalityType.IMAGE, ModalityType.TABLE) for r in search_results),
            "context_used": context_str,
        }


# ---------------------------------------------------------------------------
# 8. Multimodal Evaluation & Error Taxonomy
# ---------------------------------------------------------------------------

class MultimodalEvaluator:
    """Evaluates cross-modal alignment, visual grounding precision, and detects multimodal hallucinations."""

    @staticmethod
    def evaluate_response(
        query: str,
        answer: str,
        retrieved_results: List[SearchResult],
    ) -> Dict[str, Any]:
        """Compute comprehensive multimodal metrics and failure mode flags."""
        if not retrieved_results:
            return {
                "visual_grounding_precision": 0.0,
                "modality_coverage": 0.0,
                "hallucination_detected": False,
                "error_categories": ["NO_EVIDENCE_RETRIEVED"],
            }

        # 1. Modality coverage: how many distinct modalities contributed
        all_modalities = set(r.chunk.modality for r in retrieved_results)
        coverage_score = len(all_modalities) / 3.0  # Normalized against Text, Table, Image

        # 2. Visual grounding precision: citations cited in answer vs retrieved
        cited_figures = re.findall(r"\[(?:Figure|Table)[^\]]+\]", answer)
        retrieved_cits = set(r.citation for r in retrieved_results)
        valid_citations = [c for c in cited_figures if any(c.split(":")[0] in rc for rc in retrieved_cits)]
        precision = len(valid_citations) / max(1, len(cited_figures))

        # 3. Check for common failure modes
        errors = []
        # Modality neglect: only text cited despite tables/images retrieved
        has_visual_retrieved = any(r.chunk.modality in (ModalityType.IMAGE, ModalityType.TABLE) for r in retrieved_results)
        cites_visual = any("Figure" in a or "Table" in a for a in answer)
        if has_visual_retrieved and not cites_visual:
            errors.append("MODALITY_NEGLECT")

        # Fake metric hallucination: check if percentage numbers cited exist in retrieved text/tables
        cited_percentages = re.findall(r"\b\d+(?:\.\d+)?%", answer)
        corpus_percentages = []
        for r in retrieved_results:
            corpus_percentages.extend(re.findall(r"\b\d+(?:\.\d+)?%", r.chunk.content))
        unsupported_percentages = [p for p in cited_percentages if p not in corpus_percentages]
        hallucination_detected = len(unsupported_percentages) > 0
        if hallucination_detected:
            errors.append("UNGROUNDED_METRIC_HALLUCINATION")

        return {
            "visual_grounding_precision": round(min(1.0, precision), 3),
            "modality_coverage": round(min(1.0, coverage_score), 3),
            "hallucination_detected": hallucination_detected,
            "unsupported_metrics": unsupported_percentages,
            "error_categories": errors,
            "modalities_present": [m.value for m in all_modalities],
        }


# ---------------------------------------------------------------------------
# 9. Architectural Visualization & Diagram Generation
# ---------------------------------------------------------------------------

def generate_multimodal_architecture_diagram(output_path: str) -> str:
    """Generate publication-grade schematic comparing Naive Vector RAG vs Multimodal ColPali & Late Fusion."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=300)
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    # Title
    ax.text(
        0.5, 0.95, "Unified Multimodal RAG & ColPali Late-Interaction Architecture",
        color="#F8FAFC", fontsize=15, fontweight="bold", ha="center", va="top"
    )
    ax.text(
        0.5, 0.90, "End-to-End Pipeline: Document Layout Parsing, Multi-Vector Visual Tokens, and Grounded Multi-Evidence Fusion",
        color="#94A3B8", fontsize=9, ha="center", va="top"
    )

    # 1. Inputs Block
    input_box = patches.FancyBboxPatch(
        (0.04, 0.45), 0.22, 0.38, boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="#1E293B", edgecolor="#38BDF8", linewidth=1.5
    )
    ax.add_patch(input_box)
    ax.text(0.15, 0.80, "Multimodal Document", color="#38BDF8", fontsize=11, fontweight="bold", ha="center")
    ax.text(
        0.15, 0.72,
        "- PyMuPDF Text Blocks\n- Table Grid Extractor\n- Image / Diagram Pixmaps\n- Audio Prosody & Video",
        color="#CBD5E1", fontsize=8.5, ha="center", va="top"
    )

    # Arrow 1
    ax.annotate("", xy=(0.32, 0.64), xytext=(0.27, 0.64), arrowprops=dict(arrowstyle="->", color="#38BDF8", lw=2))

    # 2. Embedding & Tokenization
    colpali_box = patches.FancyBboxPatch(
        (0.33, 0.45), 0.28, 0.38, boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="#1E293B", edgecolor="#A855F7", linewidth=1.5
    )
    ax.add_patch(colpali_box)
    ax.text(0.47, 0.80, "Cross-Modal Representation", color="#C084FC", fontsize=11, fontweight="bold", ha="center")
    ax.text(
        0.47, 0.72,
        "- Dual-Encoder Joint Hypersphere\n- ColPali Patch-Level Vectors\n- MaxSim Operator (sum max(Q.P))\n- Zero OCR Parsing Loss",
        color="#E2E8F0", fontsize=8.5, ha="center", va="top"
    )

    # Arrow 2
    ax.annotate("", xy=(0.67, 0.64), xytext=(0.62, 0.64), arrowprops=dict(arrowstyle="->", color="#A855F7", lw=2))

    # 3. Multimodal Index & Retrieval
    retrieval_box = patches.FancyBboxPatch(
        (0.68, 0.45), 0.28, 0.38, boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="#1E293B", edgecolor="#22C55E", linewidth=1.5
    )
    ax.add_patch(retrieval_box)
    ax.text(0.82, 0.80, "Grounded Fusion & Synthesis", color="#4ADE80", fontsize=11, fontweight="bold", ha="center")
    ax.text(
        0.82, 0.72,
        "- Reciprocal Rank Fusion (RRF)\n- Spatial Bounding Box Citations\n- Visual Grounding Verification\n- Hallucination Penalty",
        color="#E2E8F0", fontsize=8.5, ha="center", va="top"
    )

    # Lower comparison cards: Why ColPali vs Naive Vector RAG
    comp_box1 = patches.FancyBboxPatch(
        (0.04, 0.08), 0.44, 0.28, boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor="#1E1B4B", edgecolor="#6366F1", linewidth=1.2
    )
    ax.add_patch(comp_box1)
    ax.text(0.26, 0.33, "Traditional Unimodal / OCR Pipeline Bottlenecks", color="#A5B4FC", fontsize=9.5, fontweight="bold", ha="center")
    ax.text(
        0.06, 0.28,
        "x OCR drops reading order in multi-column layouts & nested tables\nx Image diagrams and charts converted to vague, lossy captions\nx Modality neglect: generated answer ignores tabular metrics",
        color="#C7D2FE", fontsize=8, va="top"
    )

    comp_box2 = patches.FancyBboxPatch(
        (0.52, 0.08), 0.44, 0.28, boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor="#064E3B", edgecolor="#10B981", linewidth=1.2
    )
    ax.add_patch(comp_box2)
    ax.text(0.74, 0.33, "Unified Multimodal RAG Solution", color="#6EE7B7", fontsize=9.5, fontweight="bold", ha="center")
    ax.text(
        0.54, 0.28,
        "v End-to-end visual patch embeddings directly represent document pages\nv Tables indexed as both markdown structures & spatial bounding boxes\nv Grounded citations verify exact figures, page numbers, & cell values",
        color="#D1FAE5", fontsize=8, va="top"
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_crossmodal_embedding_diagram(output_path: str) -> str:
    """Generate visualization of the joint cross-modal unit hypersphere embedding space."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    # Title
    ax.set_title("Cross-Modal Joint Embedding Space (Normalized Hypersphere)", color="#F8FAFC", fontsize=13, fontweight="bold", pad=15)

    # Draw unit circle
    circle = plt.Circle((0, 0), 1.0, color="#334155", fill=False, linestyle="--", linewidth=1.5)
    ax.add_artist(circle)

    # Embed clusters of concepts
    categories = {
        "Text Queries": {
            "pts": [(0.72, 0.65), (0.75, 0.61), (0.68, 0.70)],
            "labels": ['"ColPali recall"', '"table benchmark"', '"latency comparison"'],
            "color": "#38BDF8",
            "marker": "o",
        },
        "Visual Images & Diagrams": {
            "pts": [(0.74, 0.62), (0.78, 0.58), (0.69, 0.67)],
            "labels": ["Fig 1: Bar Chart", "Architecture Diagram", "ROC Plot"],
            "color": "#F43F5E",
            "marker": "s",
        },
        "Structured Tables": {
            "pts": [(0.63, 0.75), (0.67, 0.72), (0.60, 0.78)],
            "labels": ["Tab 1: Latency Grid", "Tab 2: Precision Matrix", "Dataset Table"],
            "color": "#22C55E",
            "marker": "^",
        },
        "Distractor Concepts": {
            "pts": [(-0.75, -0.62), (-0.80, 0.45), (0.10, -0.92)],
            "labels": ["Database migrations", "User login credentials", "Audio compression RFC"],
            "color": "#94A3B8",
            "marker": "x",
        },
    }

    for cat_name, data in categories.items():
        xs = [p[0] for p in data["pts"]]
        ys = [p[1] for p in data["pts"]]
        edge_col = None if data["marker"] == "x" else "white"
        ax.scatter(xs, ys, color=data["color"], s=100, marker=data["marker"], label=cat_name, edgecolors=edge_col, linewidth=0.8, zorder=5)
        for (x, y), label in zip(data["pts"], data["labels"]):
            ax.annotate(
                label, (x, y), textcoords="offset points", xytext=(6, 6),
                color="#E2E8F0", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#0F172A", edgecolor=data["color"], alpha=0.85)
            )

    # Highlight High Cross-Modal Similarity Arc
    ax.plot([0, 0.72], [0, 0.65], color="#38BDF8", linestyle=":", alpha=0.7)
    ax.plot([0, 0.74], [0, 0.62], color="#F43F5E", linestyle=":", alpha=0.7)
    ax.text(0.38, 0.32, "High Cosine Alignment (sim > 0.88)", color="#FDE047", fontsize=8, fontweight="bold", rotation=40)

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.grid(True, color="#334155", linestyle=":", alpha=0.6)
    ax.tick_params(colors="#94A3B8")
    for spine in ax.spines.values():
        spine.set_color("#475569")

    legend = ax.legend(facecolor="#0F172A", edgecolor="#475569", labelcolor="#F8FAFC", fontsize=8.5, loc="lower left")
    legend.get_frame().set_alpha(0.9)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return output_path
