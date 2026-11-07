"""Unit tests for Track 10 Multimodal RAG architectures and components."""

from __future__ import annotations

import io
import os
import numpy as np
import pytest
from PIL import Image

from common.multimodal_rag import (
    AlignedTemporalUnit,
    AudioSegment,
    BoundingBox,
    ColPaliMatcher,
    CrossModalEmbedder,
    DocumentVisualParser,
    ModalityType,
    MultimodalChunk,
    MultimodalEvaluator,
    MultimodalRAGEngine,
    SearchResult,
    TemporalMediaProcessor,
    VideoSegment,
    VisualGroundedSynthesizer,
    generate_crossmodal_embedding_diagram,
    generate_multimodal_architecture_diagram,
)


def test_bounding_box_geometry_and_iou():
    """Test 2D spatial bounding box calculations, IoU, normalization, and citation formatting."""
    box_a = BoundingBox(x0=50.0, y0=50.0, x1=150.0, y1=150.0, page_num=1)
    box_b = BoundingBox(x0=100.0, y0=100.0, x1=200.0, y1=200.0, page_num=1)
    box_c = BoundingBox(x0=300.0, y0=300.0, x1=400.0, y1=400.0, page_num=1)
    box_diff_page = BoundingBox(x0=50.0, y0=50.0, x1=150.0, y1=150.0, page_num=2)

    assert box_a.width == 100.0
    assert box_a.height == 100.0
    assert box_a.area == 10000.0

    # Overlapping boxes A and B
    # Intersection = [100, 100, 150, 150] -> area 50x50 = 2500
    # Union = 10000 + 10000 - 2500 = 17500
    # IoU = 2500 / 17500 = 1/7 ~= 0.142857
    iou_ab = box_a.iou(box_b)
    assert pytest.approx(iou_ab, rel=1e-3) == 2500.0 / 17500.0

    # Disjoint boxes
    assert box_a.iou(box_c) == 0.0

    # Different pages should have 0 IoU
    assert box_a.iou(box_diff_page) == 0.0

    # Normalization
    norm_box = box_a.normalize(page_width=500.0, page_height=1000.0)
    assert norm_box.x0 == 0.1
    assert norm_box.y0 == 0.05
    assert norm_box.x1 == 0.3
    assert norm_box.y1 == 0.15

    # Citation string
    assert box_a.to_citation_str() == "p.1[50,50,150,150]"


def test_multimodal_chunk_and_citation_formats():
    """Verify chunk representations across Text, Table, Image, Audio, and Video."""
    bbox = BoundingBox(x0=10.0, y0=20.0, x1=100.0, y1=80.0, page_num=1)

    c_text = MultimodalChunk(
        chunk_id="t1", modality=ModalityType.TEXT, content="Introduction to RAG", page_num=1, bbox=bbox
    )
    c_table = MultimodalChunk(
        chunk_id="tab1", modality=ModalityType.TABLE, content="| Metric | Val |\n| --- | --- |\n| Recall | 90% |",
        page_num=1, bbox=bbox, caption="Recall Matrix"
    )
    c_img = MultimodalChunk(
        chunk_id="img1", modality=ModalityType.IMAGE, content="Performance ROC Curve", page_num=1,
        bbox=bbox, caption="ROC Curve"
    )
    c_audio = MultimodalChunk(
        chunk_id="aud1", modality=ModalityType.AUDIO, content="Key findings explained",
        audio_metadata={"timestamp_str": "01:15-01:45", "tone": "confident"}
    )
    c_video = MultimodalChunk(
        chunk_id="vid1", modality=ModalityType.VIDEO, content="Demo walkthrough",
        video_metadata={"timestamp_str": "02:00-02:30", "action": "code execution"}
    )

    assert "[Text: p.1]" == c_text.get_citation()
    assert "[Table: Recall Matrix, p.1[10,20,100,80]]" == c_table.get_citation()
    assert "[Figure: ROC Curve, p.1[10,20,100,80]]" == c_img.get_citation()
    assert "[Audio: aud1 @ 01:15-01:45]" == c_audio.get_citation()
    assert "[Video: vid1 @ 02:00-02:30]" == c_video.get_citation()


def test_synthetic_multimodal_pdf_creation_and_parsing(tmp_path):
    """Test generating a synthetic multimodal PDF and parsing text, tables, and images with PyMuPDF."""
    pdf_path = str(tmp_path / "test_multimodal.pdf")
    pdf_bytes = DocumentVisualParser.create_synthetic_multimodal_pdf(pdf_path)
    assert len(pdf_bytes) > 500
    assert os.path.exists(pdf_path)

    parser = DocumentVisualParser()
    chunks = parser.parse_pdf(pdf_path)
    assert len(chunks) >= 3

    modalities = set(c.modality for c in chunks)
    assert ModalityType.TEXT in modalities
    assert ModalityType.TABLE in modalities
    assert ModalityType.IMAGE in modalities

    # Verify table extraction
    table_chunk = next(c for c in chunks if c.modality == ModalityType.TABLE)
    assert table_chunk.table_data is not None
    assert "Recall" in table_chunk.content or "Model" in table_chunk.content

    # Verify image extraction
    img_chunk = next(c for c in chunks if c.modality == ModalityType.IMAGE)
    assert img_chunk.bbox is not None
    assert img_chunk.page_num == 1


def test_cross_modal_embedder_hypersphere_normalization():
    """Verify that text, image, table, audio, and video embeddings lie on unit hypersphere."""
    embedder = CrossModalEmbedder(embedding_dim=64, seed=123)

    # 1. Text embedding
    t_vec = embedder.embed_text("benchmark recall and latency performance on table")
    assert t_vec.shape == (64,)
    assert pytest.approx(np.linalg.norm(t_vec), rel=1e-4) == 1.0

    # 2. Image embedding
    test_img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    img_vec = embedder.embed_image(test_img)
    assert img_vec.shape == (64,)
    assert pytest.approx(np.linalg.norm(img_vec), rel=1e-4) == 1.0

    # 3. Table embedding
    tab_vec = embedder.embed_table("| Metric | Score |\n| --- | --- |\n| Accuracy | 92% |")
    assert tab_vec.shape == (64,)
    assert pytest.approx(np.linalg.norm(tab_vec), rel=1e-4) == 1.0

    # 4. Audio & video embedding
    aud_vec = embedder.embed_audio({"acoustic_energy": 0.8}, transcript="Audio acoustic analysis")
    vid_vec = embedder.embed_video({"fps": 30}, caption="Video keyframe motion")
    assert pytest.approx(np.linalg.norm(aud_vec), rel=1e-4) == 1.0
    assert pytest.approx(np.linalg.norm(vid_vec), rel=1e-4) == 1.0

    # Semantic cross-modal alignment test
    q_table = embedder.embed_text("table benchmark latency metrics")
    sim_table = embedder.cosine_similarity(q_table, tab_vec)
    sim_unrelated = embedder.cosine_similarity(q_table, embedder.embed_text("irrelevant gardening plants flowers"))
    assert sim_table > sim_unrelated


def test_colpali_late_interaction_maxsim_scoring():
    """Verify ColPali multi-vector token-to-patch extraction and MaxSim inner-product scoring."""
    embedder = CrossModalEmbedder(embedding_dim=32, seed=42)
    matcher = ColPaliMatcher(embedder=embedder, patch_grid=(3, 3))

    test_img = Image.new("RGB", (90, 90), color=(240, 240, 240))
    patches = matcher.extract_page_patches(test_img)
    assert len(patches) == 9  # 3x3 grid
    for p in patches:
        assert p.shape == (32,)

    query = "performance chart latency"
    score, sim_matrix = matcher.maxsim_score(query, patches)
    assert score > 0.0
    # Query has 3 tokens -> shape should be (3, 9)
    assert sim_matrix.shape == (3, 9)


def test_temporal_media_alignment_and_information_loss():
    """Test audio-video temporal synchronization and educational information loss documentation."""
    aud = AudioSegment(
        start_sec=12.0, end_sec=25.0, transcript="We observed substantial latency drop",
        tone="confident", acoustic_energy=0.85, speaker_id="researcher_1"
    )
    vid = VideoSegment(
        start_sec=12.0, end_sec=25.0, keyframe_caption="Latency comparison slide",
        visual_action="pointing to ColPali bar"
    )
    unit = AlignedTemporalUnit(
        start_sec=12.0, end_sec=25.0, audio=aud, video=vid, synthesis_summary="Presenter explains 32ms latency."
    )

    chunk = unit.to_chunk()
    assert chunk.modality == ModalityType.VIDEO
    assert "00:12-00:25" in chunk.content
    assert chunk.audio_metadata["tone"] == "confident"

    info_loss = TemporalMediaProcessor.explain_information_loss()
    assert "Audio Transcription-Only Loss" in info_loss
    assert "Video Keyframe-Only Loss" in info_loss


def test_multimodal_rag_engine_indexing_and_search():
    """Verify heterogeneous multimodal chunk indexing, modality filtering, and search."""
    engine = MultimodalRAGEngine()

    c1 = MultimodalChunk(chunk_id="txt1", modality=ModalityType.TEXT, content="General text about database indexing and caching.")
    c2 = MultimodalChunk(
        chunk_id="tab1", modality=ModalityType.TABLE, content="| Model | Latency |\n| --- | --- |\n| ColPali | 32ms |",
        table_data={"headers": ["Model", "Latency"], "rows": [["ColPali", "32ms"]]}
    )
    c3 = MultimodalChunk(
        chunk_id="img1", modality=ModalityType.IMAGE, content="Chart diagram showing retrieval recall bars",
        caption="Recall Diagram"
    )

    indexed = engine.index_chunks([c1, c2, c3])
    assert indexed == 3
    assert len(engine.chunks) == 3

    # Query matching table benchmark
    results = engine.search("table benchmark latency metrics", top_k=2)
    assert len(results) == 2
    # The table should rank highest for this query
    assert results[0].chunk.chunk_id == "tab1"

    # Filter by modality
    img_only = engine.search("retrieval chart", top_k=5, modality_filter=[ModalityType.IMAGE])
    assert len(img_only) == 1
    assert img_only[0].chunk.modality == ModalityType.IMAGE


def test_multimodal_rag_engine_reciprocal_rank_fusion():
    """Verify RRF rank merging across text and visual retrieval lists."""
    engine = MultimodalRAGEngine()

    c1 = MultimodalChunk(chunk_id="c1", modality=ModalityType.TEXT, content="Passage 1")
    c2 = MultimodalChunk(chunk_id="c2", modality=ModalityType.IMAGE, content="Diagram 2")
    c3 = MultimodalChunk(chunk_id="c3", modality=ModalityType.TABLE, content="Table 3")

    r1 = [
        SearchResult(chunk=c1, score=0.9, modality=ModalityType.TEXT, citation="[Text: 1]"),
        SearchResult(chunk=c2, score=0.7, modality=ModalityType.IMAGE, citation="[Figure: 2]"),
    ]
    r2 = [
        SearchResult(chunk=c2, score=0.95, modality=ModalityType.IMAGE, citation="[Figure: 2]"),
        SearchResult(chunk=c3, score=0.8, modality=ModalityType.TABLE, citation="[Table: 3]"),
    ]

    fused = engine.reciprocal_rank_fusion([r1, r2], k_constant=60)
    assert len(fused) == 3
    # c2 appears at rank 2 in list 1 and rank 1 in list 2 -> highest cumulative RRF score
    assert fused[0].chunk.chunk_id == "c2"


def test_visual_grounded_synthesizer_citations():
    """Test grounded prompt synthesis and citation formatting."""
    synthesizer = VisualGroundedSynthesizer()

    bbox = BoundingBox(x0=50.0, y0=180.0, x1=545.0, y1=280.0, page_num=1)
    c_tab = MultimodalChunk(
        chunk_id="tab1", modality=ModalityType.TABLE,
        content="| Architecture | Recall |\n| --- | --- |\n| ColPali | 88.6% |",
        bbox=bbox, caption="Table 1",
        table_data={"headers": ["Architecture", "Recall"], "rows": [["ColPali", "Multimodal", "88.6%", "32.0 ms"]]}
    )
    res = [SearchResult(chunk=c_tab, score=0.92, modality=ModalityType.TABLE, citation="[Table: Table 1, p.1[50,180,545,280]]")]

    answer_obj = synthesizer.generate_grounded_answer("What is ColPali recall?", res)
    assert "answer" in answer_obj
    assert "88.6%" in answer_obj["answer"]
    assert "[Table: Table 1, p.1[50,180,545,280]]" in answer_obj["citations"]
    assert answer_obj["visual_grounding_verified"] is True


def test_multimodal_evaluator_metrics_and_taxonomy():
    """Verify visual grounding precision, modality coverage, and metric hallucination detection."""
    c_img = MultimodalChunk(
        chunk_id="img1", modality=ModalityType.IMAGE, content="Recall Chart: ColPali 88.6%, Dense 64.2%",
        caption="Recall Diagram"
    )
    res = [SearchResult(chunk=c_img, score=0.88, modality=ModalityType.IMAGE, citation="[Figure: Recall Diagram, p.1]")]

    # 1. Grounded response
    grounded_answer = "According to [Figure: Recall Diagram, p.1], ColPali achieves 88.6% recall."
    eval_res = MultimodalEvaluator.evaluate_response("What is recall?", grounded_answer, res)
    assert eval_res["visual_grounding_precision"] == 1.0
    assert eval_res["hallucination_detected"] is False

    # 2. Response with fabricated metric (99.9% not in context)
    hallucinated_answer = "The model achieved 99.9% recall as seen in [Figure: Recall Diagram, p.1]."
    eval_bad = MultimodalEvaluator.evaluate_response("What is recall?", hallucinated_answer, res)
    assert eval_bad["hallucination_detected"] is True
    assert "99.9%" in eval_bad["unsupported_metrics"]
    assert "UNGROUNDED_METRIC_HALLUCINATION" in eval_bad["error_categories"]


def test_diagram_generation(tmp_path):
    """Verify architectural and cross-modal embedding diagrams generate valid PNG files."""
    d1_path = str(tmp_path / "arch.png")
    d2_path = str(tmp_path / "embed.png")

    generate_multimodal_architecture_diagram(d1_path)
    generate_crossmodal_embedding_diagram(d2_path)

    assert os.path.exists(d1_path) and os.path.getsize(d1_path) > 1000
    assert os.path.exists(d2_path) and os.path.getsize(d2_path) > 1000
