# Track 01 — Ingestion and Chunking

Document parsing (PyMuPDF / pypdf), text normalization, spatial layout reconstruction, and chunking strategies.

---

## 📚 Notebook Directory

| ID | Notebook File | Focus / Concepts Covered | Status |
|---|---|---|---|
| **01.1** | [`01_document_ingestion_and_pdf_parsing.ipynb`](./01_document_ingestion_and_pdf_parsing.ipynb) | PDF anatomy, reading order trap, PyMuPDF vs pypdf comparative benchmark, spatial bounding box filtering, and `pymupdf4llm` markdown formatting | ✅ `VALIDATED` |
| **01.2** | `02_text_cleaning_and_normalization.ipynb` | Whitespace normalization, unicode artifact stripping, regex hygiene, and encoding normalization | ⏳ `PLANNED` |
| **01.3** | `03_fixed_and_recursive_chunking.ipynb` | Fixed-size vs. recursive character chunking, token vs. char metrics, and chunk overlap trade-offs | ⏳ `PLANNED` |
| **01.4** | `04_semantic_and_hierarchical_chunking.ipynb` | Semantic distance chunking, parent-child, and sentence-window retrieval | ⏳ `PLANNED` |
| **01.5** | `05_structural_and_late_chunking.ipynb` | Headings, tables, code block chunking, late chunking intuition, and chunking evaluation | ⏳ `PLANNED` |

