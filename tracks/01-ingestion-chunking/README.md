# Track 01 — Ingestion and Chunking

Document parsing (PyMuPDF / pypdf), text normalization, spatial layout reconstruction, and chunking strategies.

---

## 📚 Notebook Directory

| ID | Notebook File | Focus / Concepts Covered | Status |
|---|---|---|---|
| **01.1** | [`01_document_ingestion_and_pdf_parsing.ipynb`](./01_document_ingestion_and_pdf_parsing.ipynb) | PDF anatomy, reading order trap, PyMuPDF vs pypdf comparative benchmark, spatial bounding box filtering, and `pymupdf4llm` markdown formatting | ✅ `VALIDATED` |
| **01.2** | [`02_text_cleaning_and_chunking.ipynb`](./02_text_cleaning_and_chunking.ipynb) | Unicode NFKC normalization, regex de-hyphenation, sliding-window with word boundaries, recursive character splitting, and semantic distance heuristics | ✅ `VALIDATED` |
| **01.3** | [`03_structure_aware_and_code_chunking.ipynb`](./03_structure_aware_and_code_chunking.ipynb) | Document layout trees, GFM section splitting with code fence tracking, Python AST code chunking, and table schema preservation heuristics | ✅ `VALIDATED` |
| **01.4** | `04_hierarchical_and_parent_child_retrieval.ipynb` | Hierarchical parent-child indexing, sentence-window retrieval, and late chunking intuition | ⏳ `PLANNED` |

