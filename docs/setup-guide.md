# Environment Setup Guide

This guide describes how to configure, verify, and maintain the local environment for the **RAG Engineering Academy**.

---

## 💻 Environment & Architecture

- **Operating System:** macOS (Apple Silicon, arm64) / Linux x86_64
- **Python Version:** Python 3.12 (specifically `Python 3.12.13`)
- **Virtual Environment Directory:** `.venv312`
- **Vector Database:** ChromaDB (local persistence)
- **Primary LLM Provider:** Groq API

> [!IMPORTANT]
> **ChromaDB & onnxruntime Compatibility on macOS arm64:**
> ChromaDB 1.x relies on `onnxruntime>=1.14.1`. As of late 2026, official pre-compiled `onnxruntime` wheels are unavailable for Python 3.15 on Apple Silicon arm64.
> To ensure seamless local execution, this project uses **Python 3.12**, which provides pre-compiled `onnxruntime` (1.31.1) and `chromadb` (1.5.9) binaries without compilation issues. Always run project commands against `.venv312`.

---

## 🛠️ Step-by-Step Installation

### 1. Activating the Virtual Environment

Ensure you activate `.venv312` before running commands:

```bash
# From the repository root
source .venv312/bin/activate

# Verify Python location and version
which python
python --version  # Should output Python 3.12.x
```

### 2. Installing Base Dependencies

Install the core dependencies required across all tracks:

```bash
pip install -r requirements/base.txt
```

This installs:
- Vector DB: `chromadb` (1.5.x)
- LLM SDK: `groq`
- Document Parsing: `pymupdf` (AGPL-3.0) and `pypdf` (Apache 2.0)
- Core Math & Data: `numpy`, `pandas`
- Configuration & Validation: `python-dotenv`, `pydantic`, `pydantic-settings`
- Visualization: `matplotlib`
- Interactive & Testing: `jupyterlab`, `ipykernel`, `pytest`, `nbformat`, `nbconvert`

---

## ⚖️ PyMuPDF Licensing Note

- **PyMuPDF (`pymupdf`)** is distributed under the **GNU AGPL-3.0** license for open-source software.
- Commercial applications requiring closed-source redistribution must obtain an Artifex commercial license.
- In this academy, PyMuPDF is used for first-principles PDF layout and text extraction in learning notebooks.
- For comparison, the Apache 2.0 licensed **`pypdf`** library is also installed and explored in Track 01.
- Modern import syntax: use `import pymupdf` (`import fitz` remains supported as a legacy alias).

---

## 🔐 Environment Variables & Secrets Configuration

All secrets are managed via a root `.env` file that is **strictly excluded** from Git tracking.

1. Copy the example configuration template:
   ```bash
   cp .env.example .env
   ```

2. Open `.env` and fill in optional keys as needed:
   ```dotenv
   GROQ_API_KEY=your_groq_api_key_here
   ```

> [!CAUTION]
> - Never commit `.env` or paste API keys into notebooks, commit messages, or chat transcripts.
> - Notebooks are designed to run with deterministic local mocks by default so that an API key is not required for fundamental exercises.

---

## 📡 Optional LangSmith Tracing

In tracks covering LangChain and LangGraph (Tracks 05–06) and Evaluation (Track 15), tracing can optionally be enabled:

- Modern LangSmith SDKs (`langsmith >= 0.1`) look for `LANGSMITH_API_KEY`.
- Legacy LangChain configurations looked for `LANGCHAIN_API_KEY`.
- Both are documented in `.env.example`.
- Keep `LANGSMITH_TRACING=false` by default to avoid unintended data transmission.

---

## ✅ Verifying Your Installation

Run the quick verification script:

```bash
python -c "
import chromadb
import pymupdf
import groq
import numpy
import pytest
import matplotlib
print('All core dependencies imported successfully!')
"
```
