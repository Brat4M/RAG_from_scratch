# Contributing to RAG Engineering Academy

Thank you for contributing to the RAG Engineering Academy! This repository is built as a rigorous, step-by-step educational resource. To preserve code quality, pedagogical clarity, and execution reliability, please follow these guidelines.

---

## 📐 Pedagogical & Notebook Standards

1. **Mandatory Markdown-Before-Code-Cell Rule:**
   - Every executable code cell **must** be immediately preceded by an explanatory Markdown cell explaining:
     - **What** the code does.
     - **Why** this step is performed.
     - **How** it works (concepts, mathematics, API details).
     - **What to expect** (expected output or interpretation).
     - **Caveats** (costs, optional API keys, performance).
   - Never chain multiple code cells together without intervening explanation.

2. **Cost and Resource Control:**
   - Default cells must execute without requiring API keys, paid services, or large downloads.
   - Any external API calls (e.g. Groq) must be placed in clearly marked optional cells with cost/token explanations.
   - Do not generate synthetic test datasets or large synthetic document collections. Use minimal deterministic inline fixtures or shared fixtures in `fixtures/`.

3. **Incremental Execution:**
   - Notebooks must be self-contained and independently executable from a clean kernel.
   - Outputs should remain clean and uncluttered.

---

## 🛠️ Development & Git Workflow

1. **Virtual Environment:**
   - Use Python 3.12 (`.venv312`).
   - Do not install experimental frameworks or heavy dependencies into base requirements. Track-specific requirements belong in `requirements/track-XX.txt`.

2. **Commits and Hygiene:**
   - Review `git status` and `git diff --staged` before creating commits.
   - Never commit `.env`, secrets, model weights, large datasets, or temporary cache files.
   - Use clear, descriptive commit messages following Conventional Commits (e.g., `feat(track-01): add recursive chunking notebook`).

3. **Validation:**
   - Run `pytest` before opening a pull request or submitting changes.
   - Ensure notebook JSON is well-formed.
