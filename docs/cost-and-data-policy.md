# Cost, Resource, and Test Data Policy

This policy governs resource consumption, external API calls, dataset generation, and test execution across all stages of the RAG Engineering Academy.

---

## 🚫 1. Strict Prohibition on Synthetic Data Generation

1. **No Synthetic Document Generation:**
   - Never use LLM calls to manufacture synthetic document corpora, synthetic evaluation questions, or fake benchmark results.
   - Do not create artificial multi-page documents just to populate folders.

2. **User-Supplied Source Documents:**
   - Real documents, PDFs, manuals, and evaluation benchmarks are supplied manually by the user.
   - All user-supplied sample files belong in the `fixtures/` directory.

3. **Minimal Deterministic Inline Fixtures:**
   - When introducing an algorithm (e.g., chunking, similarity search, reranking), use tiny inline string lists defined directly inside the notebook cell.
   - 3 to 5 clear sentences are sufficient to prove or disprove retrieval mechanics.

---

## 💳 2. LLM & External API Usage Rules

1. **Default Execution is Offline:**
   - Every default notebook cell and all test suites must execute successfully without requiring API keys or network calls.
   - Use deterministic mock responses and local Python implementations for standard runs.

2. **Isolated Optional API Cells:**
   - Real calls to external models (e.g., Groq API) must reside in dedicated, clearly labeled optional cells.
   - Each optional cell must explain the estimated token consumption and explain how to skip it.

3. **No Redundant or Loop Invocations:**
   - Never loop over LLM calls in automated test runners.
   - Never run unconstrained agentic loops or LLM-as-a-judge sweeps without hard limits and approval.

---

## 💾 3. Compute, Disk Space, and Downloads

1. **Respect Disk Constraints:**
   - Apple Silicon environments frequently have limited internal storage. Keep large model caches off disk unless explicitly required.
   - Do not download multi-gigabyte checkpoints or heavy embedding weights automatically.

2. **Pragmatic Dependencies (YAGNI):**
   - Install only packages required for the currently active stage.
   - Defer framework packages (LangChain, LangGraph, Neo4j, etc.) to their respective tracks.
