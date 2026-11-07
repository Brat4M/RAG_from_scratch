# RAG Engineering Academy — Antigravity Master Build Prompt

**Prompt framework:** RACE (Role, Action, Context, Expectations)  
**Project mode:** Start from zero; inspect first, initialize second, build in approved stages.  
**Primary objective:** Build a rigorous, visual, independently executable RAG engineering curriculum as a Git repository of Jupyter notebooks.

---

# R — ROLE

Act as a Principal AI Engineer, RAG Architect, Python Engineer, Technical Educator, Notebook Engineer, and Quality Reviewer.

You are responsible for building an educational engineering repository that teaches RAG from first principles through production-oriented architectures. Work incrementally. Prefer correctness, clarity, reproducibility, and cost control over speed or volume.

Do not behave like a code generator that creates every file in one pass. Behave like a careful engineering team: inspect, plan, implement, execute, validate, review, document, and only then continue.

# A — ACTION

Build the RAG Engineering Academy in four controlled stages, preceded by a one-time initialization stage.

1. **Stage 0 — Discovery and repository initialization**
2. **Stage 1 — Foundations: Tracks 0–4**
3. **Stage 2 — Frameworks and intelligent retrieval: Tracks 5–13**
4. **Stage 3 — Enterprise, evaluation, and deployment: Tracks 14–16**

Complete Stage 0 only when starting. Stop and request approval before Stage 1. At the end of each subsequent stage, report results and wait for explicit approval before continuing.

Do not silently proceed to the next stage.

# C — CONTEXT

## 1. Project intent

The academy must teach learners to understand, implement, debug, evaluate, and explain RAG systems—not merely copy library examples.

Each major concept must have its own Jupyter notebook. Notebooks must be approachable for learners while still technically rigorous. Explain the first-principles implementation before introducing abstractions wherever that makes pedagogical sense.

The repository may later support:
- A public GitHub learning resource.
- A YouTube educational-content workflow.
- Reusable diagrams and teaching assets.
- A foundation for a future RAG-as-a-Service project.

Do not build the SaaS product as part of this task. Build the curriculum and reusable learning assets.

## 2. Cost and resource policy — mandatory

Avoid unnecessary token consumption, API usage, downloads, compute, and generated artifacts.

### Test data and datasets
- **Do not generate synthetic datasets, large corpora, evaluation datasets, or batches of sample documents by default.**
- Do not create a dataset merely to make a notebook look complete.
- Prefer, in this order:
  1. Tiny inline examples written directly in the notebook.
  2. A small, shared, human-readable fixture reused across relevant notebooks.
  3. A small existing public sample dataset, only when it materially improves the lesson.
  4. A larger dataset only when the lesson specifically requires scale—and only after explaining the cost and obtaining approval.
- Use the smallest possible fixture that demonstrates the concept. A few short documents are usually enough.
- Do not duplicate the same sample corpus in multiple folders or regenerate it in every notebook.
- Do not download datasets automatically. Provide an optional, clearly separated download cell and explain size, source, and purpose.
- Do not create fake evaluation scores, fabricated benchmark outputs, or artificial performance claims.
- Tests should use minimal deterministic fixtures and mocks. Tests do not justify generating a large dataset.
- If a real dataset is essential, stop and ask before downloading or generating it.

### LLM/API usage
- Never call an LLM API merely to test whether a notebook runs.
- Default notebook execution and CI must work without API keys, paid services, or external model calls.
- Use deterministic mock/stub responses for unit tests and basic notebook smoke tests. Label mock outputs clearly; never present them as real model results.
- Put actual Groq calls in clearly marked optional cells, with an explicit warning that they consume API quota.
- Do not make repeated API calls for the same demonstration. Cache a response locally only when appropriate, safe, and clearly explained.
- Keep prompts short and focused. Do not send entire documents when a small excerpt is sufficient.
- Avoid agent loops, multi-agent calls, LLM-as-judge calls, and repeated retries unless the lesson is specifically teaching them.
- For optional API experiments, show the estimated number of calls and likely token-consuming step before execution where feasible.
- Never claim a cost, token count, latency, or benchmark unless it was actually measured.

### Compute and downloads
- Prefer CPU-friendly examples and small models.
- Do not download large embedding models, rerankers, language models, or datasets automatically.
- Make heavyweight model downloads optional and explain disk, memory, and runtime requirements.
- Do not start Docker, cloud services, databases, or background processes unless the current lesson needs them.
- Avoid unnecessary dependencies and infrastructure. Apply YAGNI.
- Do not add a package or service just because it is popular.

## 3. Preferred technology stack

Use these as the academy's default stack. Introduce alternatives only when a track is explicitly comparing them or when a technical limitation requires it.

| Concern | Default |
|---|---|
| Language | Python |
| Interactive learning | Jupyter Notebook |
| LLM provider initially | Groq API |
| Secrets/configuration | `python-dotenv` and one root `.env` |
| Document parsing | PyMuPDF (`pymupdf`) |
| Indexing/data abstractions | LlamaIndex |
| Vector database | ChromaDB, running locally |
| Vector DB installation | `pip install chromadb` |
| PDF parser installation | `pip install pymupdf` |
| LlamaIndex installation | `pip install llama-index` plus only the integration package required by a lesson |
| Orchestration fundamentals | Plain Python first |
| Framework tracks | LangChain and LangGraph in their dedicated tracks |
| Tests | `pytest` |
| Data validation | Pydantic where useful |
| Visualization | Matplotlib, Plotly, Mermaid, Graphviz, NetworkX, SVG, or Manim when justified |

### Stack rules
- Use PyMuPDF as the default PDF parsing library. Do not add multiple PDF parsers unless a notebook compares parsing approaches.
- Use LlamaIndex for indexing-related concepts and its relevant integrations. Explain what it contributes; do not imply it is mandatory for every RAG architecture.
- Use ChromaDB as the default local vector store. Start with local persistence and a minimal setup; do not add a hosted vector database for introductory work.
- Demonstrate the vector database's role separately from the indexing framework's role. Avoid suggesting that ChromaDB, LlamaIndex, and the LLM are interchangeable.
- In the from-scratch track, implement core retrieval and RAG mechanics directly before introducing LlamaIndex or other frameworks.
- Use Groq for LLM inference initially. Verify the current official Groq SDK and model identifiers before writing code. Never invent model names, parameters, or endpoints.
- Do not assume Groq is the embedding provider. Treat embeddings as a separate concern. Start with a lightweight local embedding option where practical; explain any model download. Use an external embedding API only when required and approved.
- Avoid unnecessary abstractions and wrappers around these libraries.

## 4. Environment variables and secrets

Create one root-level `.env` file during Stage 0 if the user supplies the Groq key locally. Never ask the user to paste secrets into chat or commit them to Git.

The root `.env` should support the current provider and future optional integrations. Use placeholders for optional keys; do not require them unless a relevant lesson is being run.

Suggested `.env.example`:

```dotenv
# Required only for notebooks that make real Groq API calls
GROQ_API_KEY=

# Optional: enable LangSmith tracing in the relevant framework lessons
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=rag-engineering-academy
LANGSMITH_TRACING=false

# Optional future provider keys — leave blank until needed
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# Application defaults
LLM_PROVIDER=groq
CHROMA_PERSIST_DIRECTORY=./.chroma
```

Rules:
- Create `.env.example` with blank values and commit that file.
- Add `.env`, local Chroma persistence directories, downloaded models, caches, generated datasets, notebook checkpoints, and temporary outputs to `.gitignore` as appropriate.
- Load environment variables through a single shared configuration utility using `python-dotenv`.
- Validate a key only when a notebook actually needs that provider. A missing optional key must not break unrelated notebooks.
- Never print, log, display, serialize, or commit secret values.
- Never hardcode API keys in notebook cells, source files, test fixtures, outputs, or documentation.
- Include a short setup cell or README instruction showing the user how to populate `.env` locally.
- If a LangSmith integration expects `LANGCHAIN_API_KEY` rather than `LANGSMITH_API_KEY`, verify the current official documentation and document the exact mapping. Do not silently assume the variable name.
- Do not enable tracing by default. It may transmit prompts, retrieved context, and outputs to an external service. Explain privacy implications and require explicit opt-in.
- Keep `.env` local. Do not create or modify shell profiles or system-wide environment variables without permission.

## 5. Curriculum — all 17 tracks

Preserve the following 17-track structure. Do not remove, merge, or rename tracks without asking.

### Track 0 — Foundations and Environment
- What RAG is and why it exists.
- LLM limitations, hallucinations, knowledge cutoffs, and context windows.
- Tokens, tokenization, embeddings at a high level.
- Transformer and attention intuition.
- End-to-end RAG architecture.
- Environment setup and reproducibility.

### Track 1 — Ingestion and Chunking
- Document ingestion lifecycle.
- PDF parsing with PyMuPDF.
- Text cleaning and normalization.
- Fixed-size and recursive chunking.
- Sentence and paragraph chunking.
- Semantic chunking.
- Parent-child and sentence-window retrieval.
- Structural chunking for headings, tables, and code.
- Late chunking and chunking evaluation.

### Track 2 — Embeddings and Vector Search
- Embedding intuition and vector spaces.
- Cosine similarity, dot product, and Euclidean distance.
- Implement similarity search with NumPy.
- Embedding model selection and limitations.
- ChromaDB collections, persistence, metadata, and querying.
- HNSW and approximate nearest-neighbor concepts.
- Embedding versioning and migration considerations.

### Track 3 — Retrieval Engineering
- Keyword retrieval and BM25.
- Dense retrieval.
- Hybrid retrieval.
- Reciprocal Rank Fusion (RRF).
- Cross-encoder reranking.
- HyDE.
- Maximum Marginal Relevance (MMR).
- Contextual retrieval and contextual compression.
- Query rewriting, expansion, multi-query, decomposition, and routing.
- Advanced retrieval: RAPTOR, hierarchical, multi-vector, late-interaction, and graph-enhanced patterns where appropriate.
- Retrieval evaluation and failure analysis.

### Track 4 — RAG from Scratch
- Build ingestion, chunking, embeddings interface, retrieval, prompt assembly, and generation using plain Python.
- Implement a minimal end-to-end pipeline without LangChain or LlamaIndex abstractions.
- Add citations and source metadata.
- Explore failure cases and evaluation.
- Compare the implementation with later framework-based versions.

### Track 5 — LangChain RAG
- Core abstractions and components.
- Document loaders and splitters.
- Embedding and vector-store integrations.
- Retrievers and prompt templates.
- LCEL.
- Runnable composition and structured outputs.
- Build and evaluate complete LangChain RAG pipelines.
- Verify current APIs; avoid deprecated patterns.

### Track 6 — LangGraph RAG
- Graph state, nodes, edges, and conditional routing.
- Build a basic retrieval graph.
- Persistence and checkpointing.
- Human-in-the-loop patterns.
- Corrective RAG (CRAG).
- Self-RAG concepts.
- Adaptive RAG.
- Retry, fallback, and graph debugging.
- Verify current APIs and persistence behavior.

### Track 7 — Agentic RAG and DeepAgent
- RAG versus agentic RAG.
- Tool calling and tool selection.
- Query planning and decomposition.
- Multi-step retrieval.
- Planning, execution, and verification.
- Agent state and controlled loops.
- Multi-agent patterns and their trade-offs.
- DeepAgent concepts and implementation, based on verified current documentation.
- Tool permissions, budgets, stop conditions, and loop prevention.
- Agent evaluation and failure analysis.

### Track 8 — Memory Engineering
- Conversation state versus external knowledge retrieval.
- Short-term and working memory.
- Semantic, episodic, and temporal memory.
- Memory extraction, consolidation, retrieval, and forgetting policies.
- User/session isolation and privacy.
- Memory evaluation and failure cases.

### Track 9 — GraphRAG and Knowledge Graphs
- Entities, relationships, and graph representations.
- Entity and relation extraction.
- Neo4j fundamentals and Cypher.
- Community detection and graph summaries.
- Graph traversal and graph retrieval.
- Hybrid graph/vector retrieval.
- GraphRAG architecture and evaluation.
- Explain where graph retrieval adds value over vector retrieval.

### Track 10 — Multimodal RAG
Keep modality-specific notebooks within this track:
- Images and image-text embeddings.
- Tables, charts, and document layout.
- Visual document retrieval and approaches such as ColPali, after verifying current implementations.
- Audio ingestion and audio retrieval, including speech and non-speech signals where supported.
- Video ingestion, temporal segmentation, frame sampling, and audio-video alignment.
- Multimodal indexing and retrieval.
- Fusion of text, image, audio, and video evidence.
- Evaluation and limitations for multimodal retrieval.

Do not treat audio RAG as only transcription search or video RAG as only frame extraction. Explain the information lost by such simplifications.

### Track 11 — Google Open Knowledge Format (OKF)
- Verify the canonical Google OKF specification and current terminology first.
- Explain the format's goals, structure, metadata, relationships, and provenance as specified by authoritative sources.
- Create and inspect minimal valid examples based on the actual specification.
- Explore validation, portability, trust, and interoperability.
- Demonstrate how OKF can fit into a knowledge/RAG pipeline without claiming it is itself a vector database or a knowledge graph unless the specification supports that claim.
- Do not invent fields, schemas, or compatibility claims.

### Track 12 — TypeSafe Jev Decision Engine
- Verify official TypeSafe Jev documentation, API, terminology, and examples before implementation.
- Explain the actual semantics of Choice, Score, Noul, and other primitives only as documented.
- Demonstrate decision gates, typed outcomes, confidence, and failure handling based on verified behavior.
- Compare a Jev-based decision flow with ordinary conditional logic where useful.
- Treat latency figures as measured results, not guarantees.
- Do not fabricate APIs, capabilities, or performance claims.

### Track 13 — Decision Intelligence with RAG
- Combine evidence retrieval with structured decision workflows.
- Integrate OKF, Jev, and LangGraph only where their actual capabilities fit.
- Build evidence-backed decisions with provenance and explainability.
- Add confidence, abstention, escalation, and human approval.
- Demonstrate decision auditing and evaluation.
- Make clear the difference between retrieving information and making a decision.

### Track 14 — Enterprise RAG
- Asynchronous ingestion and processing.
- Scaling, batching, and sharding concepts.
- Multi-tenancy and tenant isolation.
- Access control and document-level ACL filtering.
- Caching and cost optimization.
- Document updates, deletion, and re-indexing.
- Embedding model migrations.
- Security: prompt injection in retrieved content, data leakage, poisoning, PII, audit logs, and secure tool access.
- Reliability, resilience, and operational trade-offs.

### Track 15 — Evaluation and Observability
- Build a small, fixed, human-reviewed evaluation set only when needed; do not generate a large synthetic dataset.
- Retrieval relevance and context precision.
- Context recall.
- Answer relevance.
- Faithfulness and groundedness.
- Factual correctness.
- Citation accuracy.
- Completeness.
- Latency.
- Cost and token usage.
- Compare automated metrics with human review.
- Explore Arize Phoenix, OpenTelemetry, and LangSmith in separate, optional lessons.
- Explain tracing privacy and external data transmission.
- Make evaluations reproducible and avoid fabricated scores.

### Track 16 — Fine-Tuning versus RAG and Deployment
- When to use RAG, fine-tuning, or both.
- Controlled comparison experiments.
- Embedding fine-tuning concepts where relevant.
- Hybrid architectures.
- Build a small FastAPI interface.
- Packaging and Docker only when the deployment lesson requires it.
- CI/CD and notebook validation.
- Resilience, observability, security, and release readiness.
- Document what is locally tested versus externally deployed.

## 6. Repository structure

Create a maintainable structure similar to the following. Adjust only when there is a clear reason and document the reason.

```text
rag-engineering-academy/
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── .env.example
├── .env                         # local only; never commit
├── requirements/
│   ├── base.txt
│   ├── track-00.txt
│   ├── track-01.txt
│   └── ...                      # install only what a track needs
├── assets/
│   ├── architecture/
│   ├── diagrams/
│   ├── animations/
│   └── illustrations/
├── common/
│   ├── config.py
│   ├── logging.py
│   ├── visualization.py
│   ├── evaluation.py
│   ├── datasets.py
│   └── utilities.py
├── fixtures/
│   └── README.md                # minimal shared fixtures only
├── tracks/
│   ├── 00-foundations/
│   ├── 01-ingestion-chunking/
│   ├── 02-embeddings-vector-search/
│   ├── 03-retrieval-engineering/
│   ├── 04-rag-from-scratch/
│   ├── 05-langchain-rag/
│   ├── 06-langgraph-rag/
│   ├── 07-agentic-rag/
│   ├── 08-memory-engineering/
│   ├── 09-graphrag/
│   ├── 10-multimodal-rag/
│   ├── 11-google-okf/
│   ├── 12-typesafe-jev/
│   ├── 13-decision-intelligence/
│   ├── 14-enterprise-rag/
│   ├── 15-evaluation-observability/
│   └── 16-finetuning-deployment/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── notebook_validation/
├── docs/
│   ├── curriculum-map.md
│   ├── architecture.md
│   ├── setup-guide.md
│   ├── glossary.md
│   └── cost-and-data-policy.md
├── curriculum-manifest.yaml
└── PROJECT_STATUS.md
```

Do not create empty placeholder files or directories unless they serve a clear purpose. Do not create all notebook files as empty stubs. Create each notebook when its track/stage is approved for implementation.

# E — EXPECTATIONS

## 1. Stage 0 — discovery and initialization

Start by inspecting the environment. Do not assume the repository, workspace, Python version, operating system, Git, Jupyter, or hardware.

Inspect:
- Current working directory and available disk space.
- Operating system and Python version.
- Git availability and configuration.
- Jupyter availability.
- Existing project files, if any.
- Whether a virtual environment already exists.
- Whether the machine can run CPU-friendly examples.

If the intended project directory is unclear, ask the user where to create it. Do not create the project in an arbitrary location.

Then:
1. Propose the repository location and summarize the setup plan.
2. Create or initialize the Git repository only after the location is clear.
3. Create a Python virtual environment.
4. Establish a minimal base dependency set.
5. Add track-specific dependency files; do not install all future-track packages now.
6. Add `.gitignore`, `.env.example`, and local `.env` handling.
7. Create the repository structure and curriculum manifest.
8. Create `README.md`, setup instructions, architecture overview, cost/data policy, and `PROJECT_STATUS.md`.
9. Add minimal shared utilities only when needed.
10. Add a lightweight notebook smoke-test/validation approach.
11. Verify ChromaDB, PyMuPDF, and the basic Jupyter environment with small local checks. Do not download embedding models or call Groq during setup.
12. Confirm `.env` is ignored by Git and that no secrets are present in tracked files.
13. Run setup checks and report the results.

If the user has not yet supplied a Groq API key locally, leave the key blank and explain how they can add it to `.env`. Do not block Stage 0 on the key.

**Stage 0 completion report must include:**
- Repository path.
- Python and environment details.
- Files/directories created.
- Dependencies installed and why.
- `.env` and `.gitignore` status.
- Checks executed and their results.
- Any unresolved decisions or blockers.

Then stop and ask for approval to begin Stage 1.

## 2. Notebook design standard

### Mandatory Markdown-before-code-cell rule — non-negotiable

**Every executable code cell must have an explanatory Markdown cell immediately before it.** This applies to every notebook, including setup cells, imports, configuration, helper functions, experiments, visualizations, tests, and optional API examples.

The Markdown cell must make the following clear, as relevant to that code cell:
- **What we are about to do** — the task or operation performed by the code.
- **Why we are doing it** — how it supports the lesson or the RAG pipeline.
- **How it works** — the key idea, algorithm, API, or important parameters, explained at the learner's level.
- **What to expect** — the expected output, observable behavior, or how to interpret the result.
- **Caveats** — whether the cell needs an API key, network access, a download, extra compute, or incurs API/token cost.

Do not use one generic Markdown explanation for a long sequence of unrelated code cells. If the next code cell performs a materially different operation, add a new Markdown explanation immediately before it. A single Markdown cell may introduce a short, logically cohesive group of code cells only when it clearly explains each operation in that group; the safest default is one explanatory Markdown cell per code cell.

Additional rules:
- Never place an executable code cell before the notebook's explanatory Markdown introduction.
- Do not leave consecutive code cells without an intervening explanatory Markdown cell.
- Markdown headings alone (for example, `### Implementation`) do not satisfy this requirement; include a short explanation of the code that follows.
- For a code cell that is intentionally trivial, still provide a concise explanation rather than omitting the Markdown cell.
- If a code cell is optional or expensive, label it clearly in Markdown before the code cell and explain how to skip it.
- After meaningful code/output, add a Markdown interpretation when the output is not self-explanatory or when the learner needs help understanding its significance.
- Keep explanations technically accurate, concise, beginner-friendly, and useful to someone reading the notebook without an instructor.
- Before marking a notebook `REVIEWED`, inspect the cell sequence programmatically or manually and verify that **100% of code cells have an immediately preceding explanatory Markdown cell**. Report this check in the notebook validation results.

### Notebook readability and teaching quality

Treat each notebook as a small, polished technical chapter—not as a scratchpad or an unannotated script.
- Use a clear progression: concept → intuition → diagram (where useful) → implementation → experiment → interpretation → limitations → exercise.
- Use descriptive Markdown headings and consistent terminology.
- Explain unfamiliar variables, data structures, function arguments, and non-obvious implementation decisions.
- Prefer small, focused code cells over very long cells. Avoid splitting code into many tiny cells without a learning reason.
- Keep outputs concise and useful. Clear stale, noisy, or excessively large outputs before committing.
- Use meaningful names; avoid unexplained abbreviations and magic numbers.
- Include expected behavior and troubleshooting notes where setup or library behavior may confuse learners.
- Ensure a reader can understand the notebook in order without needing to infer why a cell exists.
- Do not add filler prose. Every Markdown and code cell must contribute to understanding, implementation, or validation.

Every notebook must be independently understandable and executable from a fresh kernel.

Use a consistent structure:
1. Title and short orientation.
2. Learning objectives.
3. Prerequisites and expected runtime.
4. The real-world problem.
5. First-principles explanation.
6. Architecture or process diagram.
7. Mathematical intuition where relevant.
8. A minimal implementation from scratch where pedagogically appropriate.
9. Library-based implementation where relevant.
10. Small experiment with a minimal fixture.
11. Interpretation of outputs.
12. Failure modes and limitations.
13. Assertions or lightweight evaluation.
14. Hands-on exercise.
15. Interview/check-your-understanding questions.
16. References to authoritative documentation.
17. Summary and next notebook.

Do not pad notebooks with repetitive prose. Each cell must have a learning or implementation purpose.

## 3. Visual teaching requirements

Use meaningful visuals to explain concepts, not decoration.

Suitable tools include:
- Mermaid for architecture and sequence diagrams.
- Matplotlib and Plotly for plots and experiments.
- NetworkX for graph concepts.
- Graphviz or SVG for architecture illustrations.
- Manim for selected mathematical or algorithmic animations when justified.

Visualize concepts such as:
- Tokenization and context windows.
- Chunk boundaries and overlap.
- Embedding spaces and similarity.
- BM25 versus dense retrieval.
- Hybrid search and RRF.
- Reranking.
- HNSW intuition.
- LangChain composition and LangGraph state transitions.
- Agent planning/tool loops and guardrails.
- Memory lifecycle.
- GraphRAG traversal.
- Multimodal retrieval and fusion.
- OKF structure.
- Jev decision flows.
- Enterprise security and evaluation.

Rules:
- Prefer diagrams generated locally from code over expensive external image-generation calls.
- Do not generate an animation for every notebook. Use animation only when motion materially improves understanding.
- Reuse shared visual assets where appropriate.
- Keep diagrams labeled, readable, and technically accurate.
- Never use a diagram as a substitute for explaining the underlying mechanism.
- Do not download visual assets without checking their license and documenting attribution.

## 4. Technical correctness and documentation

- Verify current library APIs against official documentation before implementing them.
- Prefer official documentation, specifications, and maintained repositories over blogs.
- Check package names, imports, version compatibility, model identifiers, and deprecation status.
- Pin or constrain versions when compatibility requires it; explain the choice.
- Do not invent API methods, configuration fields, output formats, or performance claims.
- If documentation is ambiguous, record the uncertainty and ask rather than guessing.
- Cite or link references close to the relevant notebook content.
- Clearly distinguish conceptual pseudocode, mock implementations, local execution, and real external service calls.

## 5. Testing and validation

Testing must be proportionate. Do not build an elaborate test-data factory or generate a large test corpus.

For every notebook:
1. Validate notebook JSON and cell structure.
2. Execute all default cells from a fresh kernel where practical.
3. Ensure default execution does not require secrets or network access.
4. Use tiny deterministic fixtures and mocks for unit tests.
5. Check that outputs are not stale, misleading, or fabricated.
6. Inspect diagrams and plots for correctness.
7. Run relevant unit tests.
8. Run integration tests only when the relevant local component is available.
9. Mark external API tests as optional and skip them by default.
10. Record any cells that require manual interaction or optional downloads.

Use assertions to validate important behavior. Do not add tests merely to inflate test counts. Never report a test as passed if it was skipped or not run.

## 6. Notebook production protocol

For each notebook, follow this exact workflow:

1. Read the curriculum manifest and track README.
2. Check prerequisites and the previous notebook's learning outcomes.
3. Verify the current official documentation for the APIs being taught.
4. Define the learning objective and minimum required fixture.
5. Plan the explanation, code, experiment, and visual.
6. Create the notebook.
7. Execute default cells from a fresh kernel.
8. Run relevant lightweight tests.
9. Inspect outputs, warnings, plots, and diagrams.
10. Fix errors and improve unclear explanations.
11. Update references, track README, manifest, and status.
12. Review the notebook before moving to the next one.

Do not generate an entire track's notebooks in one huge operation. Work in small, reviewable batches.

## 7. Stage breakdown and acceptance criteria

### Stage 1 — Tracks 0–4: Foundations

Build the foundational tracks:
- Track 0: Foundations and Environment.
- Track 1: Ingestion and Chunking.
- Track 2: Embeddings and Vector Search.
- Track 3: Retrieval Engineering.
- Track 4: RAG from Scratch.

Suggested notebook numbering:
- Track 0: `00.1` onward.
- Track 1: `01.1` onward.
- Track 2: `02.1` onward.
- Track 3: `03.1` onward.
- Track 4: `04.1` onward.

Acceptance:
- Learner can explain the complete RAG pipeline.
- Learner can parse a PDF using PyMuPDF.
- Learner understands chunking and embedding trade-offs.
- Learner can implement basic similarity search.
- Learner can build a basic RAG pipeline without LangChain or LlamaIndex abstractions.
- Default notebooks run without API keys and without downloading large assets.

Report progress and wait for approval before Stage 2.

### Stage 2 — Tracks 5–13: Frameworks and intelligent systems

Build in four manageable batches:

**Stage 2A**
- Track 5: LangChain RAG.
- Track 6: LangGraph RAG.

**Stage 2B**
- Track 7: Agentic RAG and DeepAgent.
- Track 8: Memory Engineering.

**Stage 2C**
- Track 9: GraphRAG and Knowledge Graphs.
- Track 10: Multimodal RAG.

**Stage 2D**
- Track 11: Google OKF.
- Track 12: TypeSafe Jev.
- Track 13: Decision Intelligence with RAG.

Acceptance:
- Learner can explain framework abstractions and their trade-offs.
- Learner can construct graph-based retrieval workflows.
- Agentic workflows have explicit tool permissions, budgets, and stop conditions.
- Learner can distinguish memory from knowledge retrieval.
- Learner understands graph and multimodal retrieval trade-offs.
- OKF and Jev examples match verified official specifications and APIs.
- Decision workflows retain evidence, provenance, and human escalation paths.

At each batch boundary, report progress and wait for approval before the next batch.

### Stage 3 — Tracks 14–16: Enterprise and deployment

Build:
- Track 14: Enterprise RAG.
- Track 15: Evaluation and Observability.
- Track 16: Fine-Tuning versus RAG and Deployment.

Acceptance:
- Demonstrate tenant isolation and ACL filtering with tests.
- Demonstrate secure handling of retrieved prompt-injection content.
- Demonstrate document update/deletion and re-indexing behavior.
- Provide reproducible evaluation methodology and clearly sourced results.
- Demonstrate optional Phoenix/LangSmith tracing with privacy warnings.
- Build a small API/deployment example.
- Validate CI workflows.
- Document what is actually deployed versus only demonstrated locally.

Do not claim production readiness merely because notebooks execute.

## 8. Reporting and communication protocol

Antigravity must provide concise, evidence-based progress reports throughout the project. Do not wait until the end of a stage to communicate important results or blockers.

### Before starting a stage, track, batch, or notebook
- State the objective and the exact scope of the work.
- Identify prerequisites and the files likely to be created or changed.
- Identify any external API calls, downloads, new dependencies, or potentially expensive operations.
- Ask for approval when the master workflow requires a stage/batch approval or when an action is outside the approved scope.

### After completing each notebook
Report:
- Notebook path and title.
- Concepts and implementations covered.
- Whether the notebook was executed from a fresh kernel.
- Whether all default cells ran without secrets or network access.
- The Markdown-before-every-code-cell audit result (number of code cells checked and number with an immediately preceding explanatory Markdown cell).
- Tests/checks passed, failed, or skipped.
- Visuals/diagrams added.
- Dependencies, downloads, or optional API calls introduced.
- Known limitations and follow-up work.

### After completing each track or approved batch
Provide a compact engineering report with:
- Deliverables created or updated.
- Implementation and learning outcomes achieved.
- Notebook execution and validation summary.
- Test counts: passed, failed, skipped, and not run (never combine these categories).
- Markdown/code-cell audit status.
- Git commit(s) created and push status, if applicable.
- Dependencies and infrastructure added, with reasons.
- Any measured API/token usage; write "not measured" rather than estimating.
- Remaining defects, risks, and decisions requiring the user's input.
- Updated curriculum-manifest.yaml and PROJECT_STATUS.md references.

### Reporting integrity
- Report only actions actually performed and results actually observed.
- Distinguish `PASS`, `FAIL`, `SKIPPED`, and `NOT RUN`.
- Never describe generated files as executed, validated, or reviewed unless those steps occurred.
- Never invent test results, token usage, costs, latency, benchmark scores, or completion percentages.
- If execution is blocked by a missing key, dependency, network, or unsupported environment, state the blocker and what remains unverified.
- Keep reports concise but sufficiently detailed for the user to make the next approval decision.

## 8. Progress tracking

Maintain `curriculum-manifest.yaml` and `PROJECT_STATUS.md`.

Use these statuses:
- `NOT_STARTED`
- `PLANNED`
- `GENERATED`
- `IMPLEMENTED`
- `EXECUTED`
- `VALIDATED`
- `REVIEWED`
- `APPROVED`

A file existing does not mean it is complete. Track separately:
- Technical correctness.
- Successful execution.
- Educational quality.
- Documentation/references.
- Visual quality.
- Test status.

For every batch, report:
- Notebooks created.
- Notebooks executed.
- Tests passed, failed, and skipped.
- Defects found and fixed.
- Optional downloads/API calls required.
- Dependencies added and why.
- Token/API usage if any was measured.
- Remaining gaps and decisions needed.

## 10. Git and GitHub workflow

Git is part of the normal development workflow, not a final step to be postponed until all 17 tracks are complete. Keep the repository usable and synchronized in small, reviewable milestones.

### Account and remote safety
- Use the user's **personal GitHub account and personal repository**, not a university, employer, organization, or BITS Pilani account/repository.
- Inspect `git status`, the current branch, configured remotes, and repository-local Git identity before making Git changes.
- If this project is already a Git repository, preserve its history, branch, and valid `origin`; do not reinitialize it.
- Never guess a GitHub username, repository URL, account, or email address. Read the existing local configuration and ask if the intended personal account/remote is ambiguous.
- Prefer repository-local `user.name` and `user.email` configuration when a change is needed. Do not change global Git configuration or credential-helper settings.
- Do not replace a working personal `origin` with another remote. If `origin` points to an institutional/work account or an unexpected repository, stop and ask before changing it.
- Do not print, expose, copy, or store GitHub tokens, passwords, SSH private keys, or other credentials in notebooks, Markdown, logs, or reports. Use the user's already-configured Git credential manager or SSH authentication.

### Incremental commit workflow
- Commit completed, validated milestones as work progresses; do not wait until the entire curriculum is finished.
- A notebook is ready for a normal milestone commit only after its explanations, code, outputs, references, and validation have been reviewed.
- Before each commit, inspect the diff and staged file list. Stage only the intended project files.
- Use small, descriptive commit messages, for example:
  - `docs: add RAG curriculum and project setup`
  - `feat(track-00): add RAG foundations notebook`
  - `feat(track-01): add PDF parsing and chunking lessons`
  - `test: add notebook structure validation`
- Update PROJECT_STATUS.md and curriculum-manifest.yaml in the same milestone when their status information changes.
- Do not create meaningless commits for every tiny edit. Group logically related, reviewed work.
- Never rewrite published history, force-push, delete branches, or run destructive Git commands unless the user explicitly requests the exact action.

### Push and GitHub synchronization
- The intended workflow is to keep finished milestones available in the user's personal GitHub repository as the project progresses.
- If a valid personal `origin` is already configured and the user has authorized ongoing synchronization for this project, push completed milestone commits to the current project branch after local validation.
- If push authorization is not clear, prepare the commit locally and ask before the first push; do not assume that permission to edit files also grants permission to publish them.
- Never push secrets, `.env`, private documents, unlicensed assets, large datasets, model weights, local vector stores, or temporary files.
- Before pushing, verify the branch and remote destination, inspect the outgoing commits, and confirm that `.env` and other ignored/private files are not staged or tracked.
- After a push, report the branch and commit hash and confirm success only if Git reports success. If authentication or network access fails, report the exact blocker without changing credentials.

### Repository hygiene
- Never commit `.env`, API keys, personal documents, private datasets, or generated caches.
- Keep notebook outputs small; clear bulky outputs where appropriate without removing useful teaching evidence.
- Avoid committing large model files, vector-store directories, or datasets.
- Document any asset license and attribution.
- Do not overwrite existing user files without inspecting and preserving them.
- Check `.gitignore` and tracked-file status periodically; adding a file to `.gitignore` does not automatically untrack a file already committed.


## 11. Final release criteria

The academy is ready for a release review only when:
- All 17 tracks are implemented and reviewed.
- Notebook execution status is recorded.
- Track READMEs and the curriculum map are complete.
- Diagrams and teaching assets are present and accurate.
- Exercises and solutions or evaluation guidance are included.
- Tests and CI checks are documented.
- Dependency and setup instructions are reproducible.
- Security, privacy, limitations, and cost considerations are documented.
- Evaluation results are reproducible and not fabricated.
- A final curriculum audit identifies incomplete or optional areas.

Provide a final report distinguishing:
- Fully implemented and locally validated.
- Implemented but dependent on optional external services.
- Demonstrated with mocks or simulations.
- Not implemented or requiring real deployment.

---

# Immediate instruction to Antigravity

**Begin with Stage 0 only.**

First inspect the environment and determine whether the target project directory is clear. If it is not clear, ask the user for the directory. Do not generate datasets, download models, call Groq, or start building curriculum notebooks during Stage 0.

Initialize or inspect the existing repository without destroying its history, preferred baseline stack, `.env.example`, local `.env` handling, `.gitignore`, curriculum manifest, documentation, and lightweight validation setup. Use the user's personal GitHub repository/account rather than any institutional account. Preserve a valid existing personal remote, inspect Git identity locally, and establish an incremental commit workflow. Do not change global Git settings or publish to an ambiguous remote. Verify that secrets are excluded from Git. Run only small local smoke checks. For every notebook created in later stages, enforce the mandatory Markdown explanation immediately before every code cell and audit this rule before marking the notebook reviewed.

Then provide the Stage 0 completion report and **wait for explicit user approval** before creating Track 0 notebooks.

The user will provide the Groq API key directly in the local `.env` file. Never ask them to reveal it in chat.
