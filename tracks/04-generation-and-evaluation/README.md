# Track 04 — Generation, Synthesis, and Automated RAG Evaluation (RAGAS / TruLens)

Generation and synthesis represent the second half of the RAG pipeline. This track covers grounded response synthesis, epistemic refusal handling, citation attribution, and automated quantitative evaluation using the RAG Triad (TruLens) and RAGAS evaluation paradigms.

## Notebook Curriculum

| Notebook | Title | Status | Description |
|---|---|---|---|
| [**04.1**](file:///Users/bharathm/Documents/Self%20Learning/RAG_from_scratch/tracks/04-generation-and-evaluation/01_rag_evaluation_framework.ipynb) | **Automated RAG Evaluation Framework (RAGAS & TruLens)** | **VALIDATED** | Complete offline-first implementation of Context Precision (MAP@K), Context Recall, Faithfulness (hallucination detection via atomic proposition decomposition), Answer Relevance, and the Harmonic RAG Triad composite score. Includes grounded synthesis prompts, citation audits, and diagnostic failure mode classification. |

## Architectural Overview: The RAG Triad

```mermaid
flowchart TD
    UserQuery["User Query"] --> ContextRetrieved["Retrieved Context Chunks"]
    ContextRetrieved --> Generator["Grounded Generator / LLM"]
    Generator --> AnswerGenerated["Generated Answer + Inline Citations"]
    
    subgraph RAG_Triad ["The RAG Triad Evaluation Pillars"]
        direction TB
        Pillar1["1. Context Precision / Relevance\n(Query <-> Retrieved Context)\nIs the context noise-free and relevant?"]
        Pillar2["2. Faithfulness / Groundedness\n(Context <-> Generated Answer)\nAre all claims supported by the context?"]
        Pillar3["3. Answer Relevance\n(Query <-> Generated Answer)\nDoes the answer address the question directly?"]
    end
    
    UserQuery -.-> Pillar1
    ContextRetrieved -.-> Pillar1
    
    ContextRetrieved -.-> Pillar2
    AnswerGenerated -.-> Pillar2
    
    UserQuery -.-> Pillar3
    AnswerGenerated -.-> Pillar3
    
    Pillar1 --> TriadScore["Composite Harmonic Triad Score\nDiagnostic Root-Cause Analysis"]
    Pillar2 --> TriadScore
    Pillar3 --> TriadScore

    classDef stage fill:#f8fafc,stroke:#3b82f6,stroke-width:1.5px,color:#1e293b;
    class UserQuery,ContextRetrieved,Generator,AnswerGenerated,Pillar1,Pillar2,Pillar3,TriadScore stage;
```

## Generated Visual Assets

- [RAG Triad & RAGAS Metric Fingerprints Radar Chart](file:///Users/bharathm/Documents/Self%20Learning/RAG_from_scratch/assets/diagrams/track04_evaluation_radar.png)
- [Context Precision Rank-Order Decay Curve](file:///Users/bharathm/Documents/Self%20Learning/RAG_from_scratch/assets/diagrams/track04_context_precision_decay.png)
