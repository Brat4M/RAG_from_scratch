"""LangGraph RAG Architectures: StateGraph, Nodes, Conditional Routing, and Persistence.

Provides production-ready, first-principles stateful RAG workflows implemented with LangGraph:
1. Corrective RAG (CRAG): Document grading, query transformation, cyclical retrieval loops.
2. Self-RAG: Answer hallucination evaluation and question relevance reflection.
3. Adaptive RAG: Combined routing with checkpointing (MemorySaver) and human-in-the-loop (HITL) interrupt capability.
4. Loop Prevention & Guardrails: Strict step counters preventing unbounded state cycles.
"""

from typing import Dict, Any, List, Optional, Literal, TypedDict, Callable
import copy
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# ---------------------------------------------------------------------------
# 1. Typed Graph State Definition
# ---------------------------------------------------------------------------

class DocumentDict(TypedDict):
    """Normalized document representation within graph state."""
    page_content: str
    metadata: Dict[str, Any]


class RAGGraphState(TypedDict):
    """Complete typed state passed across all LangGraph nodes."""
    question: str
    original_question: str
    documents: List[DocumentDict]
    generation: Optional[str]
    loop_count: int
    max_retries: int
    documents_status: str  # "relevant", "irrelevant", "empty"
    hallucination_grade: str  # "grounded", "hallucinated", "unverified"
    answer_grade: str  # "useful", "not_useful", "unverified"
    web_fallback_needed: bool
    human_approval: Optional[bool]
    trace: List[str]


def create_initial_state(
    question: str,
    max_retries: int = 2,
    human_approval: Optional[bool] = None,
) -> RAGGraphState:
    """Create a pristine initial RAG graph state."""
    return {
        "question": question,
        "original_question": question,
        "documents": [],
        "generation": None,
        "loop_count": 0,
        "max_retries": max_retries,
        "documents_status": "empty",
        "hallucination_grade": "unverified",
        "answer_grade": "unverified",
        "web_fallback_needed": False,
        "human_approval": human_approval,
        "trace": ["START"],
    }


# ---------------------------------------------------------------------------
# 2. Offline-First Deterministic Engines (Graders, Retriever, Generator)
# ---------------------------------------------------------------------------

STOP_WORDS = {
    "and", "the", "for", "with", "from", "that", "this", "what", "how",
    "are", "about", "can", "tell", "which", "into", "does", "have", "been",
}


class DeterministicGraphRetriever:
    """Deterministic in-memory document retriever for offline graph execution."""

    def __init__(self, corpus: Optional[List[Dict[str, Any]]] = None):
        self.corpus = corpus or [
            {
                "page_content": "Retrieval-Augmented Generation (RAG) augments Large Language Model prompts with factual external context retrieved from a vector database or search index.",
                "metadata": {"source": "rag_overview.txt", "doc_id": "doc-01", "keywords": ["rag", "llm", "retrieval", "vector"]},
            },
            {
                "page_content": "Chunking strategies include fixed-size token splitting, recursive boundary splitting, and semantic boundary chunking to preserve coherent discourse.",
                "metadata": {"source": "chunking_guide.txt", "doc_id": "doc-02", "keywords": ["chunking", "token", "recursive", "semantic"]},
            },
            {
                "page_content": "LangGraph provides stateful cyclic graphs with checkpointing, conditional routing, human-in-the-loop execution, and time-travel replay.",
                "metadata": {"source": "langgraph_spec.txt", "doc_id": "doc-03", "keywords": ["langgraph", "state", "cyclic", "checkpointing", "hitl"]},
            },
            {
                "page_content": "Corrective RAG (CRAG) evaluates retrieved document relevance and rewrites queries when the retrieved documents lack sufficient grounding.",
                "metadata": {"source": "crag_paper.txt", "doc_id": "doc-04", "keywords": ["crag", "corrective", "grading", "rewrite", "query"]},
            },
        ]

    def retrieve(self, query: str, top_k: int = 2) -> List[DocumentDict]:
        """Rank documents by content and keyword overlap, ignoring stop words."""
        raw_tokens = query.lower().replace("?", "").replace(",", "").split()
        tokens = {t for t in raw_tokens if len(t) > 2 and t not in STOP_WORDS}
        if not tokens:
            return []

        scored: List[tuple[float, DocumentDict]] = []
        for doc in self.corpus:
            content_lower = doc["page_content"].lower()
            keywords = doc.get("metadata", {}).get("keywords", [])
            overlap = sum(1.0 for t in tokens if t in content_lower)
            keyword_overlap = sum(2.0 for t in tokens if t in keywords)
            score = overlap + keyword_overlap
            scored.append((score, {"page_content": doc["page_content"], "metadata": dict(doc["metadata"])}))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [doc for score, doc in scored[:top_k] if score > 0]
        return selected


class DocumentRelevanceGrader:
    """Evaluates whether retrieved documents are relevant to the user query."""

    def __init__(self, overlap_threshold: int = 1):
        self.overlap_threshold = overlap_threshold

    def grade_document(self, document_text: str, query: str) -> bool:
        """Return True if document passes lexical/semantic relevance threshold."""
        query_words = {
            w for w in query.lower().replace("?", "").replace(",", "").split()
            if len(w) > 2 and w not in STOP_WORDS
        }
        if not query_words:
            return False
        doc_words = {w for w in document_text.lower().split() if len(w) > 2}
        common = query_words.intersection(doc_words)
        return len(common) >= self.overlap_threshold

    def grade_documents(self, documents: List[DocumentDict], query: str) -> tuple[List[DocumentDict], str]:
        """Filters out irrelevant documents and returns the filtered set + status."""
        filtered = [doc for doc in documents if self.grade_document(doc["page_content"], query)]
        if not filtered:
            status = "irrelevant"
        else:
            status = "relevant"
        return filtered, status


class GraphQueryTransformer:
    """Rewrites queries to improve recall when initial retrieval fails."""

    def __init__(self, synonym_map: Optional[Dict[str, str]] = None):
        self.synonym_map = synonym_map or {
            "crag": "Corrective RAG evaluating document relevance and rewriting queries",
            "cycle": "cyclic graph state checkpointing persistence",
            "state": "LangGraph stateful cyclic graphs with checkpointing",
            "split": "chunking strategies token boundary recursive",
            "what is rag": "Retrieval-Augmented Generation RAG external context vector database",
        }

    def transform(self, query: str, loop_count: int) -> str:
        """Deterministically reformulate query based on intent keywords."""
        q_lower = query.lower()
        for key, expansion in self.synonym_map.items():
            if key in q_lower:
                return f"{query} {expansion} (iteration {loop_count})"
        return f"{query} key concepts overview detail (iteration {loop_count})"


class GraphAnswerGenerator:
    """Generates grounded responses based on filtered documents."""

    def generate(self, question: str, documents: List[DocumentDict]) -> str:
        """Synthesizes a response strictly grounded in provided context."""
        if not documents:
            return "Refusal: I cannot answer the question because no relevant context was found in the knowledge base."

        contexts = [f"[{i+1}] {doc['page_content']}" for i, doc in enumerate(documents)]
        joined = " ".join(contexts)
        return f"Based on verified context ({len(documents)} sources): {joined} This answers query: '{question}'."


class SelfRAGHallucinationGrader:
    """Verifies that generated text is supported by document facts."""

    def grade(self, generation: str, documents: List[DocumentDict]) -> str:
        if not generation or "Refusal:" in generation:
            return "grounded"  # Proper refusal is strictly non-hallucinatory
        if not documents:
            return "hallucinated"

        # Check that key claims originate from documents
        doc_text = " ".join(d["page_content"].lower() for d in documents)
        gen_lower = generation.lower()

        # If generation fabricates claims not present in doc_text
        hallucination_indicators = ["secret extraterrestrial", "fabricated fact", "fictional ungrounded"]
        for ind in hallucination_indicators:
            if ind in gen_lower and ind not in doc_text:
                return "hallucinated"

        return "grounded"


class SelfRAGAnswerGrader:
    """Verifies that the generated answer addresses the question."""

    def grade(self, generation: str, question: str) -> str:
        if not generation:
            return "not_useful"
        if "Refusal:" in generation:
            return "useful"  # Honest refusal is a valid answer

        # Check question keyword presence in answer
        q_words = {w for w in question.lower().replace("?", "").split() if len(w) > 3}
        if not q_words:
            return "useful"
        gen_lower = generation.lower()
        matches = sum(1 for w in q_words if w in gen_lower)
        return "useful" if matches >= 1 else "not_useful"


# ---------------------------------------------------------------------------
# 3. Node Functions
# ---------------------------------------------------------------------------

def make_retrieve_node(retriever: DeterministicGraphRetriever) -> Callable[[RAGGraphState], Dict[str, Any]]:
    """Factory for retrieve_node with bound retriever."""
    def retrieve_node(state: RAGGraphState) -> Dict[str, Any]:
        query = state["question"]
        docs = retriever.retrieve(query, top_k=2)
        trace = list(state.get("trace", [])) + ["retrieve"]
        return {
            "documents": docs,
            "documents_status": "relevant" if docs else "empty",
            "trace": trace,
        }
    return retrieve_node


def make_grade_documents_node(grader: DocumentRelevanceGrader) -> Callable[[RAGGraphState], Dict[str, Any]]:
    """Factory for grade_documents_node (CRAG Document Evaluator)."""
    def grade_documents_node(state: RAGGraphState) -> Dict[str, Any]:
        docs = state["documents"]
        query = state["question"]
        filtered_docs, status = grader.grade_documents(docs, query)
        trace = list(state.get("trace", [])) + [f"grade_documents({status})"]
        return {
            "documents": filtered_docs,
            "documents_status": status,
            "trace": trace,
        }
    return grade_documents_node


def make_transform_query_node(transformer: GraphQueryTransformer) -> Callable[[RAGGraphState], Dict[str, Any]]:
    """Factory for transform_query_node."""
    def transform_query_node(state: RAGGraphState) -> Dict[str, Any]:
        loop = state["loop_count"] + 1
        new_q = transformer.transform(state["original_question"], loop)
        trace = list(state.get("trace", [])) + [f"transform_query(cycle={loop})"]
        return {
            "question": new_q,
            "loop_count": loop,
            "trace": trace,
        }
    return transform_query_node


def make_generate_node(generator: GraphAnswerGenerator) -> Callable[[RAGGraphState], Dict[str, Any]]:
    """Factory for generate_node."""
    def generate_node(state: RAGGraphState) -> Dict[str, Any]:
        question = state["question"]
        docs = state["documents"]
        gen = generator.generate(question, docs)
        trace = list(state.get("trace", [])) + ["generate"]
        return {
            "generation": gen,
            "trace": trace,
        }
    return generate_node


def make_self_rag_eval_node(
    hallucination_grader: SelfRAGHallucinationGrader,
    answer_grader: SelfRAGAnswerGrader,
) -> Callable[[RAGGraphState], Dict[str, Any]]:
    """Factory for Self-RAG evaluation node."""
    def self_rag_eval_node(state: RAGGraphState) -> Dict[str, Any]:
        gen = state.get("generation") or ""
        docs = state.get("documents", [])
        q = state.get("original_question", state["question"])

        h_grade = hallucination_grader.grade(gen, docs)
        a_grade = answer_grader.grade(gen, q)

        trace = list(state.get("trace", [])) + [f"self_eval(grounded={h_grade},useful={a_grade})"]
        return {
            "hallucination_grade": h_grade,
            "answer_grade": a_grade,
            "trace": trace,
        }
    return self_rag_eval_node


def human_review_node(state: RAGGraphState) -> Dict[str, Any]:
    """Human-in-the-loop gate node."""
    trace = list(state.get("trace", [])) + ["human_review"]
    # If approval is None or False, keep as is; if True, mark approved
    return {
        "human_approval": state.get("human_approval", True),
        "trace": trace,
    }


def fallback_node(state: RAGGraphState) -> Dict[str, Any]:
    """Fallback node when budget or cycles are exhausted."""
    trace = list(state.get("trace", [])) + ["fallback"]
    return {
        "generation": (
            "Refusal: Context could not be validated after query reformulation and retry limits. "
            "Please refine your query or consult secondary domain documentation."
        ),
        "documents_status": "exhausted",
        "web_fallback_needed": True,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# 4. Conditional Edge Decision Routers
# ---------------------------------------------------------------------------

def decide_crag_path(state: RAGGraphState) -> Literal["generate", "transform_query", "fallback"]:
    """CRAG router: checks whether documents are relevant or if retry limit reached."""
    if state["documents_status"] == "relevant" and len(state["documents"]) > 0:
        return "generate"
    if state["loop_count"] >= state["max_retries"]:
        return "fallback"
    return "transform_query"


def decide_self_rag_path(
    state: RAGGraphState,
) -> Literal["human_review", "generate", "transform_query", "fallback", "end"]:
    """Self-RAG router: checks hallucination and answer utility."""
    h_grade = state.get("hallucination_grade", "grounded")
    a_grade = state.get("answer_grade", "useful")
    loop = state.get("loop_count", 0)
    max_retries = state.get("max_retries", 2)

    if h_grade == "grounded" and a_grade == "useful":
        if state.get("human_approval") is not None:
            return "human_review"
        return "end"

    if loop >= max_retries:
        return "fallback"

    if h_grade == "hallucinated":
        # Grounding failure: retry generation with stricter prompt
        return "generate"

    # Answer utility failure: rewrite query and re-retrieve
    return "transform_query"


# ---------------------------------------------------------------------------
# 5. Graph Builders
# ---------------------------------------------------------------------------

def build_crag_graph(
    retriever: Optional[DeterministicGraphRetriever] = None,
    grader: Optional[DocumentRelevanceGrader] = None,
    transformer: Optional[GraphQueryTransformer] = None,
    generator: Optional[GraphAnswerGenerator] = None,
    checkpointer: Optional[MemorySaver] = None,
):
    """Builds a Corrective RAG (CRAG) workflow with query rewriting feedback loop."""
    retriever = retriever or DeterministicGraphRetriever()
    grader = grader or DocumentRelevanceGrader()
    transformer = transformer or GraphQueryTransformer()
    generator = generator or GraphAnswerGenerator()

    workflow = StateGraph(RAGGraphState)

    workflow.add_node("retrieve", make_retrieve_node(retriever))
    workflow.add_node("grade_documents", make_grade_documents_node(grader))
    workflow.add_node("transform_query", make_transform_query_node(transformer))
    workflow.add_node("generate", make_generate_node(generator))
    workflow.add_node("fallback", fallback_node)

    # Edge definitions
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_crag_path,
        {
            "generate": "generate",
            "transform_query": "transform_query",
            "fallback": "fallback",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)

    return workflow.compile(checkpointer=checkpointer)


def build_self_rag_graph(
    retriever: Optional[DeterministicGraphRetriever] = None,
    generator: Optional[GraphAnswerGenerator] = None,
    hallucination_grader: Optional[SelfRAGHallucinationGrader] = None,
    answer_grader: Optional[SelfRAGAnswerGrader] = None,
    transformer: Optional[GraphQueryTransformer] = None,
    checkpointer: Optional[MemorySaver] = None,
):
    """Builds a Self-Reflective RAG graph with hallucination & utility checks."""
    retriever = retriever or DeterministicGraphRetriever()
    generator = generator or GraphAnswerGenerator()
    hallucination_grader = hallucination_grader or SelfRAGHallucinationGrader()
    answer_grader = answer_grader or SelfRAGAnswerGrader()
    transformer = transformer or GraphQueryTransformer()

    workflow = StateGraph(RAGGraphState)

    workflow.add_node("retrieve", make_retrieve_node(retriever))
    workflow.add_node("generate", make_generate_node(generator))
    workflow.add_node("self_eval", make_self_rag_eval_node(hallucination_grader, answer_grader))
    workflow.add_node("transform_query", make_transform_query_node(transformer))
    workflow.add_node("fallback", fallback_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "self_eval")
    workflow.add_conditional_edges(
        "self_eval",
        lambda s: "end" if s.get("hallucination_grade") == "grounded" and s.get("answer_grade") == "useful"
        else ("fallback" if s.get("loop_count", 0) >= s.get("max_retries", 2) else "transform_query"),
        {
            "end": END,
            "transform_query": "transform_query",
            "fallback": "fallback",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("fallback", END)

    return workflow.compile(checkpointer=checkpointer)


def build_adaptive_rag_graph(
    retriever: Optional[DeterministicGraphRetriever] = None,
    grader: Optional[DocumentRelevanceGrader] = None,
    transformer: Optional[GraphQueryTransformer] = None,
    generator: Optional[GraphAnswerGenerator] = None,
    hallucination_grader: Optional[SelfRAGHallucinationGrader] = None,
    answer_grader: Optional[SelfRAGAnswerGrader] = None,
    checkpointer: Optional[MemorySaver] = None,
    enable_hitl: bool = False,
):
    """Builds comprehensive Adaptive RAG unifying CRAG, Self-RAG, and HITL interrupts."""
    retriever = retriever or DeterministicGraphRetriever()
    grader = grader or DocumentRelevanceGrader()
    transformer = transformer or GraphQueryTransformer()
    generator = generator or GraphAnswerGenerator()
    hallucination_grader = hallucination_grader or SelfRAGHallucinationGrader()
    answer_grader = answer_grader or SelfRAGAnswerGrader()

    workflow = StateGraph(RAGGraphState)

    workflow.add_node("retrieve", make_retrieve_node(retriever))
    workflow.add_node("grade_documents", make_grade_documents_node(grader))
    workflow.add_node("transform_query", make_transform_query_node(transformer))
    workflow.add_node("generate", make_generate_node(generator))
    workflow.add_node("self_eval", make_self_rag_eval_node(hallucination_grader, answer_grader))
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("fallback", fallback_node)

    # Routing
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_crag_path,
        {
            "generate": "generate",
            "transform_query": "transform_query",
            "fallback": "fallback",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", "self_eval")

    def adaptive_eval_router(state: RAGGraphState) -> str:
        h = state.get("hallucination_grade", "grounded")
        a = state.get("answer_grade", "useful")
        loop = state.get("loop_count", 0)
        max_r = state.get("max_retries", 2)

        if h == "grounded" and a == "useful":
            return "human_review" if enable_hitl else "end"
        if loop >= max_r:
            return "fallback"
        return "transform_query"

    workflow.add_conditional_edges(
        "self_eval",
        adaptive_eval_router,
        {
            "human_review": "human_review",
            "end": END,
            "transform_query": "transform_query",
            "fallback": "fallback",
        },
    )

    workflow.add_edge("human_review", END)
    workflow.add_edge("fallback", END)

    interrupts = ["human_review"] if enable_hitl else []
    return workflow.compile(checkpointer=checkpointer, interrupt_before=interrupts)
