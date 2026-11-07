"""LangChain RAG Abstractions and First-Principles Parity Framework.

Provides:
1. DeterministicLangChainEmbeddings: Implements LangChain Embeddings protocol
   using our 100% offline CPU-friendly deterministic semantic embedder.
2. DeterministicLangChainChatModel: Subclasses BaseChatModel to provide
   grounded offline synthesis with source citations and epistemic refusals.
3. CitedAnswer: Structured Pydantic output schema for evidence-backed answers.
4. FirstPrinciplesRAGPipeline: Lightweight, zero-framework baseline RAG pipeline.
5. LangChainLCELPipeline: Idiomatic LCEL (LangChain Expression Language) pipeline
   demonstrating RunnablePassthrough, RunnableParallel, prompt templates, and output parsers.
6. Comparative Benchmarking: Side-by-side latency, retrieval parity, and abstraction
   overhead profiling.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

# LangChain Core Abstractions
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser, StrOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

# Text Splitting & Vector Storage
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from langchain_chroma import Chroma

# Internal first-principles modules
from common.query_transformations import DeterministicSemanticEmbedder
from common.evaluation import RAGSynthesizer


# ---------------------------------------------------------------------------
# 1. Deterministic LangChain Embeddings Adapter
# ---------------------------------------------------------------------------

class DeterministicLangChainEmbeddings(Embeddings):
    """LangChain Embeddings adapter wrapping our deterministic CPU-friendly embedder.
    
    Conforms to the langchain_core.embeddings.Embeddings interface without requiring
    external model weights or network requests.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim
        self._embedder = DeterministicSemanticEmbedder(dim=dim)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document chunks into normalized float vectors."""
        embeddings: List[List[float]] = []
        for text in texts:
            vec = self._embedder.embed(text)
            embeddings.append(vec.tolist())
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string into a normalized float vector."""
        vec = self._embedder.embed(text)
        return vec.tolist()


# ---------------------------------------------------------------------------
# 2. Grounded Deterministic Chat Model for Offline LCEL Execution
# ---------------------------------------------------------------------------

class DeterministicLangChainChatModel(BaseChatModel):
    """A deterministic, grounded LangChain BaseChatModel for offline RAG workflows.
    
    Parses context passages from input prompt messages, detects topic relevance,
    synthesizes evidence-grounded answers with inline citations, and refuses
    to answer when evidence is missing.
    """

    groq_api_key: Optional[str] = None
    model_name: str = "offline-deterministic-grounded"
    synthesizer: Optional[Any] = None

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        model_name: str = "offline-deterministic-grounded",
        **kwargs: Any,
    ):
        super().__init__(groq_api_key=groq_api_key, model_name=model_name, **kwargs)
        self.synthesizer = RAGSynthesizer(groq_api_key=groq_api_key)

    @property
    def _llm_type(self) -> str:
        return "deterministic-grounded-chat-model"

    def _extract_query_and_contexts(self, prompt_text: str) -> Tuple[str, List[str]]:
        """Extracts context chunks and user query from formatted prompt text."""
        # Check for demarcation markers
        query = prompt_text
        contexts: List[str] = []

        if "=== CONTEXT PASSAGES ===" in prompt_text and "User Question:" in prompt_text:
            parts = prompt_text.split("=== CONTEXT PASSAGES ===")[1].split("========================")
            context_block = parts[0]
            after_context = parts[1]
            if "User Question:" in after_context:
                q_part = after_context.split("User Question:")[1].split("\n\n")[0]
                query = q_part.strip()
            # Split chunks by [Doc X] or headers
            chunks = re.split(r"\[Doc\s+\d+\]|\n\n---\n\n", context_block)
            contexts = [c.strip() for c in chunks if c.strip()]
        elif "Context:" in prompt_text and "Question:" in prompt_text:
            context_part = prompt_text.split("Context:")[1].split("Question:")[0]
            question_part = prompt_text.split("Question:")[1]
            query = question_part.strip()
            chunks = re.split(r"\[(?:Doc|chunk_)\w+\]|\n\n", context_part)
            contexts = [c.strip() for c in chunks if len(c.strip()) > 15]
        else:
            contexts = [prompt_text]

        return query, contexts

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generates grounded completion based on prompt messages."""
        # Combine message contents to extract prompt
        full_prompt = "\n".join(str(m.content) for m in messages)
        query, contexts = self._extract_query_and_contexts(full_prompt)

        synth = self.synthesizer or RAGSynthesizer()
        answer = synth.synthesize(query, contexts)

        generation = ChatGeneration(
            message=AIMessage(
                content=answer,
                additional_kwargs={"citations": re.findall(r"\[Doc \d+\]", answer)},
            )
        )
        return ChatResult(generations=[generation])


# ---------------------------------------------------------------------------
# 3. Structured Output Schema
# ---------------------------------------------------------------------------

class CitedAnswer(BaseModel):
    """Pydantic schema representing an evidence-grounded answer with source attribution."""

    question: str = Field(description="The original user question.")
    answer: str = Field(description="The factual answer grounded strictly in retrieved context.")
    citations: List[str] = Field(
        default_factory=list,
        description="List of document citations supporting the claims, e.g., ['[Doc 1]'].",
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score in [0.0, 1.0]. Low confidence (<0.5) signals potential refusal.",
    )
    is_refusal: bool = Field(
        default=False,
        description="True if the system refused to answer due to missing or contradictory evidence.",
    )


# ---------------------------------------------------------------------------
# 4. First-Principles RAG Pipeline (Baseline)
# ---------------------------------------------------------------------------

class FirstPrinciplesRAGPipeline:
    """Bare Python implementation of a RAG pipeline without framework abstractions.
    
    Used as the pedagogical and benchmark baseline to quantify abstraction overhead
    and verify parity with LangChain LCEL.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.embedder = DeterministicSemanticEmbedder(dim=dim)
        self.synthesizer = RAGSynthesizer()
        self.docs: List[Dict[str, Any]] = []
        self.embeddings_matrix: Optional[np.ndarray] = None

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Stores documents and computes vector representations."""
        self.docs.extend(documents)
        vecs = [self.embedder.embed(doc["text"]) for doc in self.docs]
        self.embeddings_matrix = np.vstack(vecs)

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Dense similarity retrieval using matrix dot products."""
        if self.embeddings_matrix is None or len(self.docs) == 0:
            return []

        q_vec = self.embedder.embed(query)
        scores = np.dot(self.embeddings_matrix, q_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": self.docs[idx]["text"],
                "metadata": self.docs[idx].get("metadata", {}),
                "score": float(scores[idx]),
            })
        return results

    def run(self, query: str, top_k: int = 2) -> Dict[str, Any]:
        """Executes full pipeline: retrieve -> format -> synthesize."""
        t0 = time.perf_counter()
        retrieved = self.retrieve(query, top_k=top_k)
        t_retrieval = (time.perf_counter() - t0) * 1000.0

        contexts = [r["text"] for r in retrieved]
        t1 = time.perf_counter()
        answer = self.synthesizer.synthesize(query, contexts)
        t_generation = (time.perf_counter() - t1) * 1000.0

        return {
            "query": query,
            "answer": answer,
            "retrieved_contexts": contexts,
            "citations": re.findall(r"\[Doc \d+\]", answer),
            "telemetry": {
                "retrieval_ms": t_retrieval,
                "generation_ms": t_generation,
                "total_ms": t_retrieval + t_generation,
            },
        }


# ---------------------------------------------------------------------------
# 5. LangChain LCEL RAG Pipeline
# ---------------------------------------------------------------------------

class LangChainLCELPipeline:
    """Idiomatic LangChain Expression Language (LCEL) RAG Pipeline.
    
    Features:
    - RecursiveCharacterTextSplitter for document chunking
    - LangChain Chroma VectorStore with DeterministicLangChainEmbeddings
    - LCEL pipe syntax (|) chaining retriever, prompt, model, and parser
    - Structured Pydantic parsing via RunnableLambda / PydanticOutputParser
    """

    def __init__(self, collection_name: str = "lcel_pipeline_col", dim: int = 128):
        self.dim = dim
        self.collection_name = collection_name
        self.embeddings = DeterministicLangChainEmbeddings(dim=dim)
        self.chroma_client = chromadb.EphemeralClient()
        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )
        self.chat_model = DeterministicLangChainChatModel()
        self.retriever = None
        self.rag_chain = None
        self.structured_chain = None

    def index_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ) -> List[Document]:
        """Splits raw texts using RecursiveCharacterTextSplitter and stores in Chroma."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        initial_docs = [
            Document(page_content=t, metadata=metadatas[i] if metadatas else {"doc_id": i})
            for i, t in enumerate(texts)
        ]
        split_docs = splitter.split_documents(initial_docs)
        self.vectorstore.add_documents(split_docs)
        return split_docs

    def build_chain(self, top_k: int = 2) -> Any:
        """Constructs canonical LCEL RAG chain using the pipe operator."""
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})

        def format_docs(docs: List[Document]) -> str:
            chunks = []
            for i, doc in enumerate(docs, start=1):
                chunks.append(f"[Doc {i}] (Source: {doc.metadata.get('doc_id', i)})\n{doc.page_content}")
            return "\n\n".join(chunks)

        template = (
            "=== CONTEXT PASSAGES ===\n"
            "{context}\n"
            "========================\n\n"
            "User Question: {question}\n\n"
            "Grounded Answer (with inline [Doc X] citations):"
        )
        prompt = ChatPromptTemplate.from_template(template)

        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.chat_model
            | StrOutputParser()
        )

        # Build structured output chain
        def parse_to_cited_answer(answer_text: str, question: str) -> CitedAnswer:
            citations = re.findall(r"\[Doc \d+\]", answer_text)
            is_refusal = "cannot answer" in answer_text.lower()
            confidence = 0.2 if is_refusal else 0.95
            return CitedAnswer(
                question=question,
                answer=answer_text,
                citations=citations,
                confidence=confidence,
                is_refusal=is_refusal,
            )

        self.structured_chain = RunnableParallel(
            answer_text=self.rag_chain,
            question=RunnablePassthrough(),
        ) | RunnableLambda(lambda x: parse_to_cited_answer(x["answer_text"], x["question"]))

        return self.rag_chain

    def invoke(self, query: str) -> str:
        """Invokes the LCEL chain synchronously."""
        if self.rag_chain is None:
            self.build_chain()
        return self.rag_chain.invoke(query)

    def invoke_structured(self, query: str) -> CitedAnswer:
        """Invokes chain and parses into a strongly-typed CitedAnswer Pydantic model."""
        if self.structured_chain is None:
            self.build_chain()
        return self.structured_chain.invoke(query)

    def batch(self, queries: List[str]) -> List[str]:
        """Batch invocation across multiple queries."""
        if self.rag_chain is None:
            self.build_chain()
        return self.rag_chain.batch(queries)

    def get_graph_nodes(self) -> List[str]:
        """Extracts node names from the LCEL execution graph without external graph libraries."""
        if self.rag_chain is None:
            self.build_chain()
        graph = self.rag_chain.get_graph()
        return [node.name for node in graph.nodes.values()]


# ---------------------------------------------------------------------------
# 6. Side-by-Side Comparison & Benchmarking Harness
# ---------------------------------------------------------------------------

@dataclass
class ArchitectureBenchmarkSummary:
    """Benchmark results comparing First-Principles and LangChain LCEL RAG."""

    query: str
    first_principles_answer: str
    langchain_answer: str
    first_principles_latency_ms: float
    langchain_latency_ms: float
    abstraction_overhead_ms: float
    overhead_percentage: float
    retrieval_parity: bool
    citation_parity: bool


def benchmark_rag_architectures(
    corpus_texts: List[str],
    test_queries: List[str],
    top_k: int = 2,
) -> Tuple[List[ArchitectureBenchmarkSummary], Dict[str, Any]]:
    """Runs identical queries through both pipelines, measuring parity and latency overhead."""
    # 1. Setup First-Principles Pipeline
    fp_pipeline = FirstPrinciplesRAGPipeline()
    fp_docs = [{"text": t, "metadata": {"doc_id": i}} for i, t in enumerate(corpus_texts)]
    fp_pipeline.add_documents(fp_docs)

    # 2. Setup LangChain LCEL Pipeline
    lc_pipeline = LangChainLCELPipeline()
    lc_pipeline.index_texts(corpus_texts, chunk_size=1000, chunk_overlap=0)
    lc_pipeline.build_chain(top_k=top_k)

    summaries: List[ArchitectureBenchmarkSummary] = []

    for q in test_queries:
        # Measure First Principles
        t0 = time.perf_counter()
        fp_res = fp_pipeline.run(q, top_k=top_k)
        fp_time = (time.perf_counter() - t0) * 1000.0

        # Measure LangChain LCEL
        t1 = time.perf_counter()
        lc_answer = lc_pipeline.invoke(q)
        lc_time = (time.perf_counter() - t1) * 1000.0

        overhead = lc_time - fp_time
        overhead_pct = (overhead / max(0.001, fp_time)) * 100.0

        # Parity checks
        fp_citations = set(fp_res["citations"])
        lc_citations = set(re.findall(r"\[Doc \d+\]", lc_answer))
        citation_parity = fp_citations == lc_citations

        # Check if both agree on refusal or groundedness
        fp_refusal = "cannot answer" in fp_res["answer"].lower()
        lc_refusal = "cannot answer" in lc_answer.lower()
        retrieval_parity = fp_refusal == lc_refusal

        summaries.append(
            ArchitectureBenchmarkSummary(
                query=q,
                first_principles_answer=fp_res["answer"],
                langchain_answer=lc_answer,
                first_principles_latency_ms=round(fp_time, 3),
                langchain_latency_ms=round(lc_time, 3),
                abstraction_overhead_ms=round(overhead, 3),
                overhead_percentage=round(overhead_pct, 1),
                retrieval_parity=retrieval_parity,
                citation_parity=citation_parity,
            )
        )

    # Aggregate telemetry
    avg_fp_latency = float(np.mean([s.first_principles_latency_ms for s in summaries]))
    avg_lc_latency = float(np.mean([s.langchain_latency_ms for s in summaries]))
    avg_overhead = avg_lc_latency - avg_fp_latency
    all_retrieval_parity = all(s.retrieval_parity for s in summaries)

    aggregate = {
        "num_queries": len(test_queries),
        "mean_first_principles_ms": round(avg_fp_latency, 3),
        "mean_langchain_ms": round(avg_lc_latency, 3),
        "mean_overhead_ms": round(avg_overhead, 3),
        "mean_overhead_pct": round((avg_overhead / max(0.001, avg_fp_latency)) * 100.0, 1),
        "retrieval_parity_rate": sum(1 for s in summaries if s.retrieval_parity) / len(summaries),
        "citation_parity_rate": sum(1 for s in summaries if s.citation_parity) / len(summaries),
    }

    return summaries, aggregate
