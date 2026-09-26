from typing import Any

from document_intelligence_agent.rag.generator import RAGGenerator
from document_intelligence_agent.reranking.reranker import DocumentReranker
from document_intelligence_agent.retrieval.retriever import Retriever


class RAGPipeline:
    """Run retrieval, reranking, and grounded answer generation."""

    def __init__(
            self,
            retriever: Retriever,
            reranker: DocumentReranker,
            generator: RAGGenerator,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

    def answer(
            self,
            query: str,
            retrieval_k: int = 5,
            rerank_k: int = 3,
    ) -> dict[str, Any]:
        """Answer a question using retrieved and reranked evidence."""

        retrieved = self.retriever.retrieve(
            query=query,
            top_k=retrieval_k,
        )

        reranked = self.reranker.rerank(
            query=query,
            results=retrieved,
            top_k=rerank_k,
        )

        generated = self.generator.generate(
            query=query,
            evidence=reranked,
        )

        return {
            "answer": generated["answer"],
            "sources": generated["sources"],
            "evidence": reranked,
        }