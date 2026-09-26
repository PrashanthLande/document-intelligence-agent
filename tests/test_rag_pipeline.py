from pathlib import Path

from document_intelligence_agent.chunking.chunker import DocumentChunker
from document_intelligence_agent.embeddings.embedder import DocumentEmbedder
from document_intelligence_agent.ingestion.loader import DocumentLoader
from document_intelligence_agent.rag.generator import RAGGenerator
from document_intelligence_agent.rag.pipeline import RAGPipeline
from document_intelligence_agent.reranking.reranker import (
    DocumentReranker,
)
from document_intelligence_agent.retrieval.retriever import Retriever
from document_intelligence_agent.vector_store.chroma_store import (
    ChromaVectorStore,
)

FIXTURE = Path("tests/fixtures/docling_technical_report.pdf")


def test_rag_pipeline():
    document = DocumentLoader().load(FIXTURE)

    chunks = DocumentChunker().chunk(
        document,
        document_id=FIXTURE.name,
    )

    embedder = DocumentEmbedder()

    embeddings = embedder.embed_chunks(chunks)

    store = ChromaVectorStore(
        collection_name="test_rag_pipeline",
    )

    store.add(
        chunks=chunks,
        embeddings=embeddings,
    )

    retriever = Retriever(
        embedder=embedder,
        vector_store=store,
    )

    reranker = DocumentReranker()

    generator = RAGGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
    )

    result = pipeline.answer(
        query="What is Docling?",
        retrieval_k=5,
        rerank_k=3,
    )

    assert result["answer"]
    assert isinstance(result["answer"], str)

    assert result["sources"]
    assert len(result["sources"]) == 3

    assert result["evidence"]
    assert len(result["evidence"]) == 3

    for evidence in result["evidence"]:
        assert evidence["text"]
        assert evidence["metadata"]
        assert "rerank_score" in evidence

    for source in result["sources"]:
        assert source["document_id"]
        assert source["page_numbers"]
        assert source["section"]
