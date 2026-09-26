from pathlib import Path

from document_intelligence_agent.chunking.chunker import DocumentChunker
from document_intelligence_agent.embeddings.embedder import DocumentEmbedder
from document_intelligence_agent.ingestion.loader import DocumentLoader
from document_intelligence_agent.reranking.reranker import (
    DocumentReranker,
)
from document_intelligence_agent.retrieval.retriever import Retriever
from document_intelligence_agent.vector_store.chroma_store import (
    ChromaVectorStore,
)

FIXTURE = Path("tests/fixtures/docling_technical_report.pdf")


def test_retrieval_and_reranking():
    # 1. Load document
    document = DocumentLoader().load(FIXTURE)

    # 2. Create provenance-aware chunks
    chunks = DocumentChunker().chunk(
        document,
        document_id=FIXTURE.name,
    )

    # 3. Generate embeddings
    embedder = DocumentEmbedder()

    embeddings = embedder.embed_chunks(chunks)

    # 4. Store chunks and embeddings
    store = ChromaVectorStore(
        collection_name="test_retrieval_reranking",
    )

    store.add(
        chunks=chunks,
        embeddings=embeddings,
    )

    # 5. Retrieve initial candidates
    retriever = Retriever(
        embedder=embedder,
        vector_store=store,
    )

    query = "What is Docling?"

    results = retriever.retrieve(
        query=query,
        top_k=5,
    )

    assert len(results) == 5

    # 6. Rerank retrieved candidates
    reranker = DocumentReranker()

    reranked = reranker.rerank(
        query=query,
        results=results,
        top_k=3,
    )

    # 7. Validate reranked results
    assert len(reranked) == 3

    assert all(
        "rerank_score" in result
        for result in reranked
    )

    assert all(
        result["text"]
        for result in reranked
    )

    assert all(
        result["metadata"]
        for result in reranked
    )

    # Higher rerank score means higher relevance.
    assert (
            reranked[0]["rerank_score"]
            >= reranked[1]["rerank_score"]
            >= reranked[2]["rerank_score"]
    )

