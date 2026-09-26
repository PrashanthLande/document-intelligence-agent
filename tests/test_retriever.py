from pathlib import Path

from document_intelligence_agent.chunking.chunker import DocumentChunker
from document_intelligence_agent.embeddings.embedder import DocumentEmbedder
from document_intelligence_agent.ingestion.loader import DocumentLoader
from document_intelligence_agent.retrieval.retriever import Retriever
from document_intelligence_agent.vector_store.chroma_store import (
    ChromaVectorStore,
)

FIXTURE = Path("tests/fixtures/docling_technical_report.pdf")


def test_retriever():
    document = DocumentLoader().load(FIXTURE)

    chunks = DocumentChunker().chunk(
        document,
        document_id=FIXTURE.name,
    )

    embedder = DocumentEmbedder()

    embeddings = embedder.embed_chunks(chunks)

    store = ChromaVectorStore(
        collection_name="test_retriever",
    )

    store.add(
        chunks=chunks,
        embeddings=embeddings,
    )

    retriever = Retriever(
        embedder=embedder,
        vector_store=store,
    )

    results = retriever.retrieve(
        query="What is Docling?",
        top_k=3,
    )

    assert len(results) == 3

    assert results[0]["text"]
    assert results[0]["metadata"]
    assert results[0]["distance"] is not None