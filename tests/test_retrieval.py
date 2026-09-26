from pathlib import Path

from document_intelligence_agent.chunking.chunker import DocumentChunker
from document_intelligence_agent.embeddings.embedder import DocumentEmbedder
from document_intelligence_agent.ingestion.loader import DocumentLoader
from document_intelligence_agent.vector_store.chroma_store import (
    ChromaVectorStore,
)

FIXTURE = Path("tests/fixtures/docling_technical_report.pdf")


def test_vector_retrieval():
    document = DocumentLoader().load(FIXTURE)

    chunks = DocumentChunker().chunk(
        document,
        document_id=FIXTURE.name,
    )

    embedder = DocumentEmbedder()
    embeddings = embedder.embed_chunks(chunks)

    store = ChromaVectorStore(
        collection_name="test_retrieval",
    )

    store.add(
        chunks=chunks,
        embeddings=embeddings,
    )

    query = "What is Docling?"

    query_embedding = embedder.embed_query(query)

    results = store.search(
        query_embedding=query_embedding,
        top_k=3,
    )

    assert results["ids"]
    assert len(results["ids"][0]) == 3
    assert results["documents"]
    assert results["metadatas"]
