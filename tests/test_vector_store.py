import json
from pathlib import Path

from document_intelligence_agent.chunking.chunker import DocumentChunker
from document_intelligence_agent.embeddings.embedder import DocumentEmbedder
from document_intelligence_agent.ingestion.loader import DocumentLoader
from document_intelligence_agent.vector_store.chroma_store import (
    ChromaVectorStore,
)


FIXTURE = Path("tests/fixtures/docling_technical_report.pdf")


def test_vector_store():
    document = DocumentLoader().load(FIXTURE)

    chunks = DocumentChunker().chunk(
        document,
        document_id=FIXTURE.name,
    )

    embedder = DocumentEmbedder()
    embeddings = embedder.embed_chunks(chunks)

    store = ChromaVectorStore(
        collection_name="test_document_chunks",
    )

    store.add(
        chunks=chunks,
        embeddings=embeddings,
    )

    result = store.collection.get()

    assert len(result["ids"]) == len(chunks)
    assert len(result["documents"]) == len(chunks)
    assert len(result["metadatas"]) == len(chunks)


def test_vector_store_preserves_provenance():
    document = DocumentLoader().load(FIXTURE)

    chunks = DocumentChunker().chunk(
        document,
        document_id=FIXTURE.name,
    )

    embedder = DocumentEmbedder()
    embeddings = embedder.embed_chunks(chunks)

    store = ChromaVectorStore(
        collection_name="test_provenance",
    )

    store.add(
        chunks=chunks,
        embeddings=embeddings,
    )

    result = store.collection.get()

    assert len(result["metadatas"]) == len(chunks)

    for metadata in result["metadatas"]:
        provenance = json.loads(metadata["provenance"])

        assert isinstance(provenance, list)
        assert provenance
        assert "page" in provenance[0]
        assert "bbox" in provenance[0]
        assert "coord_origin" in provenance[0]
        assert "charspan" in provenance[0]