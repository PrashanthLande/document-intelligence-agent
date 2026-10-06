import json
from pathlib import Path

from document_intelligence_agent.chunking.chunker import Chunk, DocumentChunker
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


def make_chunk(document_id: str, text: str) -> Chunk:
    return Chunk(
        text=text,
        document_id=document_id,
        page_numbers=[1],
        section=None,
        item_types=["TextItem"],
        provenance=[],
    )


def make_two_document_store(collection_name: str) -> ChromaVectorStore:
    """A store with two chunks from a.pdf and one from b.pdf."""
    store = ChromaVectorStore(collection_name=collection_name)

    store.add(
        chunks=[
            make_chunk("a.pdf", "first chunk of a"),
            make_chunk("a.pdf", "second chunk of a"),
            make_chunk("b.pdf", "only chunk of b"),
        ],
        embeddings=[[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]],
    )

    return store


def result_document_ids(result) -> set[str]:
    return {metadata["document_id"] for metadata in result["metadatas"][0]}


def test_search_without_filter_searches_all_documents():
    store = make_two_document_store("test_search_all")

    result = store.search(query_embedding=[1.0, 0.0], top_k=3)

    assert result_document_ids(result) == {"a.pdf", "b.pdf"}


def test_search_filters_by_document_ids():
    store = make_two_document_store("test_search_filter")

    result = store.search(
        query_embedding=[1.0, 0.0],
        top_k=3,
        document_ids=["b.pdf"],
    )

    assert result_document_ids(result) == {"b.pdf"}
    assert len(result["ids"][0]) == 1


def test_delete_document_removes_only_its_chunks():
    store = make_two_document_store("test_delete_document")

    store.delete_document("a.pdf")

    remaining = store.collection.get()

    assert {m["document_id"] for m in remaining["metadatas"]} == {"b.pdf"}