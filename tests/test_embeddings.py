from pathlib import Path

from document_intelligence_agent.chunking.chunker import DocumentChunker
from document_intelligence_agent.embeddings.embedder import DocumentEmbedder
from document_intelligence_agent.ingestion.loader import DocumentLoader


FIXTURE = Path("tests/fixtures/docling_technical_report.pdf")


def test_document_embeddings():
    document = DocumentLoader().load(FIXTURE)

    chunks = DocumentChunker().chunk(
        document,
        document_id=FIXTURE.name,
    )

    embedder = DocumentEmbedder()
    embeddings = embedder.embed_chunks(chunks)

    assert len(embeddings) == len(chunks)
    assert len(embeddings) > 0
    assert len(embeddings[0]) == 384