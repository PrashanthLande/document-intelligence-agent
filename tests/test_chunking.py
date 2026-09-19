from pathlib import Path

from document_intelligence_agent.chunking.chunker import DocumentChunker
from document_intelligence_agent.ingestion.loader import DocumentLoader


FIXTURE = Path("tests/fixtures/docling_technical_report.pdf")


def test_document_chunking():
    document = DocumentLoader().load(FIXTURE)

    chunks = DocumentChunker().chunk(
        document,
        document_id=FIXTURE.name,
    )

    assert chunks
    assert any(chunk.section for chunk in chunks)

    first = chunks[0]

    assert first.text
    assert first.document_id == FIXTURE.name
    assert first.page_numbers
    assert first.item_types
    assert first.provenance
