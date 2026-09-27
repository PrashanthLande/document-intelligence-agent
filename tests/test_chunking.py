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


def test_chunk_table_items():
    loader = DocumentLoader()
    document = loader.load(FIXTURE)

    chunker = DocumentChunker()
    chunks = chunker.chunk(
        document=document,
        document_id="docling-report",
    )

    table_chunks = [
        chunk
        for chunk in chunks
        if "TableItem" in chunk.item_types
    ]

    assert table_chunks
    assert all(chunk.text for chunk in table_chunks)
    assert all(chunk.page_numbers for chunk in table_chunks)
    assert all(chunk.provenance for chunk in table_chunks)


def test_table_chunks_preserve_structure():
    loader = DocumentLoader()
    document = loader.load(FIXTURE)

    chunker = DocumentChunker()
    chunks = chunker.chunk(
        document=document,
        document_id="docling-report",
    )

    table_chunks = [
        chunk
        for chunk in chunks
        if "TableItem" in chunk.item_types
    ]

    assert any("\n" in chunk.text for chunk in table_chunks)
    assert any(" | " in chunk.text for chunk in table_chunks)