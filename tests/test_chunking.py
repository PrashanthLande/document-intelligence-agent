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


def test_short_items_are_merged():
    document = DocumentLoader().load(FIXTURE)

    merged = DocumentChunker().chunk(document, document_id=FIXTURE.name)
    unmerged = DocumentChunker(max_chars=0).chunk(
        document,
        document_id=FIXTURE.name,
    )

    assert len(merged) < len(unmerged)


def test_merged_chunks_respect_limits():
    document = DocumentLoader().load(FIXTURE)

    chunker = DocumentChunker()
    chunks = chunker.chunk(document, document_id=FIXTURE.name)

    for chunk in chunks:
        item_count = len({entry["text_offset"] for entry in chunk.provenance})

        if item_count > 1:
            assert len(chunk.text) <= chunker.max_chars
            assert "TableItem" not in chunk.item_types


def test_merged_provenance_offsets_point_into_text():
    document = DocumentLoader().load(FIXTURE)

    chunks = DocumentChunker().chunk(document, document_id=FIXTURE.name)

    for chunk in chunks:
        for entry in chunk.provenance:
            assert 0 <= entry["text_offset"] < len(chunk.text)