from pathlib import Path

from docling.datamodel.document import DoclingDocument

from document_intelligence_agent.ingestion.loader import DocumentLoader


FIXTURE = Path("tests/fixtures/docling_technical_report.pdf")


def test_load_document():
    loader = DocumentLoader()

    document = loader.load(FIXTURE)

    assert isinstance(document, DoclingDocument)
    assert len(document.pages) > 0