from pathlib import Path

from docling.datamodel.document import DoclingDocument
from docling.pipeline.vlm_pipeline import VlmPipeline

from document_intelligence_agent.ingestion.loader import DocumentLoader


FIXTURE = Path("tests/fixtures/docling_technical_report.pdf")


def test_load_document():
    loader = DocumentLoader()

    document = loader.load(FIXTURE)

    assert isinstance(document, DoclingDocument)
    assert len(document.pages) > 0


def test_standard_pipeline_is_default():
    loader = DocumentLoader()

    assert loader.pipeline == "standard"


def test_vlm_pipeline_configuration():
    loader = DocumentLoader(pipeline="vlm")

    assert loader.pipeline == "vlm"

    pdf_options = loader.converter.format_to_options[
        "pdf"
    ]

    assert pdf_options.pipeline_cls is VlmPipeline


def test_invalid_pipeline():
    try:
        DocumentLoader(pipeline="invalid")
    except ValueError as exc:
        assert "Expected 'standard' or 'vlm'" in str(exc)
    else:
        raise AssertionError("Expected ValueError")