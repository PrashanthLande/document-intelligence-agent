from pathlib import Path
from typing import Literal

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import DoclingDocument
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

PipelineType = Literal["standard", "vlm"]


class DocumentLoader:
    """Load documents using a configurable Docling pipeline."""

    def __init__(
            self,
            pipeline: PipelineType = "standard",
            converter: DocumentConverter | None = None,
    ):
        if pipeline not in {"standard", "vlm"}:
            raise ValueError(
                f"Unsupported pipeline: {pipeline}. "
                "Expected 'standard' or 'vlm'."
            )

        self.pipeline = pipeline

        if converter is not None:
            self.converter = converter
        elif pipeline == "standard":
            self.converter = DocumentConverter()
        else:
            self.converter = self._create_vlm_converter()

    def _create_vlm_converter(self) -> DocumentConverter:
        """Create a Docling converter configured for VLM processing."""

        pipeline_options = VlmPipelineOptions()

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                )
            }
        )

    def load(self, source: str | Path) -> DoclingDocument:
        """Convert a document and return the resulting DoclingDocument."""
        source = Path(source)

        if not source.exists():
            raise FileNotFoundError(f"Document not found: {source}")

        result = self.converter.convert(source)

        return result.document
