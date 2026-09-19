from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.datamodel.document import DoclingDocument


class DocumentLoader:
    """Load documents using Docling's standard pipeline."""

    def __init__(self):
        self.converter = DocumentConverter()

    def load(self, source: str | Path) -> DoclingDocument:
        """Convert a document and return the resulting DoclingDocument."""
        source = Path(source)

        if not source.exists():
            raise FileNotFoundError(f"Document not found: {source}")

        result = self.converter.convert(source)

        return result.document
