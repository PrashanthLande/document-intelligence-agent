import hashlib
from importlib.metadata import version
from pathlib import Path

from docling.datamodel.document import DoclingDocument

from document_intelligence_agent.ingestion.loader import DocumentLoader


class CachingDocumentLoader:
    """Wrap a DocumentLoader and keep each converted document on disk.

    Converting a PDF takes about a minute; loading the saved JSON takes a
    fraction of a second. The cache key covers the file contents, the
    Docling version and the pipeline, since any of them can change the
    result.
    """

    def __init__(
            self,
            loader: DocumentLoader,
            cache_dir: str | Path,
    ) -> None:
        self.loader = loader
        self.cache_dir = Path(cache_dir)

    def load(self, source: str | Path) -> DoclingDocument:
        source = Path(source)

        if not source.exists():
            raise FileNotFoundError(f"Document not found: {source}")

        cache_file = self.cache_dir / f"{self._key(source)}.json"

        if cache_file.exists():
            return DoclingDocument.load_from_json(cache_file)

        document = self.loader.load(source)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        document.save_as_json(cache_file)

        return document

    def _key(self, source: Path) -> str:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]

        return f"{digest}-docling{version('docling')}-{self.loader.pipeline}"
