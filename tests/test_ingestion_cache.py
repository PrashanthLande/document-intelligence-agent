from docling.datamodel.document import DoclingDocument

from document_intelligence_agent.ingestion.cache import CachingDocumentLoader


class CountingLoader:
    """Stand-in for DocumentLoader that counts conversions."""

    pipeline = "standard"

    def __init__(self) -> None:
        self.calls = 0

    def load(self, source) -> DoclingDocument:
        self.calls += 1
        document = DoclingDocument(name="test")
        document.add_text(label="text", text=f"content of {source.name}")
        return document


def test_second_load_comes_from_cache(tmp_path):
    source = tmp_path / "a.pdf"
    source.write_bytes(b"pdf bytes")
    inner = CountingLoader()
    loader = CachingDocumentLoader(inner, tmp_path / "cache")

    first = loader.load(source)
    second = loader.load(source)

    assert inner.calls == 1
    assert second.export_to_markdown() == first.export_to_markdown()


def test_changed_file_is_converted_again(tmp_path):
    source = tmp_path / "a.pdf"
    inner = CountingLoader()
    loader = CachingDocumentLoader(inner, tmp_path / "cache")

    source.write_bytes(b"version 1")
    loader.load(source)
    source.write_bytes(b"version 2")
    loader.load(source)

    assert inner.calls == 2
