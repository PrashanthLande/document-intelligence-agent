from pathlib import Path

import pytest
from docling.datamodel.document import DoclingDocument

from document_intelligence_agent.ingestion.loader import DocumentLoader

FIXTURE = Path("tests/fixtures/docling_technical_report.pdf")


@pytest.fixture(scope="session")
def document() -> DoclingDocument:
    """The fixture PDF, converted once per test run (about a minute)."""
    return DocumentLoader().load(FIXTURE)


class FakeGenerator:
    """Stand-in LLM that records the prompt and returns a fixed answer."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Docling converts documents."


@pytest.fixture
def fake_llm() -> FakeGenerator:
    return FakeGenerator()
