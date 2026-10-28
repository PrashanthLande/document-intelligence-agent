import pytest


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
