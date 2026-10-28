"""Build pipeline components from settings."""

from document_intelligence_agent.config import GeneratorSettings
from document_intelligence_agent.rag.llm import (
    Generator,
    OpenAICompatibleGenerator,
    TransformersGenerator,
)


def build_generator(settings: GeneratorSettings) -> Generator:
    """Create the LLM backend named in the settings."""
    if settings.backend == "transformers":
        return TransformersGenerator(
            model_name=settings.model,
            max_new_tokens=settings.max_new_tokens,
        )

    api_key = settings.api_key.get_secret_value() if settings.api_key else None

    return OpenAICompatibleGenerator(
        model_name=settings.model,
        base_url=settings.base_url,
        api_key=api_key,
    )
