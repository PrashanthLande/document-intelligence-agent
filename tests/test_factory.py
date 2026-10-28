from document_intelligence_agent.config import GeneratorSettings, Settings
from document_intelligence_agent.factory import build_generator
from document_intelligence_agent.rag.llm import OpenAICompatibleGenerator


def test_settings_default_to_local_transformers():
    settings = Settings(_env_file=None)

    assert settings.generator.backend == "transformers"
    assert settings.generator.model == "Qwen/Qwen3-1.7B"


def test_settings_read_nested_environment_variables(monkeypatch):
    monkeypatch.setenv("GENERATOR__BACKEND", "openai_compatible")
    monkeypatch.setenv("GENERATOR__MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GENERATOR__BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("GENERATOR__API_KEY", "secret")

    settings = Settings(_env_file=None)

    assert settings.generator.backend == "openai_compatible"
    assert settings.generator.model == "llama-3.3-70b-versatile"
    assert settings.generator.base_url == "https://api.groq.com/openai/v1"
    assert settings.generator.api_key.get_secret_value() == "secret"


def test_build_openai_compatible_generator():
    # Creating the client makes no network call.
    generator = build_generator(
        GeneratorSettings(
            backend="openai_compatible",
            model="qwen3:1.7b",
            base_url="http://localhost:11434/v1",
        )
    )

    assert isinstance(generator, OpenAICompatibleGenerator)
    assert generator.model_name == "qwen3:1.7b"
    assert str(generator.client.base_url) == "http://localhost:11434/v1/"
