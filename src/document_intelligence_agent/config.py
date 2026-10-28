"""Application settings, read from environment variables and `.env`.

Nested settings use a double underscore, e.g. `GENERATOR__BACKEND`.
"""

from typing import Literal

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class GeneratorSettings(BaseModel):
    """Which LLM backend answers questions."""

    backend: Literal["transformers", "openai_compatible"] = "transformers"
    model: str = "Qwen/Qwen3-1.7B"
    # openai_compatible only. None means api.openai.com.
    base_url: str | None = None
    api_key: SecretStr | None = None
    # transformers only.
    max_new_tokens: int = 512


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    generator: GeneratorSettings = GeneratorSettings()
