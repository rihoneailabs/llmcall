from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    temperature: float = 0.2
    stream: bool = False
    n: Optional[int] = 1
    max_output_tokens: int = 4096
    num_retries: int = 3
    seed: Optional[int] = None


class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix="LLMCALL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    api_key: str
    model: str = "openai/gpt-4.1"
    base_url: Optional[str] = None
    debug: bool = False
    llm: ModelConfig = ModelConfig()


_config: Optional[LLMConfig] = None


def get_config() -> LLMConfig:
    global _config
    if _config is None:
        _config = LLMConfig()
    return _config
