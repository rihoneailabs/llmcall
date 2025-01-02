from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    temperature: float = 0.2
    stream: bool = False
    n: Optional[int] = 1
    max_tokens: int = 1024
    num_retries: int = 3
    seed: Optional[int] = 47


class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix="LLMCALL_",
        extra="ignore",
    )
    api_key: str
    model: str = "openai/gpt-4o-2024-08-06"
    debug: bool = False
    llm: ModelConfig = ModelConfig()


config = LLMConfig()
