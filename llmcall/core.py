from typing import Optional
from pydantic_settings import BaseSettings


class ModelConfig(BaseSettings):
    temperature: float = 0.2
    stream: bool = False
    n: Optional[int] = 1
    max_tokens: int = 1024
    num_retries: int = 3
    seed: Optional[int] = 47


class LLMConfig(BaseSettings):
    api_key: str
    model: str = "openai/gpt-4o-mini"
    llm: ModelConfig = ModelConfig()

    class Config:
        env_prefix = "LLMCALL_"
        case_sensitive = False


config = LLMConfig()
