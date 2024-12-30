from typing import Any, List, Optional, Union
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from litellm import completion

class LLMConfig(BaseSettings):
    api_key: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 1024

    class Config:
        env_prefix = "LLMCALL_"
        
config = LLMConfig()

