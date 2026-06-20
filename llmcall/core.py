from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    temperature: float = 0.2
    stream: bool = False
    n: int | None = 1
    max_output_tokens: int = 4096
    num_retries: int = 3
    seed: int | None = None


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
    base_url: str | None = None
    debug: bool = False
    llm: ModelConfig = ModelConfig()


_config: LLMConfig | None = None


def get_config() -> LLMConfig:
    global _config
    if _config is None:
        _config = LLMConfig()
    return _config


def config() -> LLMConfig:
    return get_config()


def split_model(model: str) -> tuple[str | None, str]:
    """Split a LiteLLM model name into optional provider and model name."""
    if "/" not in model:
        return None, model
    provider, model_name = model.split("/", 1)
    return provider or None, model_name


def optional_completion_params(cfg: LLMConfig) -> dict:
    """Return completion params that should only be sent when configured."""
    params = {}
    if cfg.llm.seed is not None:
        params["seed"] = cfg.llm.seed
    return params
