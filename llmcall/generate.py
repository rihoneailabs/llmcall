from typing import List, Optional, Union

from litellm import completion
from pydantic import BaseModel

from llmcall.core import config


def generate(
    prompt: str,
    output_schema: BaseModel = None,
    instructions: Optional[str] = None,
) -> Union[str, dict]:
    """Generate content using configured LLM."""
    DEFAULT_SYSTEM_PROMPT = "Generate content based on the following: <prompt>{prompt}</prompt>. \
        Return only the content with no additional information or comments."
    response = completion(
        api_key=config.api_key,
        model=config.model,
        messages=[
            {"content": instructions or DEFAULT_SYSTEM_PROMPT, "role": "system"},
            {"content": prompt, "role": "user"},
        ],
        response_format=output_schema,
        **config.llm.model_dump(),
    )
    return response.choices[0].message.content


def generate_decision(
    prompt: str,
    options: List[str],
    instructions: Optional[str] = None,
) -> str:
    """Generate a decision from a list of options."""
    pass  # Implementation here
