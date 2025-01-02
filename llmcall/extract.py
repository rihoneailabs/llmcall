import json
import logging
import time
from typing import Optional, Union
from typing_extensions import Annotated

from litellm import completion
from pydantic import BaseModel

from llmcall.core import config

_logger = logging.getLogger(__name__)


def extract(
    text: Annotated[str, "The unstructured text to extract information from."],
    output_schema: Annotated[BaseModel, "The Pydantic model to use for response structure validation."],
    instructions: Annotated[Optional[str], "System metaprompt to condition the model."] = None,
) -> Union[str, BaseModel]:
    """Extract structured information from unstructured text using configured LLM."""
    DEFAULT_SYSTEM_PROMPT = """You are a specialist in organising unstructured data. Given the document below, \
    your task is to extract the requested information as accurately as you can. \
    ###
    <document>{text}</document>
    ###"""

    if not text:
        raise ValueError("Text cannot be empty.")
    
    start = time.perf_counter()
    _logger.info(f"Extracting information from text: {text[:50]}")

    if instructions:
        messages = [
            {"content": instructions, "role": "system"},
            {"content": text, "role": "user"},
        ]
    else:
        messages = [
            {"content": DEFAULT_SYSTEM_PROMPT.format(text=text), "role": "system"},
        ]

    response = completion(
        api_key=config.api_key,
        model=config.model,
        messages=messages,
        response_format=output_schema,
        **config.llm.model_dump(),
    )

    _logger.info(f"Extraction completed in {time.perf_counter() - start:.2f} seconds.")

    return output_schema.model_validate(json.loads(response.choices[0].message.content), strict=True)
