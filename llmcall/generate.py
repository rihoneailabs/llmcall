import json
import logging
import time
from typing import Optional, Union

import litellm
from litellm import completion
from litellm import supports_response_schema
from pydantic import BaseModel

from llmcall.core import config

_logger = logging.getLogger(__name__)


class Decision(BaseModel):
    value: str
    prompt: Optional[str] = None
    options: Optional[list[str]] = None
    reason: Optional[str] = None


def generate(
    prompt: str,
    output_schema: BaseModel = None,
    instructions: Optional[str] = None,
) -> Union[str, BaseModel]:
    """Generate content using configured LLM."""
    DEFAULT_SYSTEM_PROMPT = "Generate content based on the following: <prompt>{prompt}</prompt>. \
        Return only the content with no additional information or comments."

    _logger.debug(f"Generating content for prompt: {prompt[:20]}..")
    start = time.perf_counter()

    if output_schema:
        if not supports_response_schema(
            model=config.model.split("/")[1], custom_llm_provider=config.model.split("/")[0]
        ):
            raise ValueError(
                f"Response schema is not supported by the configured model: {config.model}. "
                "Please use a different model(e.g. openai/gpt-4o-2024-08-06) or remove the output schema."
            )
        litellm.enable_json_schema_validation = True

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

    _logger.debug(f"Generated content in {time.perf_counter() - start:.2f}s")

    if output_schema:
        return output_schema.model_validate(json.loads(response.choices[0].message.content))

    return response.choices[0].message.content


def generate_decision(
    prompt: str,
    options: list[str],
    instructions: Optional[str] = None,
) -> Decision:
    """Generate a decision from a list of options."""

    DEFAULT_SYSTEM_PROMPT = """You are a specialized computer algorithm designed to make decisions in Control Flow scenarios. \
        Your task is to analyze the given context and options, then select the most appropriate option based on the context. Here is the context you need to consider: \
            <context>
            {{CONTEXT}}
            </context>

        Here are the options you can choose from:
            <options>
            {{OPTIONS}}
            </options>"""

    _logger.debug(f"Generating decision given options: {options} and prompt: {prompt[:20]}..")
    start = time.perf_counter()

    if instructions:
        messages = [
            {
                "content": "Pick one of the following options: <options>{options}</options>, given the following query:\n<query>{prompt}</query>.",
                "role": "user",
            },
        ]
    else:
        messages = [
            {
                "content": DEFAULT_SYSTEM_PROMPT.strip()
                .replace("{{CONTEXT}}", prompt)
                .replace("{{OPTIONS}}", "\n".join(options)),
                "role": "user",
            },
        ]

    response = completion(
        api_key=config.api_key,
        model=config.model,
        messages=messages,
        response_format=Decision,
        **config.llm.model_dump(),
    )

    _logger.debug(f"Generated decision in {time.perf_counter() - start:.2f}s")
    return Decision.model_validate(json.loads(response.choices[0].message.content))
