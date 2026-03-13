import json
import logging
import time
from typing import AsyncIterator, Iterator, Optional, Union

from litellm import acompletion, completion, supports_response_schema
from pydantic import BaseModel
from typing_extensions import Annotated

from llmcall.core import get_config

_logger = logging.getLogger(__name__)


class Decision(BaseModel):
    selection: Annotated[
        str, "The selected option - MUST be one of the provided options."
    ]
    prompt: Optional[str] = None
    options: Optional[list[str]] = None
    reason: Optional[str] = None


def generate(
    prompt: Annotated[str, "The user prompt which tells the model what to generate."],
    output_schema: Annotated[
        Optional[BaseModel],
        "The Pydantic model to use for response structure validation(optional)",
    ] = None,
    instructions: Annotated[
        Optional[str], "System metaprompt to condition the model."
    ] = None,
    stream: Annotated[bool, "Stream the response token by token."] = False,
) -> Union[str, BaseModel, Iterator[str]]:
    """Generate content using configured LLM.

    When stream=True, returns an iterator of string chunks instead of a full response.
    Streaming is not supported with output_schema.
    """

    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    if stream and output_schema:
        raise ValueError("Streaming is not supported with output_schema.")

    cfg = get_config()

    DEFAULT_SYSTEM_PROMPT = (
        "Generate content based on the following: <prompt>{prompt}</prompt>. \
        Return only the content with no additional information or comments."
    )

    _logger.debug(f"Generating content for prompt: {prompt[:20]}..")
    start = time.perf_counter()

    extra_kwargs = {}
    if output_schema:
        if not supports_response_schema(
            model=cfg.model.split("/")[1], custom_llm_provider=cfg.model.split("/")[0]
        ):
            raise ValueError(
                f"Response schema is not supported by the configured model: {cfg.model}. "
                "Please use a different model(e.g. openai/gpt-4o-2024-08-06) or remove the output schema."
            )
        extra_kwargs["response_format"] = output_schema
        extra_kwargs["json_schema_validation"] = True

    response = completion(
        api_key=cfg.api_key,
        model=cfg.model,
        base_url=cfg.base_url,
        messages=[
            {"content": instructions or DEFAULT_SYSTEM_PROMPT, "role": "system"},
            {"content": prompt, "role": "user"},
        ],
        temperature=cfg.llm.temperature,
        stream=stream,
        n=cfg.llm.n,
        max_tokens=cfg.llm.max_tokens,
        num_retries=cfg.llm.num_retries,
        seed=cfg.llm.seed,
        **extra_kwargs,
    )

    _logger.debug(f"Generated content in {time.perf_counter() - start:.2f}s")

    if stream:
        return (chunk.choices[0].delta.content or "" for chunk in response)

    if output_schema:
        return output_schema.model_validate(
            json.loads(response.choices[0].message.content), strict=True
        )

    return response.choices[0].message.content


async def agenerate(
    prompt: Annotated[str, "The user prompt which tells the model what to generate."],
    output_schema: Annotated[
        Optional[BaseModel],
        "The Pydantic model to use for response structure validation(optional)",
    ] = None,
    instructions: Annotated[
        Optional[str], "System metaprompt to condition the model."
    ] = None,
    stream: Annotated[bool, "Stream the response token by token."] = False,
) -> Union[str, BaseModel, AsyncIterator[str]]:
    """Async version of generate().

    When stream=True, returns an async iterator of string chunks instead of a full response.
    Streaming is not supported with output_schema.
    """

    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    if stream and output_schema:
        raise ValueError("Streaming is not supported with output_schema.")

    cfg = get_config()

    DEFAULT_SYSTEM_PROMPT = (
        "Generate content based on the following: <prompt>{prompt}</prompt>. \
        Return only the content with no additional information or comments."
    )

    _logger.debug(f"Generating content (async) for prompt: {prompt[:20]}..")
    start = time.perf_counter()

    extra_kwargs = {}
    if output_schema:
        if not supports_response_schema(
            model=cfg.model.split("/")[1], custom_llm_provider=cfg.model.split("/")[0]
        ):
            raise ValueError(
                f"Response schema is not supported by the configured model: {cfg.model}. "
                "Please use a different model(e.g. openai/gpt-4o-2024-08-06) or remove the output schema."
            )
        extra_kwargs["response_format"] = output_schema
        extra_kwargs["json_schema_validation"] = True

    response = await acompletion(
        api_key=cfg.api_key,
        model=cfg.model,
        base_url=cfg.base_url,
        messages=[
            {"content": instructions or DEFAULT_SYSTEM_PROMPT, "role": "system"},
            {"content": prompt, "role": "user"},
        ],
        temperature=cfg.llm.temperature,
        stream=stream,
        n=cfg.llm.n,
        max_tokens=cfg.llm.max_tokens,
        num_retries=cfg.llm.num_retries,
        seed=cfg.llm.seed,
        **extra_kwargs,
    )

    _logger.debug(f"Generated content (async) in {time.perf_counter() - start:.2f}s")

    if stream:

        async def _stream_chunks():
            async for chunk in response:
                yield chunk.choices[0].delta.content or ""

        return _stream_chunks()

    if output_schema:
        return output_schema.model_validate(
            json.loads(response.choices[0].message.content), strict=True
        )

    return response.choices[0].message.content


def generate_decision(
    prompt: Annotated[str, "The context to consider when making the decision."],
    options: Annotated[list[str], "List of options to choose from."],
    instructions: Annotated[
        Optional[str], "System metaprompt to condition the model."
    ] = None,
) -> Decision:
    """Generate a decision from a list of options."""

    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    cfg = get_config()

    DEFAULT_SYSTEM_PROMPT = """You are a specialized computer algorithm designed to make decisions in Control Flow scenarios. \
        Your task is to analyze the given context and options, then select the most appropriate option based on the context. Here is the context you need to consider: \
            <context>
            {{CONTEXT}}
            </context>

        Here are the options you can choose from:
            <options>
            {{OPTIONS}}
            </options>"""

    _logger.debug(
        f"Generating decision given options: {options} and prompt: {prompt[:20]}.."
    )
    start = time.perf_counter()

    if instructions:
        messages = [
            {
                "content": instructions,
                "role": "system",
            },
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
        api_key=cfg.api_key,
        model=cfg.model,
        base_url=cfg.base_url,
        messages=messages,
        response_format=Decision,
        temperature=cfg.llm.temperature,
        stream=cfg.llm.stream,
        n=cfg.llm.n,
        max_tokens=cfg.llm.max_tokens,
        num_retries=cfg.llm.num_retries,
        seed=cfg.llm.seed,
    )

    _logger.debug(f"Generated decision in {time.perf_counter() - start:.2f}s")
    return Decision.model_validate(
        json.loads(response.choices[0].message.content), strict=True
    )


async def agenerate_decision(
    prompt: Annotated[str, "The context to consider when making the decision."],
    options: Annotated[list[str], "List of options to choose from."],
    instructions: Annotated[
        Optional[str], "System metaprompt to condition the model."
    ] = None,
) -> Decision:
    """Async version of generate_decision()."""

    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    cfg = get_config()

    DEFAULT_SYSTEM_PROMPT = """You are a specialized computer algorithm designed to make decisions in Control Flow scenarios. \
        Your task is to analyze the given context and options, then select the most appropriate option based on the context. Here is the context you need to consider: \
            <context>
            {{CONTEXT}}
            </context>

        Here are the options you can choose from:
            <options>
            {{OPTIONS}}
            </options>"""

    _logger.debug(
        f"Generating decision (async) given options: {options} and prompt: {prompt[:20]}.."
    )
    start = time.perf_counter()

    if instructions:
        messages = [
            {
                "content": instructions,
                "role": "system",
            },
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

    response = await acompletion(
        api_key=cfg.api_key,
        model=cfg.model,
        base_url=cfg.base_url,
        messages=messages,
        response_format=Decision,
        temperature=cfg.llm.temperature,
        stream=cfg.llm.stream,
        n=cfg.llm.n,
        max_tokens=cfg.llm.max_tokens,
        num_retries=cfg.llm.num_retries,
        seed=cfg.llm.seed,
    )

    _logger.debug(f"Generated decision (async) in {time.perf_counter() - start:.2f}s")
    return Decision.model_validate(
        json.loads(response.choices[0].message.content), strict=True
    )
