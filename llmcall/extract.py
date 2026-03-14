import base64
import json
import logging
import mimetypes
import time
from pathlib import Path
from typing import Optional, Union

from litellm import acompletion, completion
from litellm.utils import supports_pdf_input, supports_vision
from pydantic import BaseModel
from typing_extensions import Annotated

from llmcall.core import get_config

_logger = logging.getLogger(__name__)
_Source = Union[str, Path, bytes]

_DEFAULT_EXTRACT_SYSTEM_PROMPT = (
    "You are a specialist in organising unstructured data. Given the document below, "
    "your task is to extract the requested information as accurately as you can. "
    "###\n<document>{text}</document>\n###"
)

_DEFAULT_MULTIMODAL_SYSTEM_PROMPT = (
    "You are a specialist in organising unstructured data. "
    "Your task is to extract the requested information from the provided document as accurately as you can."
)


def _source_to_data_uri(source: _Source, mime_type: str) -> str:
    if isinstance(source, (str, Path)):
        with open(source, "rb") as f:
            raw = f.read()
    else:
        raw = source
    encoded = base64.standard_b64encode(raw).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _is_url(source: _Source) -> bool:
    return isinstance(source, str) and source.startswith(("http://", "https://"))


def _build_pdf_content_block(source: _Source) -> dict:
    if _is_url(source):
        return {
            "type": "file",
            "file": {"file_id": source, "format": "application/pdf"},
        }
    data_uri = _source_to_data_uri(source, "application/pdf")
    return {"type": "file", "file": {"file_data": data_uri}}


def _detect_image_mime(source: _Source) -> str:
    if isinstance(source, (str, Path)):
        guessed, _ = mimetypes.guess_type(str(source))
        if guessed and guessed.startswith("image/"):
            return guessed
    return "image/jpeg"


def _build_image_content_block(
    source: _Source, media_type: Optional[str] = None
) -> dict:
    if _is_url(source):
        return {"type": "image_url", "image_url": {"url": source}}
    mime = media_type or _detect_image_mime(source)
    data_uri = _source_to_data_uri(source, mime)
    return {"type": "image_url", "image_url": {"url": data_uri}}


def _run_completion(cfg, messages, output_schema):
    return completion(
        api_key=cfg.api_key,
        model=cfg.model,
        base_url=cfg.base_url,
        messages=messages,
        response_format=output_schema,
        temperature=cfg.llm.temperature,
        stream=False,
        n=cfg.llm.n,
        max_tokens=cfg.llm.max_output_tokens,
        num_retries=cfg.llm.num_retries,
        seed=cfg.llm.seed,
    )


async def _run_acompletion(cfg, messages, output_schema):
    return await acompletion(
        api_key=cfg.api_key,
        model=cfg.model,
        base_url=cfg.base_url,
        messages=messages,
        response_format=output_schema,
        temperature=cfg.llm.temperature,
        stream=False,
        n=cfg.llm.n,
        max_tokens=cfg.llm.max_output_tokens,
        num_retries=cfg.llm.num_retries,
        seed=cfg.llm.seed,
    )


def _parse_response(response, output_schema):
    return output_schema.model_validate(
        json.loads(response.choices[0].message.content), strict=True
    )


def extract(
    text: Annotated[str, "The unstructured text to extract information from."],
    output_schema: Annotated[
        BaseModel, "The Pydantic model to use for response structure validation."
    ],
    instructions: Annotated[
        Optional[str], "System metaprompt to condition the model."
    ] = None,
) -> BaseModel:
    if not text:
        raise ValueError("Text cannot be empty.")

    cfg = get_config()
    start = time.perf_counter()
    _logger.info(f"Extracting information from text: {text[:50]}")

    if instructions:
        messages = [
            {"content": instructions, "role": "system"},
            {"content": text, "role": "user"},
        ]
    else:
        messages = [
            {
                "content": _DEFAULT_EXTRACT_SYSTEM_PROMPT.format(text=text),
                "role": "system",
            },
        ]

    response = _run_completion(cfg, messages, output_schema)
    _logger.info(f"Extraction completed in {time.perf_counter() - start:.2f} seconds.")
    return _parse_response(response, output_schema)


async def aextract(
    text: Annotated[str, "The unstructured text to extract information from."],
    output_schema: Annotated[
        BaseModel, "The Pydantic model to use for response structure validation."
    ],
    instructions: Annotated[
        Optional[str], "System metaprompt to condition the model."
    ] = None,
) -> BaseModel:
    if not text:
        raise ValueError("Text cannot be empty.")

    cfg = get_config()
    start = time.perf_counter()
    _logger.info(f"Extracting information (async) from text: {text[:50]}")

    if instructions:
        messages = [
            {"content": instructions, "role": "system"},
            {"content": text, "role": "user"},
        ]
    else:
        messages = [
            {
                "content": _DEFAULT_EXTRACT_SYSTEM_PROMPT.format(text=text),
                "role": "system",
            },
        ]

    response = await _run_acompletion(cfg, messages, output_schema)
    _logger.info(
        f"Extraction (async) completed in {time.perf_counter() - start:.2f} seconds."
    )
    return _parse_response(response, output_schema)



def extract_pdf(
    source: Annotated[
        _Source,
        "PDF source: a URL string, a local file path (str or Path), or raw PDF bytes.",
    ],
    output_schema: Annotated[
        BaseModel, "The Pydantic model to use for response structure validation."
    ],
    instructions: Annotated[
        Optional[str], "System metaprompt to condition the model."
    ] = None,
) -> BaseModel:
    """Extract structured information from a PDF using the configured LLM.

    The model must support PDF input (e.g. claude-sonnet-4-6, gpt-4.1, gemini-3-flash-preview).
    Raises ValueError if the configured model does not support PDF input.
    """
    cfg = get_config()
    provider, model_name = cfg.model.split("/", 1)

    if not supports_pdf_input(model=model_name, custom_llm_provider=provider):
        raise ValueError(
            f"PDF input is not supported by the configured model: {cfg.model}. "
            "Use a model that supports document understanding (e.g. anthropic/claude-sonnet-4-6, "
            "openai/gpt-4.1, google/gemini-3-flash-preview)."
        )

    start = time.perf_counter()
    _logger.info("Extracting information from PDF source.")

    pdf_block = _build_pdf_content_block(source)
    system = instructions or _DEFAULT_MULTIMODAL_SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract the requested information from the PDF above.",
                },
                pdf_block,
            ],
        },
    ]

    response = _run_completion(cfg, messages, output_schema)
    _logger.info(
        f"PDF extraction completed in {time.perf_counter() - start:.2f} seconds."
    )
    return _parse_response(response, output_schema)


async def aextract_pdf(
    source: Annotated[
        _Source,
        "PDF source: a URL string, a local file path (str or Path), or raw PDF bytes.",
    ],
    output_schema: Annotated[
        BaseModel, "The Pydantic model to use for response structure validation."
    ],
    instructions: Annotated[
        Optional[str], "System metaprompt to condition the model."
    ] = None,
) -> BaseModel:
    cfg = get_config()
    provider, model_name = cfg.model.split("/", 1)

    if not supports_pdf_input(model=model_name, custom_llm_provider=provider):
        raise ValueError(
            f"PDF input is not supported by the configured model: {cfg.model}. "
            "Use a model that supports document understanding (e.g. anthropic/claude-sonnet-4-6, "
            "openai/gpt-4.1, google/gemini-3-flash-preview)."
        )

    start = time.perf_counter()
    _logger.info("Extracting information (async) from PDF source.")

    pdf_block = _build_pdf_content_block(source)
    system = instructions or _DEFAULT_MULTIMODAL_SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract the requested information from the PDF above.",
                },
                pdf_block,
            ],
        },
    ]

    response = await _run_acompletion(cfg, messages, output_schema)
    _logger.info(
        f"PDF extraction (async) completed in {time.perf_counter() - start:.2f} seconds."
    )
    return _parse_response(response, output_schema)


def extract_image(
    source: Annotated[
        _Source,
        "Image source: a URL string, a local file path (str or Path), or raw image bytes.",
    ],
    output_schema: Annotated[
        BaseModel, "The Pydantic model to use for response structure validation."
    ],
    instructions: Annotated[
        Optional[str], "System metaprompt to condition the model."
    ] = None,
    media_type: Annotated[
        Optional[str],
        "MIME type for bytes/path input, e.g. 'image/png'. Auto-detected when omitted.",
    ] = None,
) -> BaseModel:
    """Extract structured information from an image using the configured LLM.

    The model must support vision (e.g. claude-sonnet-4-6, gpt-4.1, gemini-3-flash-preview).
    Raises ValueError if the configured model does not support vision.
    """
    cfg = get_config()
    provider, model_name = cfg.model.split("/", 1)

    if not supports_vision(model=model_name, custom_llm_provider=provider):
        raise ValueError(
            f"Vision/image input is not supported by the configured model: {cfg.model}. "
            "Use a vision-capable model (e.g. anthropic/claude-sonnet-4-6, "
            "openai/gpt-4.1, google/gemini-3-flash-preview)."
        )

    start = time.perf_counter()
    _logger.info("Extracting information from image source.")

    image_block = _build_image_content_block(source, media_type)
    system = instructions or _DEFAULT_MULTIMODAL_SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract the requested information from the image above.",
                },
                image_block,
            ],
        },
    ]

    response = _run_completion(cfg, messages, output_schema)
    _logger.info(
        f"Image extraction completed in {time.perf_counter() - start:.2f} seconds."
    )
    return _parse_response(response, output_schema)


async def aextract_image(
    source: Annotated[
        _Source,
        "Image source: a URL string, a local file path (str or Path), or raw image bytes.",
    ],
    output_schema: Annotated[
        BaseModel, "The Pydantic model to use for response structure validation."
    ],
    instructions: Annotated[
        Optional[str], "System metaprompt to condition the model."
    ] = None,
    media_type: Annotated[
        Optional[str],
        "MIME type for bytes/path input, e.g. 'image/png'. Auto-detected when omitted.",
    ] = None,
) -> BaseModel:
    cfg = get_config()
    provider, model_name = cfg.model.split("/", 1)

    if not supports_vision(model=model_name, custom_llm_provider=provider):
        raise ValueError(
            f"Vision/image input is not supported by the configured model: {cfg.model}. "
            "Use a vision-capable model (e.g. anthropic/claude-sonnet-4-6, "
            "openai/gpt-4.1, google/gemini-3-flash-preview)."
        )

    start = time.perf_counter()
    _logger.info("Extracting information (async) from image source.")

    image_block = _build_image_content_block(source, media_type)
    system = instructions or _DEFAULT_MULTIMODAL_SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract the requested information from the image above.",
                },
                image_block,
            ],
        },
    ]

    response = await _run_acompletion(cfg, messages, output_schema)
    _logger.info(
        f"Image extraction (async) completed in {time.perf_counter() - start:.2f} seconds."
    )
    return _parse_response(response, output_schema)
