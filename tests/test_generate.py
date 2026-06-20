import asyncio
import os
import types
from typing import Annotated
from unittest.mock import AsyncMock, patch

from litellm.exceptions import BadRequestError
from pydantic import BaseModel

from llmcall import agenerate_decision, generate, generate_decision

os.environ["LITELLM_LOG"] = "DEBUG"


def test_generate_one_word_answer(mock_cfg, mock_text_response):
    prompt = "What is the capital of France?"
    with (
        patch("llmcall.generate.get_config", return_value=mock_cfg),
        patch("llmcall.generate.completion", return_value=mock_text_response("Paris")),
    ):
        response = generate(prompt)
    assert "paris" in response.lower().strip()


def test_generate_default_instructions_do_not_include_literal_prompt(
    mock_cfg, mock_text_response
):
    with (
        patch("llmcall.generate.get_config", return_value=mock_cfg),
        patch(
            "llmcall.generate.completion",
            return_value=mock_text_response("Paris"),
        ) as completion_mock,
    ):
        generate("What is the capital of France?")

    system_message = completion_mock.call_args.kwargs["messages"][0]["content"]
    assert "{prompt}" not in system_message


def test_generate_one_word_answer_exact(mock_cfg, mock_text_response):
    class ResponseFormat(BaseModel):
        name: Annotated[str, "the name of the capital"]

    prompt = "What is the capital of France?"
    with (
        patch("llmcall.generate.get_config", return_value=mock_cfg),
        patch(
            "llmcall.generate.completion",
            return_value=mock_text_response('{"name":"Paris"}'),
        ),
    ):
        response = generate(prompt, output_schema=ResponseFormat)
    assert response.name.lower() == "paris"


def test_generate_schema_supports_model_without_provider(mock_cfg, mock_text_response):
    cfg = types.SimpleNamespace(**vars(mock_cfg))
    cfg.model = "gpt-4.1"

    class ResponseFormat(BaseModel):
        name: str

    with (
        patch("llmcall.generate.get_config", return_value=cfg),
        patch(
            "llmcall.generate.supports_response_schema", return_value=True
        ) as supports,
        patch(
            "llmcall.generate.completion",
            return_value=mock_text_response('{"name":"Paris"}'),
        ),
    ):
        response = generate("What is the capital of France?", ResponseFormat)

    assert response.name == "Paris"
    supports.assert_called_once_with(model="gpt-4.1", custom_llm_provider=None)


def test_generate_omits_seed_when_unset(mock_cfg, mock_text_response):
    cfg = types.SimpleNamespace(**vars(mock_cfg))
    cfg.llm = types.SimpleNamespace(**vars(mock_cfg.llm))
    cfg.llm.seed = None

    with (
        patch("llmcall.generate.get_config", return_value=cfg),
        patch(
            "llmcall.generate.completion",
            return_value=mock_text_response("Paris"),
        ) as completion_mock,
    ):
        generate("What is the capital of France?")

    assert "seed" not in completion_mock.call_args.kwargs


def test_simple_decision(mock_cfg, mock_text_response):
    prompt = "Which is bigger?"
    options = ["apple", "pumpkin", "nut"]
    with (
        patch("llmcall.generate.get_config", return_value=mock_cfg),
        patch("llmcall.generate.supports_response_schema", return_value=True),
        patch(
            "llmcall.generate.completion",
            return_value=mock_text_response(
                '{"selection":"pumpkin","reason":"Pumpkins are bigger."}'
            ),
        ),
    ):
        decision = generate_decision(prompt, options)
    assert decision.selection.lower() == "pumpkin"


def test_decision_rejects_empty_options():
    try:
        generate_decision("Pick one", [])
        assert False, "Should raise ValueError for empty options"
    except ValueError as exc:
        assert "Options cannot be empty" in str(exc)


def test_decision_rejects_selection_outside_options(mock_cfg, mock_text_response):
    with (
        patch("llmcall.generate.get_config", return_value=mock_cfg),
        patch("llmcall.generate.supports_response_schema", return_value=True),
        patch(
            "llmcall.generate.completion",
            return_value=mock_text_response('{"selection":"banana"}'),
        ),
    ):
        try:
            generate_decision("Which is bigger?", ["apple", "pumpkin"])
            assert False, "Should raise ValueError for invalid selection"
        except ValueError as exc:
            assert "provided options" in str(exc)


def test_decision_checks_response_schema_support(mock_cfg):
    with (
        patch("llmcall.generate.get_config", return_value=mock_cfg),
        patch("llmcall.generate.supports_response_schema", return_value=False),
    ):
        try:
            generate_decision("Which is bigger?", ["apple", "pumpkin"])
            assert False, "Should raise ValueError for unsupported response schema"
        except ValueError as exc:
            assert "Response schema is not supported" in str(exc)


def test_decision_custom_instructions_interpolate_prompt_and_options(
    mock_cfg, mock_text_response
):
    with (
        patch("llmcall.generate.get_config", return_value=mock_cfg),
        patch("llmcall.generate.supports_response_schema", return_value=True),
        patch(
            "llmcall.generate.completion",
            return_value=mock_text_response('{"selection":"pumpkin"}'),
        ) as completion_mock,
    ):
        generate_decision(
            "Which is bigger?",
            ["apple", "pumpkin"],
            instructions="Choose carefully.",
        )

    user_message = completion_mock.call_args.kwargs["messages"][1]["content"]
    assert "{options}" not in user_message
    assert "{prompt}" not in user_message
    assert "apple" in user_message
    assert "pumpkin" in user_message
    assert "Which is bigger?" in user_message


def test_async_decision_custom_instructions_interpolate_prompt_and_options(
    mock_cfg, mock_text_response
):
    async def _run():
        with (
            patch("llmcall.generate.get_config", return_value=mock_cfg),
            patch("llmcall.generate.supports_response_schema", return_value=True),
            patch(
                "llmcall.generate.acompletion",
                new_callable=AsyncMock,
                return_value=mock_text_response('{"selection":"pumpkin"}'),
            ) as completion_mock,
        ):
            await agenerate_decision(
                "Which is bigger?",
                ["apple", "pumpkin"],
                instructions="Choose carefully.",
            )
        return completion_mock

    completion_mock = asyncio.run(_run())
    user_message = completion_mock.call_args.kwargs["messages"][1]["content"]
    assert "{options}" not in user_message
    assert "{prompt}" not in user_message
    assert "apple" in user_message
    assert "pumpkin" in user_message
    assert "Which is bigger?" in user_message


def test_decision_is_in_options(mock_cfg, mock_text_response):
    prompt = "Which language is better for data science?"
    options = ["Python", "R", "Julia"]
    with (
        patch("llmcall.generate.get_config", return_value=mock_cfg),
        patch("llmcall.generate.supports_response_schema", return_value=True),
        patch(
            "llmcall.generate.completion",
            return_value=mock_text_response(
                '{"selection":"Python","reason":"Python has the broadest '
                'ecosystem for data science."}'
            ),
        ),
    ):
        decision = generate_decision(prompt, options)

    assert decision.reason is not None
    assert len(decision.reason) > 0
    assert decision.selection in options


def test_generate_handles_empty_prompt():
    try:
        generate("")
        assert False, "Should raise ValueError for empty prompt"
    except ValueError:
        pass


def test_invalid_schema_model(mock_cfg):
    try:

        class InvalidSchema(BaseModel):
            invalid_field: dict  # Unsupported complex type

        prompt = "Test prompt"
        with (
            patch("llmcall.generate.get_config", return_value=mock_cfg),
            patch(
                "llmcall.generate.completion",
                side_effect=BadRequestError(
                    message="Unsupported schema",
                    model="openai/gpt-4.1",
                    llm_provider="openai",
                ),
            ),
        ):
            generate(prompt, output_schema=InvalidSchema)
        assert False, "Should raise ValueError for unsupported schema"
    except BadRequestError:
        pass
