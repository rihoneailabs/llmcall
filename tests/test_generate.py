import os
from typing import Annotated
from unittest.mock import patch

from litellm.exceptions import BadRequestError
from pydantic import BaseModel

from llmcall import generate, generate_decision

os.environ["LITELLM_LOG"] = "DEBUG"


def test_generate_one_word_answer(mock_cfg, mock_text_response):
    prompt = "What is the capital of France?"
    with patch("llmcall.generate.get_config", return_value=mock_cfg), patch(
        "llmcall.generate.completion", return_value=mock_text_response("Paris")
    ):
        response = generate(prompt)
    assert "paris" in response.lower().strip()


def test_generate_one_word_answer_exact(mock_cfg, mock_text_response):
    class ResponseFormat(BaseModel):
        name: Annotated[str, "the name of the capital"]

    prompt = "What is the capital of France?"
    with patch("llmcall.generate.get_config", return_value=mock_cfg), patch(
        "llmcall.generate.completion",
        return_value=mock_text_response('{"name":"Paris"}'),
    ):
        response = generate(prompt, output_schema=ResponseFormat)
    assert response.name.lower() == "paris"


def test_simple_decision(mock_cfg, mock_text_response):
    prompt = "Which is bigger?"
    options = ["apple", "pumpkin", "nut"]
    with patch("llmcall.generate.get_config", return_value=mock_cfg), patch(
        "llmcall.generate.completion",
        return_value=mock_text_response('{"selection":"pumpkin","reason":"Pumpkins are bigger."}'),
    ):
        decision = generate_decision(prompt, options)
    assert decision.selection.lower() == "pumpkin"


def test_decision_is_in_options(mock_cfg, mock_text_response):
    prompt = "Which language is better for data science?"
    options = ["Python", "R", "Julia"]
    with patch("llmcall.generate.get_config", return_value=mock_cfg), patch(
        "llmcall.generate.completion",
        return_value=mock_text_response('{"selection":"Python","reason":"Python has the broadest ecosystem for data science."}'),
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
        with patch("llmcall.generate.get_config", return_value=mock_cfg), patch(
            "llmcall.generate.completion",
            side_effect=BadRequestError(
                message="Unsupported schema",
                model="openai/gpt-4.1",
                llm_provider="openai",
            ),
        ):
            generate(prompt, output_schema=InvalidSchema)
        assert False, "Should raise ValueError for unsupported schema"
    except BadRequestError:
        pass
