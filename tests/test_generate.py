import os
from typing import Annotated

from litellm.exceptions import BadRequestError
from pydantic import BaseModel


from llmcall import generate, generate_decision

os.environ["LITELLM_LOG"] = "DEBUG"


def test_generate_one_word_answer():
    prompt = "What is the capital of France?"
    response = generate(prompt)
    assert "paris" in response.lower().strip()


def test_generate_one_word_answer_exact():
    class ResponseFormat(BaseModel):
        name: Annotated[str, "the name of the capital"]

    prompt = "What is the capital of France?"
    response = generate(prompt, output_schema=ResponseFormat)
    assert response.name.lower() == "paris"


def test_simple_decision():
    prompt = "Which is bigger?"
    options = ["apple", "pumpkin", "nut"]
    decision = generate_decision(prompt, options)
    assert decision.selection.lower() == "pumpkin"


def test_decision_is_in_options():
    prompt = "Which language is better for data science?"
    options = ["Python", "R", "Julia"]
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


def test_invalid_schema_model():
    try:

        class InvalidSchema(BaseModel):
            invalid_field: dict  # Unsupported complex type

        prompt = "Test prompt"
        generate(prompt, output_schema=InvalidSchema)
        assert False, "Should raise ValueError for unsupported schema"
    except BadRequestError:
        pass
