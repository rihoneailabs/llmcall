import os
from typing import Annotated
from llmcall import generate

from pydantic import BaseModel


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
