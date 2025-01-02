import os
from llmcall import generate


os.environ["LITELLM_LOG"] = "DEBUG"


def test_generate_one_word_answer():
    prompt = "What is the capital of France?"
    response = generate(prompt)
    assert response.lower() == "paris"
