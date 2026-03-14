import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_cfg():
    return types.SimpleNamespace(
        api_key="test-key",
        model="openai/gpt-4.1",
        base_url=None,
        llm=types.SimpleNamespace(
            temperature=0.0,
            stream=False,
            n=1,
            max_output_tokens=256,
            num_retries=0,
            seed=7,
        ),
    )


@pytest.fixture
def mock_text_response():
    def _build(content: str):
        msg = MagicMock()
        msg.content = content
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    return _build
