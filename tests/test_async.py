from typing import Optional

import pytest
from pydantic import BaseModel

from llmcall import aextract, agenerate, agenerate_decision, generate


def test_generate_stream_raises_with_output_schema():
    class Schema(BaseModel):
        name: str

    try:
        generate("What is the capital of France?", output_schema=Schema, stream=True)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_agenerate_raises_empty_prompt():
    import asyncio

    async def _run():
        await agenerate("")

    try:
        asyncio.run(_run())
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_agenerate_stream_raises_with_output_schema():
    import asyncio

    class Schema(BaseModel):
        name: str

    async def _run():
        await agenerate(
            "What is the capital of France?", output_schema=Schema, stream=True
        )

    try:
        asyncio.run(_run())
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_aextract_raises_empty_text():
    import asyncio

    class Schema(BaseModel):
        title: str

    async def _run():
        await aextract("", output_schema=Schema)

    try:
        asyncio.run(_run())
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_agenerate_decision_raises_empty_prompt():
    import asyncio

    async def _run():
        await agenerate_decision("", options=["a", "b"])

    try:
        asyncio.run(_run())
        assert False, "Should raise ValueError"
    except ValueError:
        pass
