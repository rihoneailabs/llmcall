# LLMCall

A lite abstraction layer for LLM calls.

## Motivation

As AI becomes more prevalent in software development, there's a growing need for simple and intuitive APIs for interacting with AI for quick text generation, decision making, and more. This is especially important now that we have structured outputs, which allow us to seamlessly integrate AI into our application flow.

`llmcall` provides a minimal, batteries-included interface for common LLM operations without unnecessary complexity.

## Installation

```bash
pip install llmcall
```

## Quick Start

```bash
# 1. Install
pip install llmcall

# 2. Set your API key (copy .env.example to .env and fill in your key)
cp .env.example .env
```

```python
# 3. Use it
from llmcall import generate

response = generate("What is the capital of France?")
print(response)  # Paris
```

## Configuration

Copy `.env.example` to `.env` and set your values:

```bash
# Required
LLMCALL_API_KEY=sk-...

# Optional (defaults shown)
LLMCALL_MODEL=openai/gpt-4o-2024-08-06
LLMCALL_BASE_URL=          # for Ollama, Azure, LM Studio, etc.
LLMCALL_DEBUG=false
```

Or set environment variables directly:

```bash
export LLMCALL_API_KEY=sk-...
```

> Uses [LiteLLM](https://docs.litellm.ai/docs/providers) under the hood — any supported provider works. We recommend OpenAI models for structured outputs.

### Using local models (Ollama)

```bash
LLMCALL_MODEL=ollama/llama3.2
LLMCALL_BASE_URL=http://localhost:11434
LLMCALL_API_KEY=ollama
```

## Example Usage

### Generation

```python
from llmcall import generate, generate_decision
from pydantic import BaseModel

# i. Basic generation
response = generate("Write a story about a fictional holiday to the sun.")

# ii. Structured generation
class ResponseSchema(BaseModel):
    story: str
    tags: list[str]

response: ResponseSchema = generate(
    "Create a rare story about the history of civilisation.",
    output_schema=ResponseSchema,
)

# iii. Streaming — get tokens as they arrive
for chunk in generate("Tell me a joke.", stream=True):
    print(chunk, end="", flush=True)

# iv. Decision making
decision = generate_decision(
    "Which is bigger?",
    options=["apple", "berry", "pumpkin"],
)
print(decision.selection)  # pumpkin
print(decision.reason)     # Pumpkins are significantly larger than...
```

### Async generation (FastAPI, async frameworks)

```python
import asyncio
from llmcall import agenerate, agenerate_decision, aextract

# Async generate
response = await agenerate("Write a story about a fictional holiday to the sun.")

# Async streaming
async for chunk in await agenerate("Tell me a joke.", stream=True):
    print(chunk, end="", flush=True)

# Async decision
decision = await agenerate_decision("Which is bigger?", options=["apple", "berry", "pumpkin"])

# Async extract
result = await aextract(text=my_text, output_schema=MySchema)

# Run concurrently
story, decision = await asyncio.gather(
    agenerate("Write a story."),
    agenerate_decision("Which language?", options=["Python", "Go", "Rust"]),
)
```

### Extraction

```python
from llmcall import extract
from pydantic import BaseModel

class ResponseSchema(BaseModel):
    email_subject: str
    email_body: str
    email_topic: str
    email_sentiment: str

text = """To whom it may concern,

Request for Admission at Harvard University

I write to plead with the admission board to consider my application for the 2022/2023 academic year. I am a dedicated student with a passion for computer science and a strong desire to make a difference in the world. I believe that Harvard University is the perfect place for me to achieve my dreams and make a positive impact on society."""

response: ResponseSchema = extract(text=text, output_schema=ResponseSchema)
```

## Roadmap

- [x] Simple API for generating unstructured text
- [x] Structured output generation using `Pydantic`
- [x] Decision making
- [x] Custom model selection (via `LiteLLM` - See [documentation](https://docs.litellm.ai/docs/providers))
- [x] Custom base URL for OpenAI-compatible endpoints (Ollama, Azure, LM Studio)
- [x] Structured text extraction
- [ ] Structured text extraction from PDF, Docx, etc.
- [ ] Structured text extraction from Images
- [ ] Structured text extraction from Websites
- [x] Async support (`agenerate`, `aextract`, `agenerate_decision`)
- [x] Streaming support (`generate(..., stream=True)`, `agenerate(..., stream=True)`)

## Documentation

Please refer to our comprehensive [documentation](./docs/index.md) to learn more about this tool.
