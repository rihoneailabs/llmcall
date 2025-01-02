# LLMCall

A lite abstraction layer for LLM calls.

## Motivation

As AI becomes more prevalent in software development, there's a growing need for simple and intuitive APIs for interacting with AI for quick text generation, decision making, and more. This is especially important now that we have structured outputs, which allow us to seamlessly integrate AI into our application flow.

`llmcall` provides a minimal, batteries-included interface for common LLM operations without unnecessary complexity.

## Installation

```bash
pip install llmcall
```

## Example Usage

### Generation

```python
from llmcall import generate, generate_decision
from pydantic import BaseModel

# Basic generation
response = generate("Write a story about...")

# Structured generation
class ResponseSchema(BaseModel):
    story: str
    tags: list[str]
    
response: ResponseSchema = generate("Create a story...", output_schema=schema)

# Decision making
decision = generate_decision(
    "Should the character be good or evil?",
    options=["good", "evil"]
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

response: ResponseSchema = extract(text=<some-email-text>, output_schema=ResponseSchema)
```

## Configuration

Set environment variables:
- LLMCALL_API_KEY: Your API key
- LLMCALL_MODEL: Model to use (default: `openai/gpt-4o-2024-08-06`)


## Features

- [x] Simple API for generating unstructured text
- [x] Structured output generation using `Pydantic`
- [x] Decision making
- [x] Custom model selection (via `LiteLLM` - See [documentation](https://docs.litellm.ai/docs/providers))

## Documentation

Please refer to our comprehensive [documentation](./docs/index.md) to learn more about this tool.
