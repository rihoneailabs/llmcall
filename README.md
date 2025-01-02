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

# i. Basic generation
response = generate("Write a story about a fictional holiday to the sun.")

# ii. Structured generation
class ResponseSchema(BaseModel):
    story: str
    tags: list[str]
    
response: ResponseSchema = generate("Create a rare story about the history of civilisation.", output_schema=schema)

# iii. Decision making
decision = generate_decision(
    "Which is bigger?",
    options=["apple", "berry", "pumpkin"]
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

## Configuration

Set environment variables:
- LLMCALL_API_KEY: Your API key
- LLMCALL_MODEL: Model to use (default: `openai/gpt-4o-2024-08-06`)

> **Note**: We recommend using `Open AI` as the model provider due to their robust support for structured outputs. You can use other providers by setting the `LLMCALL_MODEL` or changing the [config](./llmcall/core.py) directly. Any model supported by `LiteLLM` can be used.

## Roadmap

- [x] Simple API for generating unstructured text
- [x] Structured output generation using `Pydantic`
- [x] Decision making
- [x] Custom model selection (via `LiteLLM` - See [documentation](https://docs.litellm.ai/docs/providers))
- [x] Structured text extraction
- [ ] Structured text extraction from PDF, Docx, etc.
- [ ] Structured text extraction from Images
- [ ] Structured text extraction from Websites

## Documentation

Please refer to our comprehensive [documentation](./docs/index.md) to learn more about this tool.
