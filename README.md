# LLMCall

A lite abstraction layer for LLM calls.

## Motivation

As AI becomes more prevalent in software development, there's a growing need for simple and intuitive APIs for \
interacting with AI for quick text generation, decision making, and more. This is especially important now that we \
have structured outputs, which allow us to seamlessly integrate AI into our application flow.

`llmcall` provides a minimal intelligence interface for common LLM operations without unnecessary complexity.

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
LLMCALL_MODEL=openai/gpt-4.1
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

### Async generation

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
from llmcall import extract, extract_pdf, extract_image
from pydantic import BaseModel

class EmailSchema(BaseModel):
    email_subject: str
    email_body: str
    email_topic: str
    email_sentiment: str

# i. Extract from plain text
text = """To whom it may concern, Request for Admission at Harvard University ..."""
result: EmailSchema = extract(text=text, output_schema=EmailSchema)

# ii. Extract from a PDF — URL, local path, or raw bytes all work
class InvoiceSchema(BaseModel):
    vendor: str
    total: float
    line_items: list[str]

result: InvoiceSchema = extract_pdf(
    source="https://example.com/invoice.pdf",
    output_schema=InvoiceSchema,
)
# local file
result: InvoiceSchema = extract_pdf(source="/path/to/invoice.pdf", output_schema=InvoiceSchema)
# raw bytes
with open("invoice.pdf", "rb") as f:
    result: InvoiceSchema = extract_pdf(source=f.read(), output_schema=InvoiceSchema)

# iii. Extract from an image — URL, local path, or raw bytes all work
class ReceiptSchema(BaseModel):
    store: str
    total: float
    items: list[str]

result: ReceiptSchema = extract_image(
    source="https://example.com/receipt.jpg",
    output_schema=ReceiptSchema,
)
# local PNG (MIME type auto-detected from extension)
result: ReceiptSchema = extract_image(source="/path/to/receipt.png", output_schema=ReceiptSchema)
# raw bytes with explicit MIME type
with open("receipt.webp", "rb") as f:
    result: ReceiptSchema = extract_image(source=f.read(), output_schema=ReceiptSchema, media_type="image/webp")
```

> **Model requirements:** PDF extraction requires a model with document-understanding support
> (e.g. `anthropic/claude-sonnet-4-6`, `openai/gpt-4.1`, `google/gemini-3-flash-preview`).
> Image extraction requires a vision-capable model. An informative `ValueError` is raised if
> the configured model does not support the required capability.

### Async multimodal extraction

```python
from llmcall import aextract_pdf, aextract_image
import asyncio

invoice, receipt = await asyncio.gather(
    aextract_pdf("https://example.com/invoice.pdf", InvoiceSchema),
    aextract_image("https://example.com/receipt.jpg", ReceiptSchema),
)
```

## Roadmap

- [x] Simple API for generating unstructured text
- [x] Structured output generation using `Pydantic`
- [x] Decision making
- [x] Custom model selection (via `LiteLLM` - See [documentation](https://docs.litellm.ai/docs/providers))
- [x] Custom base URL for OpenAI-compatible endpoints (Ollama, Azure, LM Studio)
- [x] Structured text extraction
- [x] Structured extraction from PDF (URL, local path, or bytes)
- [x] Structured extraction from Images (URL, local path, or bytes)
- [ ] Structured text extraction from Websites
- [x] Async support (`agenerate`, `aextract`, `aextract_pdf`, `aextract_image`, `agenerate_decision`)
- [x] Streaming support (`generate(..., stream=True)`, `agenerate(..., stream=True)`)

## Documentation

Please refer to our comprehensive [documentation](./docs/index.md) to learn more about this tool.
