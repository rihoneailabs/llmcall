# llmcall

A simple abstraction layer for LLM calls.

## Installation

```bash
pip install llmcall
```

## Usage

```python
from llmcall import generate, generate_decision

# Basic generation
response = generate("Write a story about...")

# Structured generation
schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"}
    }
}
response = generate("Create a story...", output_schema=schema)

# Decision making
decision = generate_decision(
    "Should the character be good or evil?",
    options=["good", "evil"]
)
```

## Configuration

Set environment variables:
- LLMCALL_API_KEY: Your API key
- LLMCALL_MODEL: Model to use (default: gpt-4)
