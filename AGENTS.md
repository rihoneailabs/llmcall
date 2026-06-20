# AGENTS.md

Guidance for AI agents working in the `llmcall` repository — a lite abstraction layer over [LiteLLM](https://docs.litellm.ai) for structured LLM calls (generation, decisions, multimodal extraction).

## Project Facts

| Aspect | Value |
|--------|-------|
| Language | Python `>=3.11, <3.14` |
| Package manager | Poetry |
| Lint/format | Ruff (line-length 88, double quotes) |
| Tests | pytest (+ tox for the version matrix) |
| Public API | Defined in [llmcall/__init__.py](llmcall/__init__.py) — keep `__all__` in sync |
| Config | `LLMConfig` in [llmcall/core.py](llmcall/core.py), env prefix `LLMCALL_`, `.env` supported |

## Build and Test

```bash
poetry install
poetry run pytest              # full suite
poetry run ruff check .        # lint
poetry run ruff format .       # format
tox                            # run across py311/py312/py313
```

Run `poetry run pytest && poetry run ruff check . && poetry run ruff format . --check` before pushing.

## Conventions

- **No inline comments, no docstrings by default.** Code should be self-documenting; add a comment only for genuinely non-obvious intent. Do not add comments, docstrings, or type annotations to code you did not change.
- **Provider-aware model handling.** Model names are LiteLLM-style (`provider/model`, e.g. `openai/gpt-4.1`) and may omit the provider. Use `split_model()` from [llmcall/core.py](llmcall/core.py) before any feature check (`supports_response_schema`, `supports_pdf_input`, `supports_vision`).
- **Omit unset completion params.** Pass optional params (e.g. `seed`) via `optional_completion_params()` rather than sending `None`.
- **Multimodal message order.** File/image content blocks must precede the instruction text block in the `user` message.
- **Structured-output calls need a system + user split.** For decision/extraction prompts, keep instructions in a `system` message and the payload in a `user` message (some providers require this).

## Testing Notes

- Unit tests mock `completion`/`acompletion`; no network access required.
- Integration tests in [tests/test_integration.py](tests/test_integration.py) are skipped unless `LLMCALL_API_KEY` is set. CI forwards `secrets.LLMCALL_API_KEY` in [.github/workflows/tests.yml](.github/workflows/tests.yml).

## Git Workflow

- `main` is protected — **all changes must go through a pull request**, never push directly to `main`.
- Never use `git rebase`; use merge (`git pull`, `git pull --ff-only`).
- CI runs the tox matrix and `pip-audit` on PRs to `main` ([.github/workflows/tests.yml](.github/workflows/tests.yml)).
