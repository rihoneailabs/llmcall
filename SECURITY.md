# Security Policy

## Supported Versions

| Version       | Supported          |
| ------------- | ------------------ |
| 0.1.x (latest rc) | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in `llmcall`, please report it responsibly:

1. **Do not** open a public GitHub issue.
2. Email **hello@ndamulelo.co.za** with a description of the vulnerability, steps to reproduce, and any relevant logs or screenshots.
3. You will receive an acknowledgment within 48 hours.

## Supply Chain

`llmcall` depends on [LiteLLM](https://github.com/BerriAI/litellm), which experienced a [supply chain incident](https://docs.litellm.ai/blog/security-update-march-2026) in March 2026. As a precaution:

- We explicitly exclude the compromised LiteLLM versions (`1.82.7`, `1.82.8`) in our dependency specification.
- We pin all CI/CD GitHub Actions to commit SHAs, not mutable tags.
- We run `pip-audit` in CI to catch known vulnerabilities in our dependency tree.
- Dependabot monitors our dependencies weekly.
