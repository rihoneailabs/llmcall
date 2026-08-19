---
name: dependabot-remediation
description: >-
  Automate discovering, triaging, and remediating GitHub Dependabot security alerts
  for Python/Poetry repositories inside an isolated Docker dev container sandbox,
  including vulnerability audits, test matrix validation, and PR generation.
---

# Dependabot & Security Alert Remediation Skill

This skill provides an end-to-end runbook for identifying, triaging, and resolving GitHub Dependabot security alerts safely inside an isolated Docker sandbox.

---

## When to Activate

Trigger this skill when:
- The user asks to check, list, or resolve Dependabot alerts or security vulnerabilities.
- A new vulnerability advisory / CVE is reported against project dependencies.
- You need to perform a clean dependency upgrade and verify zero remaining CVEs with `pip-audit`.

---

## Step-by-Step Remediation Workflow

### 1. Query & Triage Alerts
Fetch all open alerts directly from GitHub's Dependabot API:
```bash
gh api "repos/:owner/:repo/dependabot/alerts?state=open&per_page=100" \
  --jq '[.[] | {number: .number, package: .security_vulnerability.package.name, severity: .security_advisory.severity, summary: .security_advisory.summary, patched_version: .security_vulnerability.first_patched_version.identifier, vulnerable_range: .security_vulnerability.vulnerable_version_range}]'
```

Categorize each alert:
- **Direct Dependencies:** Listed under `[tool.poetry.dependencies]` in `pyproject.toml`.
- **Transitive Dependencies:** Pulled in by dev tools or other packages (e.g. `GitPython` via `python-semantic-release`).

---

### 2. Sandbox Setup & Environment Isolation
Always perform dependency resolution, lockfile generation, and test execution inside a Docker sandbox to prevent executing potentially vulnerable code or polluting the host environment:

1. Use `.devcontainer/Dockerfile` or build a sandbox image:
   ```bash
   docker build -t llmcall-sandbox -f .devcontainer/Dockerfile .devcontainer
   ```
2. **Key Environment Settings for Sandbox:**
   - Set `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` to prevent headless DBUS/SecretService crashes.
   - Mount an isolated volume for `/workspace/.venv` so host OS virtualenv binaries do not conflict with container Linux binaries:
     `-v $(pwd):/workspace -v /workspace/.venv -w /workspace`

---

### 3. Branching & Dependency Remediation
1. Checkout a dedicated fix branch:
   ```bash
   git checkout -b fix/dependabot-security-updates
   ```
2. For direct dependencies, update the constraint in `pyproject.toml` to the minimum patched version:
   ```toml
   aiohttp = "^3.14.3"
   ```
3. Update and regenerate the lockfile inside the sandbox:
   ```bash
   docker run --rm \
     -v $(pwd):/workspace \
     -v /workspace/.venv \
     -w /workspace \
     llmcall-sandbox \
     poetry update <direct-pkg> <transitive-pkg>
   ```

---

### 4. Quality & Security Verification
Execute the full test and audit suite inside the sandbox container:

```bash
docker run --rm \
  -v $(pwd):/workspace \
  -v /workspace/.venv \
  -w /workspace \
  llmcall-sandbox bash -c "
  poetry install && \
  pip-audit --path /workspace/.venv/lib/python*/site-packages && \
  poetry run ruff check . && \
  poetry run ruff format . --check && \
  poetry run pytest
"
```

Verify:
- **`pip-audit`**: Zero vulnerabilities for the targeted packages.
- **`ruff`**: Linting and formatting pass with 0 errors.
- **`pytest`**: All unit tests pass.

---

### 5. Commit, PR & Verification of Alert Closure
1. Stage and commit changes following conventional commit syntax:
   ```bash
   git add pyproject.toml poetry.lock .devcontainer .gitignore
   git commit -m "fix(deps): bump <packages> to patch security vulnerabilities"
   ```
2. Push branch and open Pull Request:
   ```bash
   git push -u origin fix/dependabot-security-updates
   gh pr create --title "fix(deps): bump <packages> to patch security vulnerabilities" \
     --body "Resolves open Dependabot security alerts for <packages>."
   ```
3. Monitor CI matrix run:
   ```bash
   gh run watch <run_id> --exit-status
   ```
4. Once merged into `main`, query the Dependabot API to verify alerts transitioned to closed:
   ```bash
   gh api "repos/:owner/:repo/dependabot/alerts?state=open"
   # Output should be []
   ```

---

## Known Gotchas & Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| **Poetry SIGILL / Exit 132 in Docker** | `keyring` trying to access non-existent D-Bus/SecretService in headless container | Set `ENV PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` in Dockerfile. |
| **Cross-platform `.venv` corruption** | Host `.venv` mounted directly into container | Use anonymous volume mount `-v /workspace/.venv` when running Docker. |
| **CI `pip-audit` fails on runner `setuptools`** | Default GitHub Actions runner image bundles outdated `setuptools` | Add `pip install --upgrade setuptools pip-audit` in CI workflow. |
| **Semantic Release `actions/checkout` auth failure** | Expired `RELEASE_PAT` secret in repo | Delete expired PAT so workflow falls back to built-in `GITHUB_TOKEN`. |
