FROM python:3.11-slim

# Install system dependencies & git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    libffi-dev \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install poetry (v2+ matching lockfile format), tox, pip-audit
RUN pip install --no-cache-dir "poetry>=2.0.0" tox pip-audit

ENV POETRY_NO_INTERACTION=1 \
    PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring

WORKDIR /workspace
