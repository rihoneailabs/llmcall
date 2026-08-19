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

# Install pinned versions of poetry, tox, pip-audit
RUN pip install --no-cache-dir \
    poetry==2.4.1 \
    tox==4.53.1 \
    pip-audit==2.10.1

ENV POETRY_NO_INTERACTION=1 \
    PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring

WORKDIR /workspace
