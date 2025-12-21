# Imara – Project Setup & Development Guide

This document describes the basic setup and the most important development and workflows for the **Imara project**.

## Table of Contents

- [Getting Started](#getting-started)
- [Required Tools](#required-tools)
- [Setup Instructions](#setup-instructions)
  - [Clone repository](#clone-repository)
  - [Install dependencies](#install-dependencies)
  - [Install pre-commit hooks](#install-pre-commit-hooks)
  - [S3 authentication](#s3-authentication)
  - [Download documents](#download-documents)
  - [Setup Supabase local ENV](#setup-supabase-local-env)
- [Development Tools](#development-tools)
  - [Linter (ruff)](#linter-ruff)
  - [Formatter](#formatter)
  - [Type Checking (mypy)](#type-checking-mypy)
  - [Unit Tests](#unit-tests)
- [Using DVC](#using-dvc)
- [Package Management (uv)](#package-management-uv)
- [MLFlow](#mlflow)

---

## Getting Started

Quick steps to get the project running:

1. **Clone the repository**

   ```bash
   git clone https://github.com/FH-OST-Metis/imara.git
   cd imara
   ```

2. **Install dependencies**

   ```bash
   uv sync
   ```

3. **Set up S3 authentication** (see [S3 authentication](#s3-authentication) for details)

   ```bash
   cp .dvc/config.local-template .dvc/config.local
   cp env-template env
   # Edit both files with your credentials
   ```

4. **Download data**

   ```bash
   dvc pull
   ```

5. **Start developing!**

   ```bash
   uv run start_mlflow.py
   supabase start
   supabase functions serve
   # or supabase functions serve embed (only gemini)
   # or supabase functions serve embed_local (only ollama)
   ```

---

## Required Tools

Before you start, install the following tools:

- **uv** – fast Python package and environment management
  [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)
- **dvc** – Data Version Control for reproducible ML pipelines
  [https://dvc.org/doc/install](https://dvc.org/doc/install)
- **git** – version control
- **Python 3.10+**

Optional, but recommended:

- **pre-commit**
- **pytest**, **ruff**, **mypy** (installed via `uv sync`)

---

## Setup Instructions

### Clone repository

```bash
git clone https://github.com/FH-OST-Metis/imara.git
cd imara
```

### Install dependencies

```bash
uv sync
```

This creates a virtual environment and installs all packages defined in `pyproject.toml`.

---

### Install pre-commit hooks

```bash
uv run pre-commit install
```

Pre-commit will then run automatically on every commit.

---

### S3 authentication

To enable DVC data pulling and enable database access, you need to configure S3 credentials:

1. **Create a new Access Key**

   Visit the [Supabase S3 dashboard](https://supabase.com/dashboard/project/hjijyloqvddflojzrvcn/storage/s3) and generate a new access key.

2. **Configure DVC credentials**

   ```bash
   cp .dvc/config.local-template .dvc/config.local
   ```

   Edit `.dvc/config.local` and fill in:
   - `access_key_id` - from step 1
   - `secret_access_key` - from step 1

   **Note:** `.dvc/config.local` is gitignored and not versioned.

3. **Configure environment variables**

   ```bash
   cp env-template env
   ```

   Edit `env` and fill in:
   - `AWS_ACCESS_KEY_ID` - from step 1
   - `AWS_SECRET_ACCESS_KEY` - from step 1
   - `[DB_PASSWORD]` - replace with your database password

   **Note:** `env` is gitignored and not versioned.

### Download documents

```bash
dvc pull
```

---

### Setup Supabase local ENV

For instructions on setting up Supabase, please refer to the Supabase integration documentation.

[Supabase Setup Guide](supabase/README.md)

### Setup Ollama Embeddings

For local embedding generation with Ollama and GPU acceleration, refer to the Ollama setup guide.

[Ollama Setup Guide](ollama/README.md)

This includes:

- NVIDIA GPU setup for Linux and Windows/WSL2
- Apple Silicon (MPS) configuration for Macs
- Model installation and verification
- Integration with Supabase Edge Functions

## Development Tools

### Linter (ruff)

```bash
uv run ruff check .
```

### Formatter

```bash
uv run ruff format .
```

### Type Checking (mypy)

```bash
uv run mypy .
```

### Unit Tests

```bash
uv run pytest
```

---

## Using DVC

If you are using reproducible pipelines and versioned artifacts:

### Run pipeline

```bash
dvc repro
```

### Run single pipeline step

```bash
dvc repro chunk
```

### Download documents (e.g., from S3)

```bash
dvc pull
```

### Upload documents

```bash
dvc push
```

### Add documents to version control

```bash
dvc add data/raw/documents/sample.pdf
```

## Package Management (uv)

### Add packages

```bash
uv add <package-name>
```

### Update local environment

```bash
uv sync
```

### Update lockfile

```bash
uv lock
```

## MLFlow

Start the MLFlow tracking server to log and compare ML experiments:

```bash
uv run start_mlflow.py
```
