<h1 align="center">
        Python SDK for adapter from DIAL API to Anthropic API
    </h1>
    <p align="center">
        <p align="center">
        <a href="https://dialx.ai/">
          <img src="https://dialx.ai/logo/dialx_logo.svg" alt="About DIALX">
        </a>
    </p>
<h4 align="center">
    <a href="https://pypi.org/project/aidial-adapter-anthropic/">
        <img src="https://img.shields.io/pypi/v/aidial-adapter-anthropic.svg" alt="PyPI version">
    </a>
    <a href="https://discord.gg/ukzj9U9tEe">
        <img src="https://img.shields.io/static/v1?label=DIALX%20Community%20on&message=Discord&color=blue&logo=Discord&style=flat-square" alt="Discord">
    </a>
</h4>

- [Overview](#overview)
- [Development Environment](#development-environment)
  - [Setup](#setup)
  - [Lint](#lint)
  - [Test](#test)
  - [Clean](#clean)
  - [Build](#build)
  - [Publish](#publish)

---

## Overview

The framework provides adapter from [AI DIAL Chat Completion API](https://dialx.ai/dial_api#operation/sendChatCompletionRequest) to [Anthropic Messages API](https://platform.claude.com/docs/en/api/messages).

---

## Development Environment

This project requires [Python ≥3.11](https://www.python.org/downloads/) and [Poetry ≥2.1.1](https://python-poetry.org/) for dependency management.

### Setup

1. Install Poetry. See the official [installation guide](https://python-poetry.org/docs/#installation).

2. *(Optional)* Specify custom Python or Poetry executables in `.env.dev`. This is useful if multiple versions are installed. By default, `python` and `poetry` are used.

   ```sh
   POETRY_PYTHON=path-to-python-exe
   POETRY=path-to-poetry-exe
   ```

3. Create and activate the virtual environment:

   ```sh
   make init_env
   source .venv/bin/activate
   ```

4. Install project dependencies (including linting, formatting, and test tools):

   ```sh
   make install
   ```

### Lint

Run the linting before committing:

```sh
make lint
```

To auto-fix formatting issues run:

```sh
make format
```

### Test

Run unit tests locally for available python versions:

```sh
make test
```

Run unit tests for the specific python version:

```sh
make test PYTHON=3.13
```

### Clean

To remove the virtual environment and build artifacts run:

```sh
make clean
```

### Build

To build the package run:

```sh
make build
```

### Publish

To publish the package to PyPI run:

```sh
make publish
```
