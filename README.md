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
- [Anthropic API passthrough](#anthropic-api-passthrough)
  - [Usage](#usage)
  - [Proxied endpoints](#proxied-endpoints)
  - [Supported backends](#supported-backends)
- [Prompt caching](#prompt-caching)
  - [Automatic caching](#automatic-caching)
  - [Explicit cache breakpoints](#explicit-cache-breakpoints)
  - [TTL support](#ttl-support)
- [Web search](#web-search)
- [Development Environment](#development-environment)
  - [Setup](#setup)
  - [Lint](#lint)
  - [Test](#test)
  - [Clean](#clean)
  - [Build](#build)
  - [Publish](#publish)
  - [Git hooks](#git-hooks)

---

## Overview

The framework provides adapter from [AI DIAL Chat Completion API](https://dialx.ai/dial_api#operation/sendChatCompletionRequest) to [Anthropic Messages API](https://platform.claude.com/docs/en/api/messages).

---

## Anthropic API passthrough

In addition to the DIAL-to-Anthropic adapter, the library exposes a transparent **passthrough** for the native Anthropic Messages API.

The exposed Anthropic Messages API is compatible with the vanilla Anthropic Client from Anthropic SDK:

```py
from anthropic import Anthropic, AsyncAnthropic
client = Anthropic(api_key="...", base_url="${ADAPTER_ORIGIN}/anthropic")
```

The upstream errors are relayed to the caller in the native [Anthropic error schema](https://platform.claude.com/docs/en/api/errors).

### Usage

Mount the passthrough onto any Starlette/FastAPI host application (e.g. a `DIALApp`) with `mount_anthropic_api`. The upstream client is chosen per request by a factory you supply:

```python
from aidial_sdk import DIALApp
from anthropic import AsyncAnthropic
from aidial_adapter_anthropic.passthrough import mount_anthropic_api

app = DIALApp(...)

async def get_client(request):
    return AsyncAnthropic(api_key=...)

mount_anthropic_api(app, get_client)
```

The passthrough is mounted at `/anthropic` by default; pass `path=...` to change it. The `get_client` argument may also be a plain client instance instead of a factory.

### Proxied endpoints

The following Anthropic endpoints are forwarded (relative to the mount path):

- `POST /v1/messages` — create a message (streaming and non-streaming)
- `POST /v1/messages/batches` — create a message batch
- `POST /v1/messages/count_tokens` — count tokens

### Supported backends

The client factory may return any of the Anthropic SDK's async clients: `AsyncAnthropic`, `AsyncAnthropicBedrock`, `AsyncAnthropicBedrockMantle`, `AsyncAnthropicVertex`, and `AsyncAnthropicFoundry`.

The Bedrock backends require `botocore`, which is an optional dependency:

```sh
pip install aidial-adapter-anthropic[bedrock]
```

Endpoints a backend does not implement (e.g. Bedrock has no token-counting or batches route) surface as a `404` error.

---

## Prompt caching

### Automatic caching

Automatic caching is the simplest way to use prompt caching. A single top-level cache breakpoint instructs Anthropic to automatically apply a cache point to the last cacheable block of the request. This is ideal for multi-turn conversations where the growing message history should be cached automatically. See [Automatic caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching#automatic-caching) in the Anthropic docs.

To enable automatic caching, set `custom_fields.cache_breakpoint` at the top level of the Chat Completion request:

<details><summary>Top-level cache breakpoint</summary>

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "custom_fields": {
    "cache_breakpoint": {}
  }
}
```

</details>

### Explicit cache breakpoints

Explicit cache breakpoints give fine-grained control over which parts of the prompt get cached. You can place a cache breakpoint on individual system messages, user/assistant messages, or tool definitions. See [Explicit cache breakpoints](https://platform.claude.com/docs/en/build-with-claude/prompt-caching#explicit-cache-breakpoints) in the Anthropic docs.

To add a breakpoint, set `custom_fields.cache_breakpoint` on a message or tool object:

<details><summary>System cache breakpoint</summary>

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant with extensive knowledge.",
      "custom_fields": {
        "cache_breakpoint": {}
      }
    },
    {"role": "user", "content": "Hello!"}
  ]
}
```

</details>

<details><summary>Message cache breakpoint</summary>

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {
      "role": "user",
      "content": "Here is a long document: ...",
      "custom_fields": {
        "cache_breakpoint": {}
      }
    },
    {"role": "user", "content": "Summarize it."}
  ]
}
```

</details>

<details><summary>Tools cache breakpoint</summary>

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [
    {"role": "user", "content": "What's the weather?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          },
          "required": ["location"]
        }
      },
      "custom_fields": {
        "cache_breakpoint": {}
      }
    }
  ]
}
```

</details>

### TTL support

A cache breakpoint may include an optional `ttl` field. Supported values are `5m` (5 minutes, default) and `1h` (one hour). The `ttl` field is supported on both top-level and explicit breakpoints. See [TTL support](https://platform.claude.com/docs/en/build-with-claude/prompt-caching#ttl-support) in the Anthropic docs.

<details><summary>Top-level cache breakpoint with TTL</summary>

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "custom_fields": {
    "cache_breakpoint": {
      "ttl": "1h"
    }
  }
}
```

</details>

---

## Web search

Web search gives Claude direct access to real-time web content, allowing it to answer questions with up-to-date information beyond its knowledge cutoff. It is an Anthropic server-side tool: the searches are executed on Anthropic's side, and the final response includes citations for the sources used. See [Web search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool) in the Anthropic docs.

To enable web search, add a static tool named `web_search` to the request's `tools` list. The Anthropic web search tool definition goes into `static_function.configuration`; the `name` is defaulted from the static function, so you don't have to repeat it. Being a server-side tool, web search never forces a `tool_choice` and can be combined with ordinary function tools.

<details><summary>Enable web search</summary>

```json
{
  "model": "claude-opus-4-8",
  "messages": [
    {"role": "user", "content": "What is the weather in NYC?"}
  ],
  "tools": [
    {
      "type": "static_function",
      "static_function": {
        "name": "web_search",
        "configuration": {
          "type": "web_search_20250305"
        }
      }
    }
  ]
}
```

</details>

The tool definition supports optional fields such as `max_uses`, `allowed_domains`, `blocked_domains`, and `user_location`. See [Tool definition](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool#tool-definition) in the Anthropic docs.

<details><summary>Web search with optional fields</summary>

```json
{
  "model": "claude-opus-4-8",
  "messages": [
    {"role": "user", "content": "What is the weather in San Francisco?"}
  ],
  "tools": [
    {
      "type": "static_function",
      "static_function": {
        "name": "web_search",
        "configuration": {
          "type": "web_search_20250305",
          "max_uses": 5,
          "allowed_domains": ["example.com", "trusteddomain.org"],
          "user_location": {
            "type": "approximate",
            "city": "San Francisco",
            "region": "California",
            "country": "US",
            "timezone": "America/Los_Angeles"
          }
        }
      }
    }
  ]
}
```

</details>

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

### Git hooks

You may optionally install Git hooks that will automatically run the linting step on Git push. You only need to do it once for the given repository.

```sh
make install_git_hooks
```

> [!IMPORTANT]
> This command doesn't work if you have already installed Git hooks locally or globally.
