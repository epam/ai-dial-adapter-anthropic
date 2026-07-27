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
- [Chat Completions API](#chat-completions-api)
  - [Basic request](#basic-request)
    - [Structured outputs](#structured-outputs)
    - [Maximum completion tokens](#maximum-completion-tokens)
    - [Function calling](#function-calling)
    - [Multi-modal inputs](#multi-modal-inputs)
    - [Reasoning effort](#reasoning-effort)
  - [DIAL extensions](#dial-extensions)
    - [Attachments](#attachments)
    - [Configuration](#configuration)
      - [Extended thinking](#extended-thinking)
      - [Reasoning level](#reasoning-level)
    - [Web search](#web-search)
    - [Prompt truncation](#prompt-truncation)
    - [Prompt caching](#prompt-caching)
      - [Automatic caching](#automatic-caching)
      - [Explicit cache breakpoints](#explicit-cache-breakpoints)
      - [TTL support](#ttl-support)
      - [DIAL Core configuration](#dial-core-configuration)
- [Anthropic API](#anthropic-api)
  - [Usage](#usage)
  - [Proxied endpoints](#proxied-endpoints)
  - [Supported backends](#supported-backends)
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

The package exposes Claude models via two APIs:

|API|Description|
|---|---|
|[Chat Completions API](#chat-completions-api)|The [AI DIAL Chat Completion API](https://dialx.ai/dial_api#operation/sendChatCompletionRequest) adapted to the Anthropic Messages API|
|[Anthropic API](#anthropic-api)|The native [Anthropic Messages API](https://platform.claude.com/docs/en/api/messages) served in the passthrough mode|

---

## Chat Completions API

The package provides an adapter from the Chat Completions API *(ingress)* to the [Anthropic Messages API](https://platform.claude.com/docs/en/api/messages) *(upstream)*.

### Basic request

The standard Chat Completions request fields are supported as follows:

|Field|Support|
|---|---|
|`messages`|Supported, including the system, developer, tool and function messages. See [Multi-modal inputs](#multi-modal-inputs) for the non-text content|
|`stream`, `stop`, `top_p`|Relayed to the Anthropic API as-is|
|`temperature`|Mapped from the OpenAI `[0, 2]` range to the Anthropic `[0, 1]` range|
|`n`|Supported via parallel requests to the upstream|
|`max_tokens`|See [Maximum completion tokens](#maximum-completion-tokens)|
|`response_format`|See [Structured outputs](#structured-outputs)|
|`tools`, `tool_choice`|See [Function calling](#function-calling)|
|`reasoning_effort`|See [Reasoning effort](#reasoning-effort)|
|`seed`|Unsupported by Claude, ignored|

The token usage is reported in the `usage` object, including `completion_tokens_details.reasoning_tokens` and the cache counters *(see [Prompt caching](#prompt-caching))*.

#### Structured outputs

`response_format` of type `json_schema` is supported via the Anthropic [structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs). Claude accepts no other value of `additionalProperties` but `false`, so the adapter sets it throughout the schema.

The `json_object` type is unsupported and ignored.

#### Maximum completion tokens

Unlike OpenAI models, Claude models require the `max_tokens` parameter. When the request omits it, the adapter falls back to the default configured by the host application.

We recommend configuring the default on a per-model basis in the DIAL Core config instead, since all the token-related information *(like pricing and token limits)* is then kept in the same place. The DIAL Core default takes precedence over the adapter one.

```json
{
  "models": {
    "${DIAL_DEPLOYMENT_ID}": {
      "type": "chat",
      "endpoint": "...",
      "defaults": {
        "max_tokens": 2048
      }
    }
  }
}
```

Make sure the default doesn't exceed the [max output tokens](https://platform.claude.com/docs/en/about-claude/models/overview) of the model, otherwise the request fails with an error like `max_tokens: 10000 > 8192, which is the maximum allowed number of output tokens for claude-...`.

#### Function calling

The `tools` and `tool_choice` fields are supported, `tool_choice` including the `auto`, `none`, `required` and named-function modes.

The legacy Functions API *(`functions` and `function_call`)* is supported as well and converted to tools transparently. Claude may generate more than one call per response, while the Functions API allows a single one; the extra calls are discarded in this mode.

#### Multi-modal inputs

|Content part|Support|
|---|---|
|`text`|Supported|
|`image_url`|The URL is either a data URL, a public URL or a DIAL file URL|
|`file`|The `file.file_data` field is either a data URL or a base64-encoded PDF. The `file_id` field is unsupported|
|`input_audio`, `refusal`|Unsupported|

Files of any supported type may also be passed as [DIAL attachments](#attachments), which is the only way to reference a file by URL.

#### Reasoning effort

The `reasoning_effort` field sets the [effort level](https://platform.claude.com/docs/en/build-with-claude/effort) of the response. It only accepts the OpenAI values, so the `xhigh` and `max` levels are reachable via the [configuration](#reasoning-level) alone.

### DIAL extensions

The features below are the DIAL extensions of the Chat Completions API.

#### Attachments

The attachments are passed in the `custom_content.attachments` field of a message, either inline *(`data`)* or by reference *(`url`)*. The supported types are:

|Type|MIME types|
|---|---|
|Images|`image/png`, `image/jpeg`, `image/gif`, `image/webp`|
|PDF documents|`application/pdf`|
|Text documents|`text/plain`, `text/html`, `text/css`, `text/javascript`, `text/x-typescript`, `text/csv`, `text/markdown`, `text/x-python`, `text/xml`, `text/rtf`, `application/json`|

The documents are supported only by the models with [PDF support](https://platform.claude.com/docs/en/build-with-claude/pdf-support); the host application declares whether the model at hand is one of them.

Setting `enable_citations` in the [configuration](#configuration) makes Claude cite the documents it used; the citations are returned as numbered DIAL attachments.

#### Configuration

The adapter accepts a per-request configuration object in the `custom_fields.configuration` field. All its fields are optional; the host application serves its JSON Schema via the DIAL `/configuration` endpoint.

|Field|Description|
|---|---|
|`thinking`|[Extended thinking](#extended-thinking) configuration|
|`effort`|[Reasoning level](#reasoning-level) of the response|
|`betas`|List of [beta feature flags](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/anthropic_beta_param.py) to enable, e.g. `["token-efficient-tools-2025-02-19"]`|
|`enable_citations`|Enables [citations](https://platform.claude.com/docs/en/build-with-claude/citations) for the document [attachments](#attachments). Defaults to `false`|

Not every Claude deployment supports every field or beta flag; consult the official documentation before use.

<details><summary>Configuration example</summary>

```json
{
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "custom_fields": {
    "configuration": {
      "thinking": {"type": "adaptive"},
      "effort": "high",
      "betas": ["token-efficient-tools-2025-02-19"],
      "enable_citations": true
    }
  }
}
```

</details>

##### Extended thinking

The `thinking` object is relayed to the Anthropic API as-is, so any [thinking configuration](https://platform.claude.com/docs/en/build-with-claude/extended-thinking) is supported:

|Configuration|Comment|
|---|---|
|`{"type": "adaptive"}`|The model decides when to think|
|`{"type": "enabled", "budget_tokens": 1024}`|Thinking with the given limit on reasoning tokens|
|`{"type": "disabled"}`|Thinking disabled|

The thinking blocks are reported in a dedicated `Thinking` stage and preserved across conversation turns, so multi-turn tool use works with thinking enabled.

`temperature` is ignored when thinking is enabled; `top_p` is ignored as well when thinking is adaptive, since Claude rejects both parameters in these modes.

##### Reasoning level

The `effort` field extends the [reasoning effort](#reasoning-effort) with the Claude-specific levels: `low`, `medium`, `high`, `xhigh` and `max`. The value is relayed to the Anthropic API as-is, so the levels added later work without an adapter update.

Setting both `effort` and `reasoning_effort` to different values is a validation error.

#### Web search

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

#### Prompt truncation

When `max_prompt_tokens` is set, the adapter discards the oldest messages until the prompt fits the limit, keeping the system prompt and the last message. The indices of the discarded messages are reported in the `discarded_messages` field of the response.

The token counting is delegated to the Anthropic [count tokens](https://platform.claude.com/docs/en/api/messages-count-tokens) endpoint. For the backends that don't implement it (e.g. Bedrock), the host application may supply the bundled approximate tokenizer instead, which deliberately **overestimates** the token count, so that the truncated prompt never overflows the limit.

#### Prompt caching

##### Automatic caching

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

##### Explicit cache breakpoints

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

##### TTL support

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

##### DIAL Core configuration

When a DIAL deployment has multiple upstreams, caching only pays off if the requests sharing a prefix reach the same upstream. Enable the corresponding feature flag in the DIAL Core config to make DIAL Core route them consistently: `cacheSupported` for explicit breakpoints, `autoCachingSupported` for automatic caching.

A top-level cache breakpoint may also be preset for all the requests to the deployment via `defaults`:

```json
{
  "models": {
    "${DIAL_DEPLOYMENT_ID}": {
      "type": "chat",
      "endpoint": "...",
      "defaults": {
        "custom_fields": {
          "cache_breakpoint": {}
        }
      },
      "features": {
        "autoCachingSupported": true
      },
      "upstreams": ["..."]
    }
  }
}
```

The cache usage is reported in the `usage.prompt_tokens_details` object: `cached_tokens` for the cache hits and `cache_write_tokens` for the tokens written to the cache.

---

## Anthropic API

The package supports the native [Anthropic Messages API](https://platform.claude.com/docs/en/api/messages) in the **passthrough** mode: the requests are forwarded to the upstream as-is and the upstream errors are relayed to the caller in the native [Anthropic error schema](https://platform.claude.com/docs/en/api/errors).

The exposed API is compatible with the vanilla client from the Anthropic SDK:

```py
from anthropic import Anthropic, AsyncAnthropic
client = Anthropic(api_key="...", base_url="${ADAPTER_ORIGIN}/anthropic")
```

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
