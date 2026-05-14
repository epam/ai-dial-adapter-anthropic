import anthropic
import pytest
from aidial_sdk.chat_completion import Response
from aidial_sdk.chat_completion.request import ChatCompletionRequest, Request
from pydantic import SecretStr
from starlette.requests import Request as StarletteRequest

from aidial_adapter_anthropic.adapter import ChatCompletionAdapter
from aidial_adapter_anthropic.adapter._claude import caching as caching_module
from aidial_adapter_anthropic.adapter._claude.adapter import Adapter
from aidial_adapter_anthropic.adapter._claude.tokenizer import (
    ApproximateTokenizer,
)
from aidial_adapter_anthropic.adapter.claude import create_adapter
from aidial_adapter_anthropic.dial.consumer import ChoiceConsumer
from aidial_adapter_anthropic.dial.request import ModelParameters

_DIAL_CACHE_BREAKPOINT_PATH = "X-DIAL-CACHE-BREAKPOINT-PATH"
_DIAL_CACHE_EXPIRE_AT = "X-DIAL-CACHE-EXPIRE-AT"


def _user(content: str, *, cache_breakpoint: dict | None = None) -> dict:
    msg: dict = {"role": "user", "content": content}
    if cache_breakpoint is not None:
        msg["custom_fields"] = {"cache_breakpoint": cache_breakpoint}
    return msg


def _sys(content: str, *, cache_breakpoint: dict | None = None) -> dict:
    msg: dict = {"role": "system", "content": content}
    if cache_breakpoint is not None:
        msg["custom_fields"] = {"cache_breakpoint": cache_breakpoint}
    return msg


def _tool(cache_breakpoint: dict | None = None) -> dict:
    tool: dict = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        },
    }
    if cache_breakpoint is not None:
        tool["custom_fields"] = {"cache_breakpoint": cache_breakpoint}
    return tool


def _request(request: dict) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(request)


def _response(request: ChatCompletionRequest) -> Response:
    return Response(
        Request(
            headers={},
            api_key_secret=SecretStr("test"),
            deployment_id="test-deployment",
            original_request=StarletteRequest(
                {"type": "http", "method": "POST", "path": "/", "headers": []}
            ),
            **request.model_dump(exclude_none=True),
        )
    )


@pytest.fixture
async def adapter() -> ChatCompletionAdapter:
    return await create_adapter(
        deployment="test-deployment",
        storage=None,
        client=anthropic.AsyncAnthropic(),
        custom_tokenizer=ApproximateTokenizer(),
        default_max_tokens=1024,
        supports_thinking=True,
        supports_documents=True,
    )


@pytest.fixture(autouse=True)
def mock_current_time_1000s(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(caching_module.time, "time", lambda: 1000)


@pytest.fixture(autouse=True)
def mock_adapter_chat(monkeypatch: pytest.MonkeyPatch):
    async def _fake_chat(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(Adapter, "chat", _fake_chat)


async def _invoke_chat(
    adapter: ChatCompletionAdapter,
    request: dict,
) -> ChoiceConsumer:
    req = _request(request)
    consumer = ChoiceConsumer(_response(req))
    await adapter.chat(consumer, ModelParameters.create(req), req.messages)
    return consumer


async def test_adapter_chat_sets_headers_for_last_message_breakpoint(
    adapter: ChatCompletionAdapter,
):
    consumer = await _invoke_chat(
        adapter,
        {
            "messages": [
                _user("first", cache_breakpoint={"ttl": "1h"}),
                _user("second"),
                _user("third", cache_breakpoint={"ttl": "5m"}),
            ]
        },
    )

    assert consumer.response.headers == [
        (_DIAL_CACHE_BREAKPOINT_PATH, "prefix.body.messages[2]"),
        (_DIAL_CACHE_EXPIRE_AT, "4600"),
    ]


async def test_adapter_chat_does_not_set_headers_without_breakpoints(
    adapter: ChatCompletionAdapter,
):
    consumer = await _invoke_chat(
        adapter,
        {"messages": [_user("first"), _user("second")]},
    )

    assert consumer.response.headers == []


async def test_adapter_chat_sets_headers_for_system_message_breakpoint(
    adapter: ChatCompletionAdapter,
):
    consumer = await _invoke_chat(
        adapter,
        {
            "messages": [
                _sys("be helpful", cache_breakpoint={"ttl": "5m"}),
                _user("hello"),
            ]
        },
    )

    assert consumer.response.headers == [
        (_DIAL_CACHE_BREAKPOINT_PATH, "prefix.body.messages[0]"),
        (_DIAL_CACHE_EXPIRE_AT, "1300"),
    ]


async def test_adapter_chat_sets_headers_for_tool_breakpoint(
    adapter: ChatCompletionAdapter,
):
    consumer = await _invoke_chat(
        adapter,
        {
            "messages": [_user("What's the weather?")],
            "tools": [_tool(cache_breakpoint={})],
        },
    )

    assert consumer.response.headers == [
        (_DIAL_CACHE_BREAKPOINT_PATH, "prefix.body.tools[0]"),
        (_DIAL_CACHE_EXPIRE_AT, "1300"),
    ]


async def test_adapter_chat_uses_default_ttl_for_invalid_breakpoint_ttl(
    adapter: ChatCompletionAdapter,
):
    consumer = await _invoke_chat(
        adapter,
        {
            "messages": [
                _user("first"),
                _user("second", cache_breakpoint={"ttl": "invalid"}),
            ]
        },
    )

    assert consumer.response.headers == [
        (_DIAL_CACHE_BREAKPOINT_PATH, "prefix.body.messages[1]"),
        (_DIAL_CACHE_EXPIRE_AT, "1300"),
    ]


async def test_adapter_chat_prefers_message_path_over_tool_breakpoint(
    adapter: ChatCompletionAdapter,
):
    consumer = await _invoke_chat(
        adapter,
        {
            "tools": [_tool(cache_breakpoint={"ttl": "1h"})],
            "messages": [_user("first", cache_breakpoint={"ttl": "5m"})],
        },
    )

    assert consumer.response.headers == [
        (_DIAL_CACHE_BREAKPOINT_PATH, "prefix.body.messages[0]"),
        (_DIAL_CACHE_EXPIRE_AT, "4600"),
    ]


async def test_adapter_chat_sets_headers_for_last_tool_breakpoint(
    adapter: ChatCompletionAdapter,
):
    consumer = await _invoke_chat(
        adapter,
        {
            "messages": [_user("What's the weather?")],
            "tools": [
                _tool(),
                _tool(cache_breakpoint={"ttl": "5m"}),
            ],
        },
    )

    assert consumer.response.headers == [
        (_DIAL_CACHE_BREAKPOINT_PATH, "prefix.body.tools[1]"),
        (_DIAL_CACHE_EXPIRE_AT, "1300"),
    ]
