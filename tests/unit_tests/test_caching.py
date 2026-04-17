import pytest
from aidial_sdk.chat_completion import CacheBreakpoint
from aidial_sdk.chat_completion.request import (
    ChatCompletionRequest,
)
from anthropic import Omit
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam as CacheControlEphemeralParam,
)

from aidial_adapter_anthropic.adapter._claude.adapter import (
    Adapter,
    ClaudeRequest,
)
from aidial_adapter_anthropic.adapter._claude.converters import (
    to_claude_cache_control,
)
from aidial_adapter_anthropic.dial.request import ModelParameters

_EPHEMERAL = CacheControlEphemeralParam(type="ephemeral")


async def _to_clade_request(adapter: Adapter, request: dict) -> ClaudeRequest:
    req = ChatCompletionRequest.model_validate(request)
    params = ModelParameters.create(req)
    return await adapter._prepare_claude_request(params, req.messages)


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


def _create_request_with_tool(add_breakpoint: bool) -> dict:
    tool = {"type": "function", "function": {"name": "get_weather"}}
    if add_breakpoint:
        tool["custom_fields"] = {"cache_breakpoint": {}}
    return {"messages": [_user("hi")], "tools": [tool]}


def test_to_claude_cache_control_returns_ephemeral():
    result = to_claude_cache_control(CacheBreakpoint())
    assert result == _EPHEMERAL


def test_to_claude_cache_control_ignores_expire_at():
    result = to_claude_cache_control(CacheBreakpoint(expire_at="2099-01-01"))
    assert result == _EPHEMERAL


def test_to_claude_cache_control_proxy_extra_fields():
    result = to_claude_cache_control(CacheBreakpoint.model_construct(foo="bar"))
    assert result == {"type": "ephemeral", "foo": "bar"}


async def test_top_level_cache_breakpoint(adapter: Adapter):
    request = await _to_clade_request(
        adapter,
        {"messages": [_user("hi")], "custom_fields": {"cache_breakpoint": {}}},
    )
    assert request.params["cache_control"] == _EPHEMERAL


async def test_no_top_level_cache_breakpoint(adapter: Adapter):
    request = await _to_clade_request(adapter, {"messages": [_user("hi")]})
    assert isinstance(request.params["cache_control"], Omit)


async def test_user_message_cache_control(adapter: Adapter):
    request = await _to_clade_request(
        adapter, {"messages": [_user("hello", cache_breakpoint={})]}
    )
    content = request.claude_messages[0]["content"]
    assert isinstance(content, list)
    assert len(content) == 1
    last = content[0]
    assert isinstance(last, dict)
    assert last.get("cache_control") == _EPHEMERAL


async def test_user_message_no_cache_control(adapter: Adapter):
    request = await _to_clade_request(adapter, {"messages": [_user("hello")]})
    content = request.claude_messages[0]["content"]
    assert isinstance(content, list)
    assert len(content) == 1
    assert isinstance(content[0], dict)
    assert "cache_control" not in content[0]


async def test_system_message_cache_control(adapter: Adapter):
    request = await _to_clade_request(
        adapter,
        {"messages": [_sys("be helpful", cache_breakpoint={}), _user("hi")]},
    )
    system = request.params["system"]
    assert isinstance(system, list)
    assert len(system) == 1
    assert system[0].get("cache_control") == _EPHEMERAL


async def test_tool_with_cache_control(adapter: Adapter):
    request = await _to_clade_request(adapter, _create_request_with_tool(True))
    tools = request.params["tools"]

    assert isinstance(tools, list)
    assert tools[0].get("cache_control") == _EPHEMERAL


async def test_tool_with_no_cache_control(adapter: Adapter):
    request = await _to_clade_request(adapter, _create_request_with_tool(False))

    tools = request.params["tools"]
    assert isinstance(tools, list)
    assert "cache_control" not in tools[0]


@pytest.mark.parametrize("ttl", [None, "5m", "1h", "foobar"])
async def test_cache_control_ttl(adapter: Adapter, ttl: str | None):
    request = await _to_clade_request(
        adapter,
        {
            "messages": [_user("hi")],
            "custom_fields": {"cache_breakpoint": {"ttl": ttl}},
        },
    )

    assert request.params["cache_control"] == {"type": "ephemeral", "ttl": ttl}
