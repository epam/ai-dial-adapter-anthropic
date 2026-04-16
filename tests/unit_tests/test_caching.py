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
from tests.utils.openai import sys, user

_EPHEMERAL = CacheControlEphemeralParam(type="ephemeral")


def test_to_claude_cache_control_returns_ephemeral():
    result = to_claude_cache_control(CacheBreakpoint())
    assert result == _EPHEMERAL


def test_to_claude_cache_control_ignores_expire_at():
    result = to_claude_cache_control(CacheBreakpoint(expire_at="2099-01-01"))
    assert result == _EPHEMERAL


async def _to_clade_request(adapter: Adapter, request: dict) -> ClaudeRequest:
    req = ChatCompletionRequest.model_validate(request)
    params = ModelParameters.create(req)
    return await adapter._prepare_claude_request(params, req.messages)


async def test_top_level_cache_breakpoint(adapter: Adapter):
    params = ModelParameters(cache_breakpoint=CacheBreakpoint())
    request = await adapter._prepare_claude_request(params, [user("hi")])
    assert request.params["cache_control"] == _EPHEMERAL


async def test_no_top_level_cache_breakpoint(adapter: Adapter):
    params = ModelParameters()
    request = await adapter._prepare_claude_request(params, [user("hi")])
    assert isinstance(request.params["cache_control"], Omit)


async def test_user_message_cache_control(adapter: Adapter):
    msg = user(content="hello", cache_breakpoint=CacheBreakpoint())
    request = await adapter._prepare_claude_request(ModelParameters(), [msg])
    content = request.claude_messages[0]["content"]
    assert isinstance(content, list)
    assert len(content) == 1
    last = content[0]
    assert isinstance(last, dict)
    assert last.get("cache_control") == _EPHEMERAL


async def test_user_message_no_cache_control(adapter: Adapter):
    request = await adapter._prepare_claude_request(
        ModelParameters(), [user("hello")]
    )
    content = request.claude_messages[0]["content"]
    assert isinstance(content, list)
    for block in content:
        assert isinstance(block, dict)
        assert "cache_control" not in block


async def test_system_message_cache_control(adapter: Adapter):
    sys_msg = sys("be helpful", cache_breakpoint=CacheBreakpoint())
    request = await adapter._prepare_claude_request(
        ModelParameters(), [sys_msg, user("hi")]
    )
    system = request.params["system"]
    assert isinstance(system, list)
    assert len(system) == 1
    assert system[0].get("cache_control") == _EPHEMERAL


def _create_request_with_tools(add_breakpoint: bool) -> dict:
    tool = {"type": "function", "function": {"name": "get_weather"}}
    if add_breakpoint:
        tool["custom_fields"] = {"cache_breakpoint": {}}
    return {"messages": [{"role": "user", "content": "hi"}], "tools": [tool]}


async def test_tool_with_cache_control(adapter: Adapter):
    request = await _to_clade_request(adapter, _create_request_with_tools(True))
    tools = request.params["tools"]

    assert isinstance(tools, list)
    assert tools[0].get("cache_control") == _EPHEMERAL


async def test_tool_with_no_cache_control(adapter: Adapter):
    request = await _to_clade_request(
        adapter, _create_request_with_tools(False)
    )

    tools = request.params["tools"]
    assert isinstance(tools, list)
    assert "cache_control" not in tools[0]
