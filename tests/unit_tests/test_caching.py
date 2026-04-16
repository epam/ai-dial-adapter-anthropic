from typing import Any, TypeGuard

from aidial_sdk.chat_completion import CacheBreakpoint
from aidial_sdk.chat_completion.request import AzureChatCompletionRequest
from anthropic import Omit
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam as CacheControlEphemeralParam,
)
from anthropic.types.beta import BetaTextBlockParam as TextBlockParam

from aidial_adapter_anthropic.adapter._claude.adapter import Adapter
from aidial_adapter_anthropic.adapter._claude.converters import (
    to_claude_cache_control,
    to_claude_tool_config,
)
from aidial_adapter_anthropic.dial.request import ModelParameters
from aidial_adapter_anthropic.dial.tools import ToolsConfig
from tests.utils.openai import sys, user

_EPHEMERAL = CacheControlEphemeralParam(type="ephemeral")


# --- Type-narrowing helpers ---


def _is_list(x: object) -> TypeGuard[list[Any]]:
    return isinstance(x, list)


def _is_dict(x: object) -> TypeGuard[dict[str, Any]]:
    return isinstance(x, dict)


def _is_text_block_param(x: object) -> TypeGuard[TextBlockParam]:
    return isinstance(x, dict) and x.get("type") == "text"


def test_to_claude_cache_control_returns_ephemeral():
    result = to_claude_cache_control(CacheBreakpoint())
    assert result == _EPHEMERAL


def test_to_claude_cache_control_ignores_expire_at():
    result = to_claude_cache_control(CacheBreakpoint(expire_at="2099-01-01"))
    assert result == _EPHEMERAL


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
    assert _is_list(content)
    assert len(content) == 1
    last = content[0]
    assert _is_text_block_param(last)
    assert last.get("cache_control") == _EPHEMERAL


async def test_user_message_no_cache_control(adapter: Adapter):
    request = await adapter._prepare_claude_request(
        ModelParameters(), [user("hello")]
    )
    content = request.claude_messages[0]["content"]
    assert _is_list(content)
    for block in content:
        assert _is_dict(block)
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


def _create_request_with_tools(
    cache_breakpoint: bool,
) -> AzureChatCompletionRequest:
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    if cache_breakpoint:
        tool["custom_fields"] = {"cache_breakpoint": {}}
    return AzureChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [tool],
        }
    )


def test_tool_with_cache_control():
    req = _create_request_with_tools(True)
    tools_config = ToolsConfig.from_request(req)
    claude_tools = to_claude_tool_config(tools_config)
    assert claude_tools is not None
    assert claude_tools.tools[0].get("cache_control") == _EPHEMERAL


def test_tool_with_no_cache_control():
    req = _create_request_with_tools(False)
    tools_config = ToolsConfig.from_request(req)
    claude_tools = to_claude_tool_config(tools_config)
    assert claude_tools is not None
    assert "cache_control" not in claude_tools.tools[0]
