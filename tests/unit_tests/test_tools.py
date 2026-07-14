from typing import Any

import pytest
from aidial_sdk.chat_completion.request import AzureChatCompletionRequest
from anthropic import Omit
from openai.types.chat import ChatCompletionToolParam

from aidial_adapter_anthropic.adapter import ValidationError
from aidial_adapter_anthropic.adapter._claude.adapter import Adapter
from aidial_adapter_anthropic.adapter._claude.converters import (
    to_claude_tool_config,
)
from aidial_adapter_anthropic.dial.request import ModelParameters
from aidial_adapter_anthropic.dial.tools import ToolsConfig
from tests.utils.openai import (
    GET_WEATHER_TOOL,
    GET_WEATHER_TOOL_WITH_REFERENCES,
    user,
)

WEB_SEARCH_CONFIGURATION = {
    "type": "web_search_20250305",
    "max_uses": 5,
    "allowed_domains": ["example.com", "example.org"],
    "user_location": {
        "type": "approximate",
        "city": "San Francisco",
        "region": "California",
        "country": "US",
        "timezone": "America/Los_Angeles",
    },
}

# The `name` is defaulted from the static function name.
WEB_SEARCH_TOOL_REQUEST = {**WEB_SEARCH_CONFIGURATION, "name": "web_search"}


def web_search_static_tool(configuration: dict | None = None) -> dict:
    static_function: dict = {"name": "web_search"}
    if configuration is not None:
        static_function["configuration"] = configuration
    return {"type": "static_function", "static_function": static_function}


def _tool_config(tools: list[Any]) -> ToolsConfig | None:
    request = AzureChatCompletionRequest.model_validate(
        {"messages": [], "tools": tools}
    )
    return ToolsConfig.from_request(request)


def _run_schema_references_check(tool: ChatCompletionToolParam, has_refs: bool):
    request = AzureChatCompletionRequest.model_validate(
        {"messages": [], "tools": [tool]}
    )

    dial_tools = ToolsConfig.from_request(request)
    assert dial_tools is not None
    parameters = dial_tools.tools[0].function.parameters
    assert parameters is not None
    assert (parameters.get("$defs") is not None) == has_refs

    claude_tools = to_claude_tool_config(dial_tools)
    assert claude_tools is not None

    function_tool: Any = claude_tools.tools[0]
    assert ("$defs" in function_tool["input_schema"]) == has_refs


def test_tools_schemas_with_references():
    _run_schema_references_check(GET_WEATHER_TOOL_WITH_REFERENCES, True)


def test_tools_schemas_without_references():
    _run_schema_references_check(GET_WEATHER_TOOL, False)


@pytest.mark.parametrize(
    "tool_type",
    ["web_search_20250305", "web_search_20260209"],
)
async def test_web_search_minimal_passthrough(adapter: Adapter, tool_type: str):
    request = await adapter._prepare_claude_request(
        ModelParameters(
            tool_config=_tool_config(
                [web_search_static_tool({"type": tool_type})]
            )
        ),
        [user("What is the weather in NYC?")],
    )

    assert request.params["tools"] == [
        {"type": tool_type, "name": "web_search"}
    ]


async def test_web_search_all_optional_fields_preserved(adapter: Adapter):
    request = await adapter._prepare_claude_request(
        ModelParameters(
            tool_config=_tool_config(
                [web_search_static_tool(WEB_SEARCH_CONFIGURATION)]
            )
        ),
        [user("What is the weather in NYC?")],
    )

    tools = request.params["tools"]
    assert not isinstance(tools, Omit)
    assert tools == [WEB_SEARCH_TOOL_REQUEST]


async def test_web_search_tool_choice_left_default(adapter: Adapter):
    # Web search is a server tool: enabling it must not force a tool_choice.
    request = await adapter._prepare_claude_request(
        ModelParameters(
            tool_config=_tool_config(
                [web_search_static_tool(WEB_SEARCH_CONFIGURATION)]
            )
        ),
        [user("hello")],
    )

    assert isinstance(request.params["tool_choice"], Omit)


async def test_web_search_appended_after_function_tools(adapter: Adapter):
    request = await adapter._prepare_claude_request(
        ModelParameters(
            tool_config=_tool_config(
                [
                    GET_WEATHER_TOOL,
                    web_search_static_tool(WEB_SEARCH_CONFIGURATION),
                ]
            ),
        ),
        [user("What is the weather in NYC?")],
    )

    tools = request.params["tools"]
    assert not isinstance(tools, Omit)
    assert len(tools) == 2
    assert tools[0]["name"] == "get_temperature"
    assert tools[-1] == WEB_SEARCH_TOOL_REQUEST


async def test_no_web_search_keeps_tools_omitted(adapter: Adapter):
    request = await adapter._prepare_claude_request(
        ModelParameters(),
        [user("hello")],
    )

    assert isinstance(request.params["tools"], Omit)


async def test_web_search_invalid_definition_rejected(adapter: Adapter):
    # Missing the required `type` discriminator.
    request = adapter._prepare_claude_request(
        ModelParameters(
            tool_config=_tool_config([web_search_static_tool({"max_uses": 5})])
        ),
        [user("hello")],
    )

    with pytest.raises(ValidationError):
        await request


def test_unsupported_static_tool_rejected():
    with pytest.raises(ValidationError, match="Unsupported static tool"):
        _tool_config(
            [
                {
                    "type": "static_function",
                    "static_function": {"name": "code_execution"},
                }
            ]
        )


def test_web_search_static_tool_kept_out_of_function_tools():
    tool_config = _tool_config(
        [web_search_static_tool({"type": "web_search_20250305"})]
    )
    assert tool_config is not None
    assert tool_config.tools == []
    assert len(tool_config.static_tools) == 1
    assert tool_config.build_web_search_tools() == [
        {"type": "web_search_20250305", "name": "web_search"}
    ]
