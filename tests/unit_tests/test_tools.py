import pytest
from aidial_sdk.chat_completion.request import AzureChatCompletionRequest
from aidial_sdk.exceptions import RequestValidationError
from anthropic import Omit
from openai.types.chat import ChatCompletionToolParam

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

WEB_SEARCH_TOOL_REQUEST = {
    "type": "web_search_20250305",
    "name": "web_search",
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

    assert ("$defs" in claude_tools.tools[0]["input_schema"]) == has_refs


def test_tools_schemas_with_references():
    _run_schema_references_check(GET_WEATHER_TOOL_WITH_REFERENCES, True)


def test_tools_schemas_without_references():
    _run_schema_references_check(GET_WEATHER_TOOL, False)


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        pytest.param(
            {"type": "web_search_20250305", "name": "web_search"},
            [{"type": "web_search_20250305", "name": "web_search"}],
            id="minimal-20250305",
        ),
        pytest.param(
            {"type": "web_search_20260209", "name": "web_search"},
            [{"type": "web_search_20260209", "name": "web_search"}],
            id="minimal-20260209",
        ),
        pytest.param(
            WEB_SEARCH_TOOL_REQUEST,
            [WEB_SEARCH_TOOL_REQUEST],
            id="optional-fields-preserved",
        ),
    ],
)
async def test_web_search_passthrough_payload(
    adapter: Adapter, tool: dict, expected: list[dict]
):
    request = await adapter._prepare_claude_request(
        ModelParameters(configuration={"web_search": tool}),
        [user("What is the weather in NYC?")],
    )

    assert request.params["tools"] == expected


async def test_web_search_tool_choice_left_default(adapter: Adapter):
    # Web search is a server tool: enabling it must not force a tool_choice.
    request = await adapter._prepare_claude_request(
        ModelParameters(configuration={"web_search": WEB_SEARCH_TOOL_REQUEST}),
        [user("hello")],
    )

    assert isinstance(request.params["tool_choice"], Omit)


async def test_web_search_appended_after_function_tools(adapter: Adapter):
    dial_request = AzureChatCompletionRequest.model_validate(
        {"messages": [], "tools": [GET_WEATHER_TOOL]}
    )
    tool_config = ToolsConfig.from_request(dial_request)

    request = await adapter._prepare_claude_request(
        ModelParameters(
            configuration={"web_search": WEB_SEARCH_TOOL_REQUEST},
            tool_config=tool_config,
        ),
        [user("What is the weather in NYC?")],
    )

    tools = request.params["tools"]
    assert not isinstance(tools, Omit)
    assert len(tools) == 2
    assert tools[0]["name"] == "get_temperature"
    assert tools[-1] == WEB_SEARCH_TOOL_REQUEST


async def test_web_search_preserves_existing_function_tool_choice(
    adapter: Adapter,
):
    dial_request = AzureChatCompletionRequest.model_validate(
        {
            "messages": [],
            "tools": [GET_WEATHER_TOOL],
            "tool_choice": "required",
        }
    )
    tool_config = ToolsConfig.from_request(dial_request)
    assert tool_config is not None

    request = await adapter._prepare_claude_request(
        ModelParameters(
            configuration={"web_search": WEB_SEARCH_TOOL_REQUEST},
            tool_config=tool_config,
        ),
        [user("What is the weather in NYC?")],
    )

    expected_config = to_claude_tool_config(tool_config)
    assert expected_config is not None
    assert request.params["tool_choice"] == expected_config.tool_choice


async def test_no_web_search_keeps_tools_omitted(adapter: Adapter):
    request = await adapter._prepare_claude_request(
        ModelParameters(),
        [user("hello")],
    )

    assert isinstance(request.params["tools"], Omit)


@pytest.mark.parametrize(
    "invalid_tool",
    [
        pytest.param({"name": "web_search"}, id="missing-type"),
        pytest.param({"type": "web_search_20250305"}, id="missing-name"),
    ],
)
async def test_web_search_invalid_definition_rejected(
    adapter: Adapter, invalid_tool: dict
):
    request = adapter._prepare_claude_request(
        ModelParameters(configuration={"web_search": invalid_tool}),
        [user("hello")],
    )

    with pytest.raises(RequestValidationError):
        await request
