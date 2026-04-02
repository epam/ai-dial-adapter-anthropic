from aidial_sdk.chat_completion.request import AzureChatCompletionRequest
from openai.types.chat import ChatCompletionToolParam

from aidial_adapter_anthropic.adapter._claude.converters import (
    to_claude_tool_config,
)
from aidial_adapter_anthropic.dial.tools import ToolsConfig
from tests.utils.openai import (
    GET_WEATHER_TOOL,
    GET_WEATHER_TOOL_WITH_REFERENCES,
)


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
