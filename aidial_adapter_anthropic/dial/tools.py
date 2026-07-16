import logging
from enum import Enum
from typing import Literal, Self

from aidial_sdk.chat_completion import (
    Function,
    FunctionChoice,
    Message,
    Role,
    Tool,
    ToolChoice,
)
from aidial_sdk.chat_completion.request import (
    AzureChatCompletionRequest,
    StaticTool,
)
from pydantic import BaseModel

from aidial_adapter_anthropic.adapter._errors import ValidationError
from aidial_adapter_anthropic.dial.static_tools import (
    WebSearchToolParam,
    parse_static_function,
)

_log = logging.getLogger(__name__)


class ToolsMode(Enum):
    TOOLS = "TOOLS"
    FUNCTIONS = "FUNCTIONS"
    """
    Functions are deprecated instrument that came before tools
    """


class ToolsConfig(BaseModel):
    tools: list[Tool]
    """
    List of functions/tools.
    """

    web_search_tools: list[WebSearchToolParam] = []
    """
    List of server-side (static) tools, e.g. web search.
    Executed on the provider's side rather than round-tripped to the client.
    """

    tools_mode: ToolsMode

    tool_choice: Literal["auto", "none", "required"] | ToolChoice

    tool_ids: dict[str, str]
    """
    Mapping from tool call IDs to corresponding tool names.
    Empty when there are no tool calls in the messages.
    """

    def not_supported(self) -> None:
        if not self.tools and not self.web_search_tools:
            return
        if self.tools_mode == ToolsMode.TOOLS:
            raise ValidationError("The tools aren't supported")
        raise ValidationError("The functions aren't supported")

    def create_fresh_tool_call_id(self, tool_name: str) -> str:
        idx = 1
        while True:
            tool_id = f"{tool_name}_{idx}"
            if tool_id not in self.tool_ids:
                self.tool_ids[tool_id] = tool_name
                return tool_id
            idx += 1

    def get_tool_name(self, tool_call_id: str) -> str:
        tool_name = self.tool_ids.get(tool_call_id)
        if tool_name is None:
            raise ValidationError(f"Tool call ID not found: {self.tool_ids}")
        return tool_name

    @staticmethod
    def _function_call_to_tool_choice(
        function_call: Literal["auto", "none"] | FunctionChoice | None,
    ) -> Literal["auto", "none", "required"] | ToolChoice | None:
        match function_call:
            case FunctionChoice():
                return ToolChoice(type="function", function=function_call)
            case _:
                return function_call

    @staticmethod
    def _split_tools(
        tools: list[Function] | list[Tool | StaticTool],
    ) -> tuple[list[Tool], list[WebSearchToolParam]]:
        function_tools: list[Tool] = []
        web_search_tools: list[WebSearchToolParam] = []
        for idx, tool in enumerate(tools):
            if isinstance(tool, StaticTool):
                web_search_tools.append(
                    parse_static_function(str(idx), tool.static_function)
                )
            elif isinstance(tool, Function):
                function_tools.append(Tool(type="function", function=tool))
            else:
                function_tools.append(tool)
        return function_tools, web_search_tools

    @classmethod
    def from_request(cls, request: AzureChatCompletionRequest) -> Self | None:
        validate_messages(request)

        tool_ids = _collect_tool_ids(request.messages)

        web_search_tools: list[WebSearchToolParam] = []

        if request.functions is not None:
            tools_mode = ToolsMode.FUNCTIONS
            tools, web_search_tools = cls._split_tools(request.functions)
            tool_choice = cls._function_call_to_tool_choice(
                request.function_call
            )
        elif request.tools is not None:
            tools_mode = ToolsMode.TOOLS
            tools, web_search_tools = cls._split_tools(request.tools)
            tool_choice = request.tool_choice
        elif tool_ids:
            tools_mode = ToolsMode.TOOLS
            tools = []
            tool_choice = None
        else:
            return None

        return cls(
            tools=tools,
            web_search_tools=web_search_tools,
            tools_mode=tools_mode,
            tool_choice=tool_choice or "auto",
            tool_ids=tool_ids,
        )


def validate_messages(request: AzureChatCompletionRequest) -> None:
    decl_tools = request.tools is not None
    decl_functions = request.functions is not None

    if decl_functions and decl_tools:
        raise ValidationError("Both functions and tools are not allowed")

    def warn(msg: str):
        _log.warning(
            f"The request is incomplete: {msg}. The model may misbehave."
        )

    tool_defs_are_missing = (
        "the request is missing tool definitions in the 'tools' field"
    )
    func_defs_are_missing = (
        "the request is missing function definitions in the 'functions' field"
    )

    for idx, message in enumerate(request.messages):
        if (
            message.role == Role.ASSISTANT
            and message.tool_calls is not None
            and not decl_tools
        ):
            warn(
                f"'messages[{idx}]' is an Assistant message with a tool call, but {tool_defs_are_missing}"
            )
        if (
            message.role == Role.ASSISTANT
            and message.function_call is not None
            and not decl_functions
        ):
            warn(
                f"'messages[{idx}]' is an Assistant messages with a function call, but {func_defs_are_missing}"
            )
        if message.role == Role.FUNCTION and not decl_functions:
            warn(
                f"'messages[{idx}]' is a Function message, but {func_defs_are_missing}"
            )
        if message.role == Role.TOOL and not decl_tools:
            warn(
                f"'messages[{idx}]' is a Tool message, but {tool_defs_are_missing}"
            )


def _collect_tool_ids(messages: list[Message]) -> dict[str, str]:
    ret: dict[str, str] = {}

    for message in messages:
        if message.role == Role.ASSISTANT and message.tool_calls is not None:
            for tool_call in message.tool_calls:
                ret[tool_call.id] = tool_call.function.name

    return ret
