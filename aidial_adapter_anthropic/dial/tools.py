import logging
from enum import Enum
from typing import Any, Literal, Self

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
    StaticFunction,
    StaticTool,
)
from anthropic.types.beta import (
    BetaWebSearchTool20250305Param,
    BetaWebSearchTool20260209Param,
)
from pydantic import BaseModel, TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from aidial_adapter_anthropic.adapter._errors import ValidationError

WebSearchToolParam = (
    BetaWebSearchTool20250305Param | BetaWebSearchTool20260209Param
)

_log = logging.getLogger(__name__)


class ToolsMode(Enum):
    TOOLS = "TOOLS"
    FUNCTIONS = "FUNCTIONS"
    """
    Functions are deprecated instrument that came before tools
    """


class StaticToolName(str, Enum):
    """
    Names of the server-side (static) tools supported by the adapter.
    """

    WEB_SEARCH = "web_search"


_web_search_tool_adapter: TypeAdapter[WebSearchToolParam] = TypeAdapter(
    WebSearchToolParam
)


def _to_web_search_tool(static_function: StaticFunction) -> WebSearchToolParam:
    """
    Convert the ``web_search`` static function into the Anthropic web search
    server-tool definition. The ``name`` is defaulted from the static function
    so clients don't have to repeat it inside the configuration.

    See https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/web-search-tool
    """
    config: dict[str, Any] = dict(static_function.configuration or {})
    config.setdefault("name", static_function.name)

    try:
        return _web_search_tool_adapter.validate_python(config)
    except PydanticValidationError as e:
        error = e.errors()[0]
        path = ".".join(map(str, error["loc"]))
        raise ValidationError(
            "Invalid web search tool definition at "
            f"'static_function.configuration.{path}': {error['msg']}"
        ) from None


class ToolsConfig(BaseModel):
    tools: list[Tool]
    """
    List of functions/tools.
    """

    static_tools: list[StaticTool] = []
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
        if not self.tools and not self.static_tools:
            return
        if self.tools_mode == ToolsMode.TOOLS:
            raise ValidationError("The tools aren't supported")
        raise ValidationError("The functions aren't supported")

    def build_web_search_tools(self) -> list[WebSearchToolParam]:
        return [
            _to_web_search_tool(tool.static_function)
            for tool in self.static_tools
            if tool.static_function.name == StaticToolName.WEB_SEARCH.value
        ]

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
    def _validate_static_tool(tool: StaticTool) -> StaticTool:
        name = tool.static_function.name
        supported = [t.value for t in StaticToolName]
        if name not in supported:
            raise ValidationError(
                f"Unsupported static tool: {name!r}. "
                f"Supported static tools: {supported}."
            )
        return tool

    @classmethod
    def _split_tools(
        cls,
        tools: list[Function] | list[Tool | StaticTool],
    ) -> tuple[list[Tool], list[StaticTool]]:
        function_tools: list[Tool] = []
        static_tools: list[StaticTool] = []
        for tool in tools:
            if isinstance(tool, StaticTool):
                static_tools.append(cls._validate_static_tool(tool))
            elif isinstance(tool, Function):
                function_tools.append(Tool(type="function", function=tool))
            else:
                function_tools.append(tool)
        return function_tools, static_tools

    @classmethod
    def from_request(cls, request: AzureChatCompletionRequest) -> Self | None:
        validate_messages(request)

        tool_ids = _collect_tool_ids(request.messages)

        static_tools: list[StaticTool] = []

        if request.functions is not None:
            tools_mode = ToolsMode.FUNCTIONS
            tools, static_tools = cls._split_tools(request.functions)
            tool_choice = cls._function_call_to_tool_choice(
                request.function_call
            )
        elif request.tools is not None:
            tools_mode = ToolsMode.TOOLS
            tools, static_tools = cls._split_tools(request.tools)
            tool_choice = request.tool_choice
        elif tool_ids:
            tools_mode = ToolsMode.TOOLS
            tools = []
            tool_choice = None
        else:
            return None

        return cls(
            tools=tools,
            static_tools=static_tools,
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
