from aidial_sdk.chat_completion import (
    Attachment,
    CacheBreakpoint,
    CustomContent,
    Message,
)
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params.function_definition import FunctionDefinition

from aidial_adapter_anthropic.dial._message import (
    AIRegularMessage,
    HumanRegularMessage,
    SystemMessage,
)


def sys(
    content: str, *, cache_breakpoint: CacheBreakpoint | None = None
) -> Message:
    return SystemMessage(
        content=content, cache_breakpoint=cache_breakpoint
    ).to_message()


def ai(
    content: str, *, cache_breakpoint: CacheBreakpoint | None = None
) -> Message:
    return AIRegularMessage(
        content=content, cache_breakpoint=cache_breakpoint
    ).to_message()


def user(
    content: str, *, cache_breakpoint: CacheBreakpoint | None = None
) -> Message:
    return HumanRegularMessage(
        content=content,
        cache_breakpoint=cache_breakpoint,
    ).to_message()


def user_with_image(content: str, image_base64: str) -> Message:
    custom_content = CustomContent(
        attachments=[Attachment(type="image/png", data=image_base64)]
    )
    return HumanRegularMessage(
        content=content, custom_content=custom_content
    ).to_message()


def function_to_tool(function: FunctionDefinition) -> ChatCompletionToolParam:
    return {"type": "function", "function": function}


GET_WEATHER_FUNCTION_WITH_REFERENCES: FunctionDefinition = {
    "name": "get_temperature",
    "description": "Get reliable information about the temperature in the given city",
    "strict": True,
    "parameters": {
        "$defs": {
            "Location": {
                "type": "string",
                "description": "The city, e.g. San Francisco",
            },
            "Unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The temperature unit to use. Infer this from the location.",
            },
        },
        "type": "object",
        "properties": {
            "location": {"$ref": "#/$defs/Location"},
            "unit": {"$ref": "#/$defs/Unit"},
        },
        "required": ["location", "unit"],
    },
}

GET_WEATHER_TOOL_WITH_REFERENCES: ChatCompletionToolParam = function_to_tool(
    GET_WEATHER_FUNCTION_WITH_REFERENCES
)

GET_WEATHER_FUNCTION: FunctionDefinition = {
    "name": "get_temperature",
    "description": "Get reliable information about the temperature in the given city",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city, e.g. San Francisco",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The temperature unit to use. Infer this from the location.",
            },
        },
        "required": ["location", "unit"],
        "additionalProperties": False,
    },
}

GET_WEATHER_TOOL: ChatCompletionToolParam = function_to_tool(
    GET_WEATHER_FUNCTION
)
