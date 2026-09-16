from aidial_sdk.chat_completion import (
    Attachment,
    CustomContent,
    FunctionCall,
    ToolCall,
)
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params.function_definition import FunctionDefinition

from aidial_adapter_anthropic._utils.list import ListProjection
from aidial_adapter_anthropic.dial._message import (
    AdapterMessage,
    AIRegularMessage,
    AIToolCallMessage,
    HumanRegularMessage,
    HumanToolResultMessage,
    SystemMessage,
)


def prompt(*messages: AdapterMessage) -> ListProjection[AdapterMessage]:
    return ListProjection.create(list(messages))


def sys(content: str) -> SystemMessage:
    return SystemMessage(content=content)


def ai(content: str) -> AIRegularMessage:
    return AIRegularMessage(content=content)


def user(content: str) -> HumanRegularMessage:
    return HumanRegularMessage(content=content)


def user_with_image(content: str, image_base64: str) -> HumanRegularMessage:
    custom_content = CustomContent(
        attachments=[Attachment(type="image/png", data=image_base64)]
    )
    return HumanRegularMessage(content=content, custom_content=custom_content)


def ai_tool_call(
    message_id: str, content: str = "call", name: str = "test_tool"
) -> AIToolCallMessage:
    return AIToolCallMessage(
        content=content,
        calls=[
            ToolCall(
                id=message_id,
                type="function",
                function=FunctionCall(name=name, arguments="{}"),
            )
        ],
    )


def tool_result(
    message_id: str, content: str = "result"
) -> HumanToolResultMessage:
    return HumanToolResultMessage(id=message_id, content=content)


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
