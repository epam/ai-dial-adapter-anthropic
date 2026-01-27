from typing import List, Literal, Optional, Sequence, Set, Tuple, assert_never

from aidial_sdk.chat_completion import FinishReason, Tool
from aidial_sdk.chat_completion import ToolChoice as DialToolChoice
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam as CacheControlEphemeralParam,
)
from anthropic.types.beta import BetaContentBlockParam as ContentBlockParam
from anthropic.types.beta import BetaMessageParam as MessageParam
from anthropic.types.beta import BetaStopReason as ClaudeStopReason
from anthropic.types.beta import BetaTextBlockParam as TextBlockParam
from anthropic.types.beta import BetaToolChoiceAnyParam as ToolChoiceAnyParam
from anthropic.types.beta import BetaToolChoiceAutoParam as ToolChoiceAutoParam
from anthropic.types.beta import BetaToolChoiceNoneParam as ToolChoiceNoneParam
from anthropic.types.beta import BetaToolChoiceParam as ToolChoice
from anthropic.types.beta import BetaToolChoiceToolParam as ToolChoiceToolParam
from anthropic.types.beta import BetaToolParam as ToolParam
from anthropic.types.beta import BetaUsage as Usage
from pydantic import BaseModel

from aidial_adapter_anthropic._utils.list import ListProjection, group_by
from aidial_adapter_anthropic.adapter._claude.blocks import (
    create_text_block,
    create_tool_result_block,
    create_tool_use_block,
)
from aidial_adapter_anthropic.adapter._claude.config import Configuration
from aidial_adapter_anthropic.adapter._claude.state import (
    get_message_content_from_state,
)
from aidial_adapter_anthropic.adapter._errors import ValidationError
from aidial_adapter_anthropic.dial._attachments import (
    AttachmentProcessors,
    WithResources,
)
from aidial_adapter_anthropic.dial._message import (
    AIRegularMessage,
    AIToolCallMessage,
    BaseMessage,
    HumanRegularMessage,
    HumanToolResultMessage,
    SystemMessage,
)
from aidial_adapter_anthropic.dial.token_usage import TokenUsage
from aidial_adapter_anthropic.dial.tools import ToolsConfig, ToolsMode

DialMessage = BaseMessage | HumanToolResultMessage | AIToolCallMessage

ClaudeMessage = WithResources[ContentBlockParam]

_claude_cache_breakpoint = CacheControlEphemeralParam(type="ephemeral")


def _add_cache_control(
    message: DialMessage, claude_messages: Sequence[ContentBlockParam]
) -> None:
    if message.cache_breakpoint is not None:
        for block in reversed(claude_messages):
            if (
                isinstance(block, dict)
                and block["type"] != "thinking"
                and block["type"] != "redacted_thinking"
            ):
                block["cache_control"] = _claude_cache_breakpoint
                break


def _get_claude_message_role(
    dial_message: (
        AIRegularMessage
        | AIToolCallMessage
        | HumanRegularMessage
        | HumanToolResultMessage
    ),
) -> Literal["assistant", "user"]:
    match dial_message:
        case AIRegularMessage() | AIToolCallMessage():
            return "assistant"
        case HumanRegularMessage() | HumanToolResultMessage():
            return "user"
        case _:
            assert_never(dial_message)


_Elem = Tuple[WithResources[MessageParam], Set[int]]


def _merge_messages_with_same_role(
    messages: ListProjection[WithResources[MessageParam]],
) -> ListProjection[WithResources[MessageParam]]:

    def _key(message: _Elem) -> str:
        return message[0].payload["role"]

    def _merge_message_param(
        msg1: MessageParam, msg2: MessageParam
    ) -> MessageParam:
        content1 = msg1["content"]
        content2 = msg2["content"]

        if isinstance(content1, str):
            content1 = [TextBlockParam(type="text", text=content1)]

        if isinstance(content2, str):
            content2 = [TextBlockParam(type="text", text=content2)]

        return MessageParam(
            role=msg1["role"],
            content=list(content1) + list(content2),
        )

    def _merge(a: _Elem, b: _Elem) -> _Elem:
        (msg1, set1), (msg2, set2) = a, b
        payload = _merge_message_param(msg1.payload, msg2.payload)
        resources = msg1.resources + msg2.resources
        return (WithResources(payload, resources), set1 | set2)

    return ListProjection(group_by(messages.list, _key, lambda x: x, _merge))


async def _get_claude_blocks(
    handlers: AttachmentProcessors[
        TextBlockParam, ContentBlockParam, Configuration
    ],
    message: (
        HumanRegularMessage
        | AIRegularMessage
        | AIToolCallMessage
        | HumanToolResultMessage
    ),
    message_idx: int,
) -> WithResources[Sequence[ContentBlockParam]]:

    match message:
        case HumanRegularMessage():
            return await handlers.process_attachments(message)

        case HumanToolResultMessage():
            blocks = [create_tool_result_block(message)]
            return WithResources(payload=blocks)

        case AIRegularMessage():
            content = await handlers.process_attachments(message)

            # Take the message content from the state if possible,
            # since it may include certain content blocks that
            # are missing from the DIAL message itself,
            # such as thinking signatures and redacted thinking blocks.
            if state := get_message_content_from_state(message_idx, message):
                content.payload = state

            return content

        case AIToolCallMessage():
            blocks = [create_tool_use_block(call) for call in message.calls]
            if text_content := message.content:
                blocks.insert(0, create_text_block(text_content))

            content = WithResources(payload=blocks)
            if state := get_message_content_from_state(message_idx, message):
                content.payload = state

            return content

        case _:
            assert_never(message)


async def to_claude_messages(
    handlers: AttachmentProcessors[
        TextBlockParam, ContentBlockParam, Configuration
    ],
    messages: List[DialMessage],
) -> Tuple[List[TextBlockParam], ListProjection[WithResources[MessageParam]]]:

    idx_offset: int = 0
    system_messages: List[TextBlockParam] = []

    for message in messages:
        if not isinstance(message, SystemMessage):
            break

        idx_offset += 1
        sys_content = await handlers.process_system_message(message)
        _add_cache_control(message, sys_content)

        system_messages.extend(sys_content)

    claude_messages: ListProjection[WithResources[MessageParam]] = (
        ListProjection()
    )

    for idx, message in enumerate(messages[idx_offset:], start=idx_offset):
        if isinstance(message, SystemMessage):
            raise ValidationError(
                "System and developer messages are only allowed in the beginning of the conversation."
            )

        blocks = await _get_claude_blocks(handlers, message, idx)
        _add_cache_control(message, blocks.payload)

        role = _get_claude_message_role(message)
        claude_message = WithResources(
            payload=MessageParam(role=role, content=blocks.payload),
            resources=blocks.resources,
        )

        claude_messages.append(claude_message, idx)

    return system_messages, _merge_messages_with_same_role(claude_messages)


def to_dial_finish_reason(
    finish_reason: Optional[ClaudeStopReason],
    tools_mode: ToolsMode | None,
) -> FinishReason:
    if finish_reason is None:
        return FinishReason.STOP

    match finish_reason:
        case "end_turn":
            return FinishReason.STOP
        case "max_tokens" | "model_context_window_exceeded":
            return FinishReason.LENGTH
        case "stop_sequence" | "pause_turn" | "refusal":
            return FinishReason.STOP
        case "tool_use":
            match tools_mode:
                case ToolsMode.TOOLS:
                    return FinishReason.TOOL_CALLS
                case ToolsMode.FUNCTIONS:
                    return FinishReason.FUNCTION_CALL
                case None:
                    raise ValidationError(
                        "A model has called a tool, but no tools were given to the model in the first place."
                    )
                case _:
                    assert_never(tools_mode)

        case _:
            assert_never(finish_reason)


def to_dial_usage(usage: Usage) -> TokenUsage:
    read = usage.cache_creation_input_tokens or 0
    write = usage.cache_read_input_tokens or 0
    return TokenUsage(
        completion_tokens=usage.output_tokens,
        prompt_tokens=usage.input_tokens + read + write,
        cache_write_input_tokens=read,
        cache_read_input_tokens=write,
    )


def _to_claude_tool(tool: Tool) -> ToolParam:
    function = tool.function
    tool_param = ToolParam(
        input_schema=function.parameters
        or {"type": "object", "properties": {}},
        name=function.name,
        description=function.description or "",
    )

    if tool.custom_fields and tool.custom_fields.cache_breakpoint:
        tool_param["cache_control"] = _claude_cache_breakpoint

    return tool_param


def _to_claude_tool_choice(
    tool_choice: Literal["auto", "none", "required"] | DialToolChoice,
) -> ToolChoice:
    # NOTE tool_choice.disable_parallel_tool_use=True option isn't supported
    # by older Claude3 versions, so we limit the number of generated function calls
    # to one in the adapter itself for the functions mode.

    match tool_choice:
        case DialToolChoice(function=function):
            return ToolChoiceToolParam(type="tool", name=function.name)
        case "required":
            return ToolChoiceAnyParam(type="any")
        case "auto":
            return ToolChoiceAutoParam(type="auto")
        case "none":
            return ToolChoiceNoneParam(type="none")
        case _:
            assert_never(tool_choice)


class ClaudeToolsConfig(BaseModel):
    tools: List[ToolParam]
    tool_choice: ToolChoice


def to_claude_tool_config(
    tools_config: ToolsConfig | None,
) -> ClaudeToolsConfig | None:
    if tools_config is None or not tools_config.tools:
        return None

    tools = [_to_claude_tool(tool) for tool in tools_config.tools]
    tool_choice = _to_claude_tool_choice(tools_config.tool_choice)
    return ClaudeToolsConfig(tools=tools, tool_choice=tool_choice)
