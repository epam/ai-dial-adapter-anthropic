from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Literal, Self, TypeGuard, assert_never

from aidial_sdk.chat_completion import (
    Attachment,
    CustomContent,
    FunctionCall,
    MessageContentAudioPart,
    MessageContentFilePart,
    MessageContentImagePart,
    MessageContentPart,
    MessageContentTextPart,
    MessageCustomFields,
    PromptCacheBreakpoint,
    Role,
    ToolCall,
)
from aidial_sdk.chat_completion import CacheBreakpoint as DialCacheBreakpoint
from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion.request import MessageContentRefusalPart
from pydantic import BaseModel

from aidial_adapter_anthropic._utils.cache import CacheBreakpoint
from aidial_adapter_anthropic.adapter._errors import ValidationError

MessageContent = str | list[MessageContentPart] | None
MessageContentSpecialized = (
    MessageContent
    | list[MessageContentTextPart]
    | list[MessageContentImagePart]
)


class MessageCacheBreakpoints(BaseModel):
    """The cache breakpoints of a single message."""

    trailing: CacheBreakpoint | None = None
    """DIAL cache breakpoint - per message"""

    parts: dict[int, CacheBreakpoint | None] = {}
    """Native cache breakpoints - per content part"""

    def __bool__(self) -> bool:
        return self.trailing is not None or bool(self.parts)

    def all(self) -> Iterator[CacheBreakpoint]:
        if self.trailing is not None:
            yield self.trailing
        yield from (brk for brk in self.parts.values() if brk is not None)

    def to_custom_fields(self) -> MessageCustomFields | None:
        # The native breakpoints round-trip within the content parts themselves.
        if (breakpoint := self.trailing) is None:
            return None
        # The DIAL breakpoint carries the TTL as an undeclared extra field.
        ttl = {} if breakpoint.ttl is None else {"ttl": breakpoint.ttl}
        return MessageCustomFields(cache_breakpoint=DialCacheBreakpoint(**ttl))

    @classmethod
    def parse(cls, message: DialMessage) -> Self:
        parts = {
            idx: None if brk is None else CacheBreakpoint()
            for idx, brk in _native_cache_breakpoints(message.content)
        }
        if parts:
            return cls(parts=parts)

        cf = message.custom_fields
        return cls(
            trailing=CacheBreakpoint.from_dial(
                cf.cache_breakpoint if cf else None
            )
        )


def _native_cache_breakpoints(
    content: MessageContent,
) -> Iterator[tuple[int, PromptCacheBreakpoint | None]]:
    if not isinstance(content, list):
        return

    for idx, part in enumerate(content):
        if not isinstance(part, MessageContentRefusalPart):
            yield (idx, part.prompt_cache_breakpoint)


class MessageABC(ABC, BaseModel):
    cache_breakpoints: MessageCacheBreakpoints = MessageCacheBreakpoints()

    @property
    def custom_fields(self) -> MessageCustomFields | None:
        return self.cache_breakpoints.to_custom_fields()

    @abstractmethod
    def to_message(self) -> DialMessage: ...

    @classmethod
    @abstractmethod
    def from_message(cls, message: DialMessage) -> Self | None: ...


class BaseMessageABC(MessageABC):
    @property
    @abstractmethod
    def text_content(self) -> str: ...


class SystemMessage(BaseMessageABC):
    content: str | list[MessageContentTextPart]
    is_developer: bool = False

    def to_message(self) -> DialMessage:
        return DialMessage(
            role=Role.DEVELOPER if self.is_developer else Role.SYSTEM,
            content=to_message_content(self.content),
            custom_fields=self.custom_fields,
        )

    @classmethod
    def from_message(cls, message: DialMessage) -> Self | None:
        if not is_system_role(message.role):
            return None

        content = message.content

        if not is_text_content(content):
            raise ValidationError(
                "System message is expected to be a string or a list of text content parts"
            )

        return cls(
            cache_breakpoints=MessageCacheBreakpoints.parse(message),
            is_developer=message.role == Role.DEVELOPER,
            content=content,
        )

    @property
    def text_content(self) -> str:
        return collect_text_content(self.content)


class HumanRegularMessage(BaseMessageABC):
    content: str | list[MessageContentPart]
    custom_content: CustomContent | None = None

    def to_message(self) -> DialMessage:
        return DialMessage(
            role=Role.USER,
            content=self.content,
            custom_content=self.custom_content,
            custom_fields=self.custom_fields,
        )

    @classmethod
    def from_message(cls, message: DialMessage) -> Self | None:
        if message.role != Role.USER:
            return None

        content = message.content
        if content is None:
            raise ValidationError(
                "User message is expected to have content field"
            )

        return cls(
            cache_breakpoints=MessageCacheBreakpoints.parse(message),
            content=content,
            custom_content=message.custom_content,
        )

    @property
    def text_content(self) -> str:
        return collect_text_content(self.content)

    @property
    def attachments(self) -> list[Attachment]:
        return (
            self.custom_content.attachments or [] if self.custom_content else []
        )


class HumanToolResultMessage(MessageABC):
    id: str
    content: str
    custom_content: CustomContent | None = None

    def to_message(self) -> DialMessage:
        return DialMessage(
            role=Role.TOOL,
            tool_call_id=self.id,
            content=self.content,
            custom_content=self.custom_content,
            custom_fields=self.custom_fields,
        )

    @classmethod
    def from_message(cls, message: DialMessage) -> Self | None:
        if message.role != Role.TOOL:
            return None

        if not is_plain_text_content(message.content):
            raise ValidationError(
                "The tool message shouldn't contain content parts"
            )

        if message.content is None or message.tool_call_id is None:
            raise ValidationError(
                "The tool message is expected to have content and tool_call_id fields"
            )

        return cls(
            cache_breakpoints=MessageCacheBreakpoints.parse(message),
            id=message.tool_call_id,
            content=message.content,
            custom_content=message.custom_content,
        )

    @property
    def attachments(self) -> list[Attachment]:
        return (
            self.custom_content.attachments or [] if self.custom_content else []
        )


class HumanFunctionResultMessage(MessageABC):
    name: str
    content: str

    def to_message(self) -> DialMessage:
        return DialMessage(
            role=Role.FUNCTION,
            name=self.name,
            content=self.content,
            custom_fields=self.custom_fields,
        )

    @classmethod
    def from_message(cls, message: DialMessage) -> Self | None:
        if message.role != Role.FUNCTION:
            return None

        if not is_plain_text_content(message.content):
            raise ValidationError(
                "The function message shouldn't contain content parts"
            )

        if message.content is None or message.name is None:
            raise ValidationError(
                "The function message is expected to have content and name fields"
            )

        return cls(
            cache_breakpoints=MessageCacheBreakpoints.parse(message),
            name=message.name,
            content=message.content,
        )


class AIRegularMessage(BaseMessageABC):
    content: str | list[MessageContentPart]
    """
    According to Azure OpenAI API, the assistant message could only have textual content.
    However, we leave a loophole to provide image content parts just in case
    one day multi-modal Bedrock models will be able to accept images in assistant messages.
    """

    custom_content: CustomContent | None = None

    def to_message(self) -> DialMessage:
        return DialMessage(
            role=Role.ASSISTANT,
            content=self.content,  # type: ignore
            custom_content=self.custom_content,
            custom_fields=self.custom_fields,
        )

    @classmethod
    def from_message(cls, message: DialMessage) -> Self | None:
        if message.role != Role.ASSISTANT:
            return None

        if message.function_call is not None or message.tool_calls is not None:
            return None

        content = message.content
        if content is None:
            raise ValidationError(
                "Assistant message is expected to have content field"
            )

        return cls(
            cache_breakpoints=MessageCacheBreakpoints.parse(message),
            content=content,
            custom_content=message.custom_content,
        )

    @property
    def text_content(self) -> str:
        return collect_text_content(self.content)

    @property
    def attachments(self) -> list[Attachment]:
        return (
            self.custom_content.attachments or [] if self.custom_content else []
        )


class AIToolCallMessage(MessageABC):
    calls: list[ToolCall]
    content: str | None = None
    custom_content: CustomContent | None = None

    def to_message(self) -> DialMessage:
        return DialMessage(
            role=Role.ASSISTANT,
            content=self.content,
            tool_calls=self.calls,
            custom_content=self.custom_content,
            custom_fields=self.custom_fields,
        )

    @classmethod
    def from_message(cls, message: DialMessage) -> Self | None:
        if message.role != Role.ASSISTANT:
            return None

        if message.tool_calls is None or message.function_call is not None:
            return None

        if not is_plain_text_content(message.content):
            raise ValidationError(
                "The assistant message with tool calls shouldn't contain content parts"
            )

        return cls(
            cache_breakpoints=MessageCacheBreakpoints.parse(message),
            calls=message.tool_calls,
            content=message.content,
            custom_content=message.custom_content,
        )


class AIFunctionCallMessage(MessageABC):
    call: FunctionCall
    content: str | None = None

    def to_message(self) -> DialMessage:
        return DialMessage(
            role=Role.ASSISTANT,
            content=self.content,
            function_call=self.call,
            custom_fields=self.custom_fields,
        )

    @classmethod
    def from_message(cls, message: DialMessage) -> Self | None:
        if message.role != Role.ASSISTANT:
            return None

        if message.function_call is None or message.tool_calls is not None:
            return None

        if not is_plain_text_content(message.content):
            raise ValidationError(
                "The assistant message with function call shouldn't contain content parts"
            )

        return cls(
            cache_breakpoints=MessageCacheBreakpoints.parse(message),
            call=message.function_call,
            content=message.content,
        )


BaseMessage = SystemMessage | HumanRegularMessage | AIRegularMessage

ToolMessage = (
    HumanToolResultMessage
    | HumanFunctionResultMessage
    | AIToolCallMessage
    | AIFunctionCallMessage
)

AdapterMessage = BaseMessage | ToolMessage


def parse_dial_message(msg: DialMessage) -> AdapterMessage:
    message = (
        SystemMessage.from_message(msg)
        or HumanRegularMessage.from_message(msg)
        or HumanToolResultMessage.from_message(msg)
        or HumanFunctionResultMessage.from_message(msg)
        or AIRegularMessage.from_message(msg)
        or AIToolCallMessage.from_message(msg)
        or AIFunctionCallMessage.from_message(msg)
    )

    if message is None:
        raise ValidationError("Unknown message type or invalid message")

    return message


def collect_text_content(
    content: MessageContentSpecialized, delimiter: str = "\n\n"
) -> str:
    match content:
        case None:
            return ""
        case str():
            return content
        case list():
            texts: list[str] = []
            for part in content:
                match part:
                    case MessageContentTextPart(text=text):
                        texts.append(text)
                    case MessageContentImagePart():
                        raise ValidationError(
                            "Can't extract text from an image content part"
                        )
                    case MessageContentAudioPart():
                        raise ValidationError(
                            "Can't extract text from an audio content part"
                        )
                    case MessageContentFilePart():
                        raise ValidationError(
                            "Can't extract text from a file content part"
                        )
                    case MessageContentRefusalPart():
                        raise ValidationError(
                            "Can't extract text from a refusal content part"
                        )
                    case _:
                        assert_never(part)
            return delimiter.join(texts)
        case _:
            assert_never(content)


def to_message_content(content: MessageContentSpecialized) -> MessageContent:
    match content:
        case None | str():
            return content
        case list():
            return [*content]
        case _:
            assert_never(content)


def is_text_content(
    content: MessageContent,
) -> TypeGuard[str | list[MessageContentTextPart]]:
    match content:
        case None:
            return False
        case str():
            return True
        case list():
            return all(
                isinstance(part, MessageContentTextPart) for part in content
            )
        case _:
            assert_never(content)


def is_plain_text_content(content: MessageContent) -> TypeGuard[str | None]:
    return content is None or isinstance(content, str)


def is_system_role(
    role: Role,
) -> TypeGuard[Literal[Role.SYSTEM, Role.DEVELOPER]]:
    return role in [Role.SYSTEM, Role.DEVELOPER]
