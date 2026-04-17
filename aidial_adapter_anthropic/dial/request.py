from typing import (
    Literal,
    TypeGuard,
    TypeVar,
    assert_never,
)

from aidial_sdk.chat_completion import (
    CacheBreakpoint,
    MessageContentImagePart,
    MessageContentPart,
    MessageContentTextPart,
    Role,
)
from aidial_sdk.chat_completion.request import (
    ChatCompletionRequest,
    MessageContentRefusalPart,
    ResponseFormat,
)
from aidial_sdk.exceptions import RequestValidationError
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from aidial_adapter_anthropic.adapter._errors import ValidationError
from aidial_adapter_anthropic.dial.tools import (
    ToolsConfig,
    ToolsMode,
    validate_messages,
)

MessageContent = str | list[MessageContentPart] | None
MessageContentSpecialized = (
    MessageContent
    | list[MessageContentTextPart]
    | list[MessageContentImagePart]
)

_Model = TypeVar("_Model", bound=BaseModel)


class ModelParameters(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    n: int = 1
    stop: list[str] = []
    seed: int | None = None
    max_tokens: int | None = None
    max_prompt_tokens: int | None = None
    stream: bool = False
    tool_config: ToolsConfig | None = None
    configuration: dict | None = None
    response_format: ResponseFormat | None = None
    cache_breakpoint: CacheBreakpoint | None = None

    @classmethod
    def create(cls, request: ChatCompletionRequest) -> "ModelParameters":
        stop: list[str] = []
        if request.stop is not None:
            stop = (
                [request.stop]
                if isinstance(request.stop, str)
                else request.stop
            )

        validate_messages(request)

        configuration = (
            cf.configuration
            if (cf := request.custom_fields) is not None
            else None
        )

        cache_breakpoint = (
            cf.cache_breakpoint
            if (cf := request.custom_fields) is not None
            else None
        )

        return cls(
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n or 1,
            stop=stop,
            seed=request.seed,
            max_tokens=request.max_tokens,
            max_prompt_tokens=request.max_prompt_tokens,
            stream=request.stream,
            tool_config=ToolsConfig.from_request(request),
            configuration=configuration,
            response_format=request.response_format,
            cache_breakpoint=cache_breakpoint,
        )

    @property
    def tools_mode(self) -> ToolsMode | None:
        if self.tool_config is not None:
            return self.tool_config.tools_mode
        return None

    def parse_configuration(self, cls: type[_Model]) -> _Model:
        try:
            return cls.model_validate(self.configuration or {})
        except PydanticValidationError as e:
            if self.configuration is None:
                msg = "The configuration at path 'custom_fields.configuration' is missing."
            else:
                error = e.errors()[0]
                path = ".".join(map(str, error["loc"]))
                msg = f"Invalid request. Path: 'custom_fields.configuration.{path}', error: {error['msg']}"

            raise RequestValidationError(msg) from None


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
