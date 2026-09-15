from dataclasses import dataclass, field
from typing import Self, TypeVar

from aidial_sdk.chat_completion import CacheBreakpoint
from aidial_sdk.chat_completion.request import (
    ChatCompletionRequest,
    ReasoningEffort,
    ResponseFormat,
)
from aidial_sdk.exceptions import RequestValidationError
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from aidial_adapter_anthropic._utils.list import ListProjection
from aidial_adapter_anthropic.dial._message import (
    AdapterMessage,
    parse_dial_message,
)
from aidial_adapter_anthropic.dial.tools import (
    ToolsConfig,
    ToolsMode,
    validate_tools_usage,
)

_Model = TypeVar("_Model", bound=BaseModel)


@dataclass
class AdapterRequest:
    messages: ListProjection[AdapterMessage]
    temperature: float | None = None
    top_p: float | None = None
    n: int = 1
    stop: list[str] = field(default_factory=list)
    seed: int | None = None
    max_tokens: int | None = None
    max_prompt_tokens: int | None = None
    stream: bool = False
    tool_config: ToolsConfig | None = None
    configuration: dict | None = None
    response_format: ResponseFormat | None = None
    cache_breakpoint: CacheBreakpoint | None = None
    reasoning_effort: ReasoningEffort | None = None

    @classmethod
    def create(cls, request: ChatCompletionRequest) -> Self:
        stop: list[str] = []
        if request.stop is not None:
            stop = (
                [request.stop]
                if isinstance(request.stop, str)
                else request.stop
            )

        validate_tools_usage(request)

        cf = request.custom_fields
        configuration = cf.configuration if cf is not None else None
        cache_breakpoint = cf.cache_breakpoint if cf is not None else None
        messages = [parse_dial_message(m) for m in request.messages]

        return cls(
            messages=ListProjection.create(messages),
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
            reasoning_effort=request.reasoning_effort,
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
