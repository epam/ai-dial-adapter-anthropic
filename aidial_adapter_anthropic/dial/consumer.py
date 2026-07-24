from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Protocol, Self

from aidial_sdk.chat_completion import (
    Attachment,
    Choice,
    FinishReason,
    FunctionCall,
    Response,
    ToolCall,
)

from aidial_adapter_anthropic.adapter._truncate_prompt import DiscardedMessages
from aidial_adapter_anthropic.dial._lazy_stage import LazyStage
from aidial_adapter_anthropic.dial.token_usage import TokenUsage


class _ArgumentConsumer(Protocol):
    def append_arguments(self, arguments: str) -> Self: ...


@dataclasses.dataclass
class ToolUseMessage:
    call: _ArgumentConsumer
    snapshot: str

    def append_arguments(self, arguments: str) -> Self:
        self.call.append_arguments(arguments)
        self.snapshot += arguments
        return self

    def close(self) -> Self:
        if not self.snapshot.strip():
            self.append_arguments("{}")
        return self


class Consumer(AbstractAsyncContextManager, ABC):
    @abstractmethod
    def fork(self) -> Consumer: ...

    @property
    @abstractmethod
    def choice(self) -> Choice: ...

    @abstractmethod
    async def close_content(
        self, finish_reason: FinishReason | None = None
    ): ...

    @abstractmethod
    def get_response(self) -> Response: ...

    @abstractmethod
    async def append_content(self, content: str): ...

    @abstractmethod
    async def add_attachment(self, attachment: Attachment): ...

    @abstractmethod
    async def add_citation_attachment(
        self, document_id: int, document: Attachment | None
    ) -> int: ...

    @abstractmethod
    async def add_usage(self, usage: TokenUsage): ...

    @abstractmethod
    async def set_discarded_messages(
        self, discarded_messages: DiscardedMessages | None
    ): ...

    @abstractmethod
    async def get_discarded_messages(self) -> DiscardedMessages | None: ...

    @abstractmethod
    async def create_function_tool_call(
        self, call: ToolCall
    ) -> ToolUseMessage: ...

    @abstractmethod
    async def create_function_call(
        self, call: FunctionCall
    ) -> ToolUseMessage: ...

    @property
    @abstractmethod
    def has_function_call(self) -> bool: ...

    def create_stage(self, title: str) -> LazyStage:
        # NOTE: eta conversion to `factory = self.choice.create_stage`
        # is invalid, since `self.choice` must be created lazily.
        def factory(content: str):
            return self.choice.create_stage(content)

        return LazyStage(factory, title)


@dataclasses.dataclass
class _ResponseState:
    usage: TokenUsage | None = None
    discarded_messages: DiscardedMessages | None = None

    def add_usage(self, usage: TokenUsage):
        self.usage = (self.usage or TokenUsage()).accumulate(usage)

    def flush(self, response: Response):
        if self.usage is not None:
            response.set_usage(
                prompt_tokens=self.usage.prompt_tokens,
                completion_tokens=self.usage.completion_tokens,
                prompt_tokens_details={
                    "cached_tokens": self.usage.cache_read_input_tokens,
                    "cache_write_tokens": self.usage.cache_write_input_tokens,
                },
                completion_tokens_details={
                    "reasoning_tokens": self.usage.reasoning_tokens
                },
            )

        if self.discarded_messages is not None:
            response.set_discarded_messages(self.discarded_messages)


class ChoiceConsumer(Consumer):
    response: Response
    _response_state: _ResponseState

    _is_root: bool
    _choice: Choice | None
    _tool_calls: list[ToolUseMessage]
    _citations: dict[int, tuple[int, Attachment | None]]

    def __init__(
        self,
        response: Response,
        *,
        response_state: _ResponseState | None = None,
    ):
        self.response = response
        self._response_state = response_state or _ResponseState()

        self._is_root = response_state is None
        self._choice = None
        self._tool_calls = []
        self._citations = {}

    def fork(self) -> Consumer:
        return ChoiceConsumer(
            self.response, response_state=self._response_state
        )

    @property
    def choice(self) -> Choice:
        if self._choice is None:
            choice = self._choice = self.response.create_choice()
            # Delay opening a choice to the very last moment
            # so as to give opportunity for exceptions to bubble up to
            # the level of HTTP response (instead of error objects in a stream).
            choice.open()
            return choice
        else:
            return self._choice

    async def __aenter__(self) -> ChoiceConsumer:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        for tool_call in self._tool_calls:
            tool_call.close()

        if exc is None and self._choice is not None:
            self._choice.close()

        if self._is_root:
            self._response_state.flush(self.response)

        return False

    def get_response(self) -> Response:
        return self.response

    async def close_content(self, finish_reason: FinishReason | None = None):
        # Choice.close(finish_reason: Optional[FinishReason]) can be called only once
        # Currently, there's no other way to explicitly set the finish reason
        self.choice._last_finish_reason = finish_reason

    async def append_content(self, content: str):
        self.choice.append_content(content)

    async def add_attachment(self, attachment: Attachment):
        self.choice.add_attachment(attachment)

    async def add_citation_attachment(
        self, document_id: int, document: Attachment | None
    ) -> int:
        if document_id in self._citations:
            return self._citations[document_id][0]

        display_index = len(self._citations) + 1
        self._citations[document_id] = (display_index, document)

        if document:
            document = document.model_copy()
            document.title = f"[{display_index}] {document.title or ''}".strip()
            document.reference_type = document.reference_type or document.type
            document.reference_url = document.reference_url or document.url
            await self.add_attachment(document)

        return display_index

    async def add_usage(self, usage: TokenUsage):
        self._response_state.add_usage(usage)

    async def set_discarded_messages(
        self, discarded_messages: DiscardedMessages | None
    ):
        self._response_state.discarded_messages = discarded_messages

    async def get_discarded_messages(self) -> DiscardedMessages | None:
        return self._response_state.discarded_messages

    async def create_function_tool_call(self, call: ToolCall) -> ToolUseMessage:
        tool_call = ToolUseMessage(
            call=self.choice.create_function_tool_call(
                id=call.id,
                name=call.function.name,
                arguments=call.function.arguments,
            ),
            snapshot=call.function.arguments,
        )
        self._tool_calls.append(tool_call)
        return tool_call

    async def create_function_call(self, call: FunctionCall) -> ToolUseMessage:
        tool_call = ToolUseMessage(
            call=self.choice.create_function_call(
                name=call.name,
                arguments=call.arguments,
            ),
            snapshot=call.arguments,
        )
        self._tool_calls.append(tool_call)
        return tool_call

    @property
    def has_function_call(self) -> bool:
        return self._choice is not None and self._choice.has_function_call
