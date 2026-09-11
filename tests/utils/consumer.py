from types import SimpleNamespace
from typing import Any, cast

from aidial_sdk.chat_completion import (
    Attachment,
    Choice,
    FunctionCall,
    Response,
    ToolCall,
)

from aidial_adapter_anthropic.adapter._claude.adapter import Adapter
from aidial_adapter_anthropic.dial._lazy_stage import LazyStage
from aidial_adapter_anthropic.dial.consumer import Consumer, ToolUseMessage
from aidial_adapter_anthropic.dial.request import ModelParameters
from tests.utils.openai import user


class _ChoiceSpy:
    state: dict | None

    def __init__(self):
        self.state = None
        self.reasoning_content: list[str] = []

    def set_state(self, state: dict):
        self.state = state

    def append_reasoning_content(self, content: str):
        self.reasoning_content.append(content)


class _StageSpy:
    def __init__(self, title: str, sink: dict[str, list[str]]):
        self._title = title
        self._sink = sink

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    def append_content(self, content: str):
        self._sink.setdefault(self._title, []).append(content)


class ConsumerSpy(Consumer):
    def __init__(self):
        self._choice = _ChoiceSpy()
        self.stages: dict[str, list[str]] = {}
        self.attachments: list[Attachment] = []
        self.content: list[str] = []
        self.usage = None
        self.discarded_messages = None
        self.finish_reason = None

    def fork(self) -> Consumer:
        return self

    @property
    def choice(self) -> Choice:
        return cast(Choice, self._choice)

    def get_response(self) -> Response:
        raise AssertionError("Not used in these tests")

    def create_stage(self, title: str) -> LazyStage:
        return cast(LazyStage, _StageSpy(title, self.stages))

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    async def append_content(self, content: str):
        self.content.append(content)

    async def add_attachment(self, attachment: Attachment):
        self.attachments.append(attachment)

    async def add_citation_attachment(
        self, document_id: int, document: Attachment | None
    ) -> int:
        raise AssertionError("Not used in these tests")

    async def close_content(self, finish_reason=None):
        self.finish_reason = finish_reason

    async def add_usage(self, usage):
        self.usage = usage

    async def set_discarded_messages(self, discarded_messages):
        self.discarded_messages = discarded_messages

    async def get_discarded_messages(self):
        return self.discarded_messages

    async def create_function_tool_call(self, call: ToolCall) -> ToolUseMessage:
        raise AssertionError("Not used in these tests")

    async def create_function_call(self, call: FunctionCall) -> ToolUseMessage:
        raise AssertionError("Not used in these tests")

    @property
    def has_function_call(self) -> bool:
        return False


def _mock_non_streaming_create(
    adapter: Adapter, content: list[object], stop_reason: str = "end_turn"
):
    message = SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=1,
            output_tokens=2,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            output_tokens_details=None,
        ),
    )

    async def _create(**kwargs):
        return message

    adapter.client = cast(
        Any,
        SimpleNamespace(
            beta=SimpleNamespace(messages=SimpleNamespace(create=_create))
        ),
    )


async def invoke_non_streaming(
    adapter: Adapter, content: list[object]
) -> ConsumerSpy:
    request = await adapter._prepare_claude_request(
        ModelParameters(),
        [user("hello")],
    )
    consumer = ConsumerSpy()
    _mock_non_streaming_create(adapter, content)
    await adapter.invoke_non_streaming(consumer, None, request, None)
    return consumer
