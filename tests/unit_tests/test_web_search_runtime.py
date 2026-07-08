from types import SimpleNamespace
from typing import Any, cast

import pytest
from aidial_sdk.chat_completion import (
    Attachment,
    Choice,
    FunctionCall,
    Response,
    ToolCall,
)
from anthropic.types.beta import BetaServerToolUseBlock as ServerToolUseBlock
from anthropic.types.beta import (
    BetaWebSearchResultBlock as WebSearchResultBlock,
)
from anthropic.types.beta import (
    BetaWebSearchToolResultBlock as WebSearchToolResultBlock,
)
from anthropic.types.beta import (
    BetaWebSearchToolResultError as WebSearchToolResultError,
)

from aidial_adapter_anthropic.adapter import ValidationError
from aidial_adapter_anthropic.adapter._claude.adapter import Adapter
from aidial_adapter_anthropic.dial._lazy_stage import LazyStage
from aidial_adapter_anthropic.dial.consumer import Consumer, ToolUseMessage
from aidial_adapter_anthropic.dial.request import ModelParameters
from tests.utils.openai import user


class _ChoiceSpy:
    state: dict | None

    def __init__(self):
        self.state = None

    def set_state(self, state: dict):
        self.state = state


class _StageSpy:
    def __init__(self, title: str, sink: dict[str, list[str]]):
        self._title = title
        self._sink = sink

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    def append_content(self, content: str):
        self._sink.setdefault(self._title, []).append(content)


class _ConsumerSpy(Consumer):
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
        raise AssertionError("Not used in this test")

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
        raise AssertionError("Not used in this test")

    async def close_content(self, finish_reason=None):
        self.finish_reason = finish_reason

    async def add_usage(self, usage):
        self.usage = usage

    async def set_discarded_messages(self, discarded_messages):
        self.discarded_messages = discarded_messages

    async def get_discarded_messages(self):
        return self.discarded_messages

    async def create_function_tool_call(self, call: ToolCall) -> ToolUseMessage:
        raise AssertionError("Not used in this test")

    async def create_function_call(self, call: FunctionCall) -> ToolUseMessage:
        raise AssertionError("Not used in this test")

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


async def _invoke_non_streaming(
    adapter: Adapter, content: list[object]
) -> _ConsumerSpy:
    request = await adapter._prepare_claude_request(
        ModelParameters(),
        [user("hello")],
    )
    consumer = _ConsumerSpy()
    _mock_non_streaming_create(adapter, content)
    await adapter.invoke_non_streaming(consumer, None, request, None)
    return consumer


async def test_web_search_server_tool_creates_web_search_stage(
    adapter: Adapter,
):
    consumer = await _invoke_non_streaming(
        adapter,
        [
            ServerToolUseBlock(
                type="server_tool_use",
                id="srv_1",
                name="web_search",
                input={"query": "weather in NYC"},
            )
        ],
    )

    assert consumer.stages["Web Search"] == ["weather in NYC"]


async def test_web_search_result_creates_attachments(adapter: Adapter):
    consumer = await _invoke_non_streaming(
        adapter,
        [
            WebSearchToolResultBlock(
                type="web_search_tool_result",
                tool_use_id="srv_1",
                content=[
                    WebSearchResultBlock(
                        type="web_search_result",
                        title="Example",
                        url="https://example.com",
                        encrypted_content="ZW5jcnlwdGVk",
                        page_age="1 day ago",
                    )
                ],
            )
        ],
    )

    assert consumer.attachments == [
        Attachment(title="Example", url="https://example.com")
    ]


async def test_web_search_result_error_raises_validation_error(
    adapter: Adapter,
):
    with pytest.raises(
        ValidationError, match="Web search failed: max_uses_exceeded"
    ):
        await _invoke_non_streaming(
            adapter,
            [
                WebSearchToolResultBlock(
                    type="web_search_tool_result",
                    tool_use_id="srv_1",
                    content=WebSearchToolResultError(
                        type="web_search_tool_result_error",
                        error_code="max_uses_exceeded",
                    ),
                )
            ],
        )


async def test_server_tool_use_persists_state_without_thinking(
    adapter: Adapter,
):
    adapter.supports_thinking = False
    consumer = await _invoke_non_streaming(
        adapter,
        [
            ServerToolUseBlock(
                type="server_tool_use",
                id="srv_1",
                name="web_search",
                input={"query": "latest AI news"},
            )
        ],
    )

    assert consumer._choice.state is not None


async def test_plain_text_does_not_persist_state_without_thinking(
    adapter: Adapter,
):
    adapter.supports_thinking = False
    consumer = await _invoke_non_streaming(adapter, [])

    assert consumer._choice.state is None
