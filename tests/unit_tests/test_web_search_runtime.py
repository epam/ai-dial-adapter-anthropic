from types import SimpleNamespace
from typing import Any, cast

import anthropic
import pytest
from aidial_sdk.chat_completion import Attachment, FunctionCall, ToolCall
from anthropic.lib.streaming import (
    ParsedBetaContentBlockStopEvent as ParsedContentBlockStopEvent,
)
from anthropic.types.beta import (
    BetaServerToolUseBlock as ServerToolUseBlock,
)
from anthropic.types.beta import BetaTextBlock as TextBlock
from anthropic.types.beta import BetaUsage as Usage
from anthropic.types.beta import (
    BetaWebSearchToolResultBlock as WebSearchToolResultBlock,
)

from aidial_adapter_anthropic.adapter import ValidationError
from aidial_adapter_anthropic.adapter._claude import adapter as adapter_module
from aidial_adapter_anthropic.adapter._claude.adapter import Adapter
from aidial_adapter_anthropic.adapter._claude.tokenizer import (
    ApproximateTokenizer,
)
from aidial_adapter_anthropic.dial.consumer import Consumer, ToolUseMessage
from aidial_adapter_anthropic.dial.request import ModelParameters
from tests.utils.openai import user

_WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
}


class _CapturedCall:
    def __init__(self, name: str, arguments: str | None):
        self.name = name
        self.arguments = arguments or ""

    def append_arguments(self, arguments: str) -> "_CapturedCall":
        self.arguments += arguments
        return self


class _FakeStage:
    def __init__(self, title: str, owner: "_FakeConsumer"):
        self.title = title
        self._owner = owner
        self._parts: list[str] = []

    def __enter__(self) -> "_FakeStage":
        return self

    def __exit__(self, *exc: Any) -> None:
        self._flush()

    async def __aenter__(self) -> "_FakeStage":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self._flush()

    def append_content(self, text: str) -> None:
        self._parts.append(text)

    def _flush(self) -> None:
        self._owner.stages.append((self.title, "".join(self._parts)))


class _FakeChoice:
    def __init__(self):
        self.states: list[dict] = []

    def set_state(self, state: dict) -> None:
        self.states.append(state)


class _FakeConsumer:
    def __init__(self):
        self.content: list[str] = []
        self.attachments: list[Attachment] = []
        self.stages: list[tuple[str, str]] = []
        self.discarded_messages = None
        self.usage = None
        self.finish_reason = None
        self._choice = _FakeChoice()
        self._tool_calls: list[_CapturedCall] = []

    def fork(self) -> "_FakeConsumer":
        return self

    @property
    def choice(self) -> _FakeChoice:
        return self._choice

    async def close_content(self, finish_reason=None):
        self.finish_reason = finish_reason

    def get_response(self):
        return None

    async def append_content(self, content: str):
        self.content.append(content)

    async def add_attachment(self, attachment: Attachment):
        self.attachments.append(attachment)

    async def add_citation_attachment(self, document_id: int, document):
        return 1

    async def add_usage(self, usage):
        self.usage = usage

    async def set_discarded_messages(self, discarded_messages):
        self.discarded_messages = discarded_messages

    async def get_discarded_messages(self):
        return self.discarded_messages

    async def create_function_tool_call(self, call: ToolCall) -> ToolUseMessage:
        captured = _CapturedCall(call.function.name, call.function.arguments)
        self._tool_calls.append(captured)
        return ToolUseMessage(call=captured, snapshot=captured.arguments)

    async def create_function_call(self, call: FunctionCall) -> ToolUseMessage:
        captured = _CapturedCall(call.name, call.arguments)
        self._tool_calls.append(captured)
        return ToolUseMessage(call=captured, snapshot=captured.arguments)

    @property
    def has_function_call(self) -> bool:
        return False

    def create_stage(self, title: str) -> _FakeStage:
        return _FakeStage(title, self)

    async def __aenter__(self) -> "_FakeConsumer":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False


def _build_adapter(*, supports_thinking: bool) -> Adapter:
    return Adapter(
        deployment="test-deployment",
        storage=None,
        client=anthropic.AsyncAnthropic(),
        tokenizer=ApproximateTokenizer(),
        default_max_tokens=1024,
        supports_thinking=supports_thinking,
        supports_documents=True,
    )


def _ws_use_block() -> ServerToolUseBlock:
    return ServerToolUseBlock.model_validate(
        {
            "type": "server_tool_use",
            "id": "wsu_1",
            "name": "web_search",
            "input": {"query": "latest weather nyc"},
        }
    )


def _ws_result_block() -> WebSearchToolResultBlock:
    return WebSearchToolResultBlock.model_validate(
        {
            "type": "web_search_tool_result",
            "tool_use_id": "wsu_1",
            "content": [
                {
                    "type": "web_search_result",
                    "title": "Weather Source",
                    "url": "https://example.com/weather",
                    "encrypted_content": "encrypted",
                }
            ],
        }
    )


def _ws_error_block() -> WebSearchToolResultBlock:
    return WebSearchToolResultBlock.model_validate(
        {
            "type": "web_search_tool_result",
            "tool_use_id": "wsu_1",
            "content": {
                "type": "web_search_tool_result_error",
                "error_code": "too_many_requests",
            },
        }
    )


def _text_block() -> TextBlock:
    return TextBlock.model_validate({"type": "text", "text": "Final answer"})


async def _prepare_request(adapter: Adapter):
    return await adapter._prepare_claude_request(
        ModelParameters(configuration={"web_search": _WEB_SEARCH_TOOL}),
        [user("Weather?")],
    )


async def test_non_streaming_web_search_stage_attachment_and_state(monkeypatch):
    adapter = _build_adapter(supports_thinking=False)
    request = await _prepare_request(adapter)
    consumer = _FakeConsumer()

    message = SimpleNamespace(
        content=[_ws_use_block(), _ws_result_block(), _text_block()],
        stop_reason="end_turn",
        usage=Usage.model_validate({"input_tokens": 5, "output_tokens": 7}),
    )

    async def _create(**_kwargs):
        return message

    monkeypatch.setattr(
        adapter.client.beta.messages,
        "create",
        _create,
    )

    await adapter.invoke_non_streaming(
        consumer=cast(Consumer, consumer),
        tools_mode=None,
        request=request,
        discarded_messages=None,
    )

    assert ("Web Search", "latest weather nyc") in consumer.stages
    assert consumer.content == ["Final answer"]
    assert len(consumer.attachments) == 1
    assert consumer.attachments[0].title == "Weather Source"
    assert consumer.attachments[0].url == "https://example.com/weather"
    assert len(consumer.choice.states) == 1
    assert "claude_message_content" in consumer.choice.states[0]


async def test_non_streaming_web_search_error_raises(monkeypatch):
    adapter = _build_adapter(supports_thinking=False)
    request = await _prepare_request(adapter)
    consumer = _FakeConsumer()

    message = SimpleNamespace(
        content=[_ws_error_block()],
        stop_reason="end_turn",
        usage=Usage.model_validate({"input_tokens": 1, "output_tokens": 1}),
    )

    async def _create(**_kwargs):
        return message

    monkeypatch.setattr(
        adapter.client.beta.messages,
        "create",
        _create,
    )

    with pytest.raises(ValidationError, match="Web search failed"):
        await adapter.invoke_non_streaming(
            consumer=cast(Consumer, consumer),
            tools_mode=None,
            request=request,
            discarded_messages=None,
        )


class _FakeStream:
    def __init__(self, events: list[Any]):
        self._events = events

    async def __aenter__(self) -> "_FakeStream":
        return self

    async def __aexit__(self, *_exc: Any) -> bool:
        return False

    def __aiter__(self):
        async def _gen():
            for event in self._events:
                yield event

        return _gen()


async def test_streaming_web_search_stage_attachment_and_state(monkeypatch):
    adapter = _build_adapter(supports_thinking=False)
    request = await _prepare_request(adapter)
    consumer = _FakeConsumer()

    events = [
        ParsedContentBlockStopEvent(
            index=0,
            type="content_block_stop",
            content_block=_ws_use_block(),
        ),
        ParsedContentBlockStopEvent(
            index=1,
            type="content_block_stop",
            content_block=_ws_result_block(),
        ),
    ]

    class _PatchedAsyncMessagesAdapter:
        def __init__(self, _api):
            pass

        def stream(self, **_kwargs):
            return _FakeStream(events)

    monkeypatch.setattr(
        adapter_module, "_AsyncMessagesAdapter", _PatchedAsyncMessagesAdapter
    )

    await adapter.invoke_streaming(
        consumer=cast(Consumer, consumer),
        tools_mode=None,
        request=request,
        discarded_messages=None,
    )

    assert ("Web Search", "latest weather nyc") in consumer.stages
    assert len(consumer.attachments) == 1
    assert consumer.attachments[0].url == "https://example.com/weather"
    assert len(consumer.choice.states) == 1
    assert "web_search_content" in consumer.choice.states[0]


async def test_streaming_web_search_error_raises(monkeypatch):
    adapter = _build_adapter(supports_thinking=False)
    request = await _prepare_request(adapter)
    consumer = _FakeConsumer()

    events = [
        ParsedContentBlockStopEvent(
            index=0,
            type="content_block_stop",
            content_block=_ws_error_block(),
        )
    ]

    class _PatchedAsyncMessagesAdapter:
        def __init__(self, _api):
            pass

        def stream(self, **_kwargs):
            return _FakeStream(events)

    monkeypatch.setattr(
        adapter_module, "_AsyncMessagesAdapter", _PatchedAsyncMessagesAdapter
    )

    with pytest.raises(ValidationError, match="Web search failed"):
        await adapter.invoke_streaming(
            consumer=cast(Consumer, consumer),
            tools_mode=None,
            request=request,
            discarded_messages=None,
        )
