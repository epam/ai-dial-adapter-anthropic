import anthropic
import pytest
from aidial_sdk.chat_completion import Response
from aidial_sdk.chat_completion.request import (
    ChatCompletionRequest,
    Request,
)
from pydantic import SecretStr
from starlette.requests import Request as StarletteRequest

from aidial_adapter_anthropic.adapter import ChatCompletionAdapter
from aidial_adapter_anthropic.adapter._claude import caching as caching_module
from aidial_adapter_anthropic.adapter._claude.adapter import Adapter
from aidial_adapter_anthropic.adapter._claude.tokenizer import (
    ApproximateTokenizer,
)
from aidial_adapter_anthropic.adapter.claude import create_adapter
from aidial_adapter_anthropic.dial.consumer import ChoiceConsumer
from aidial_adapter_anthropic.dial.request import ModelParameters

_DIAL_CACHE_BREAKPOINT_PATH = "X-DIAL-CACHE-BREAKPOINT-PATH"
_DIAL_CACHE_EXPIRE_AT = "X-DIAL-CACHE-EXPIRE-AT"


def _user(content: str, *, cache_breakpoint: dict | None = None) -> dict:
    msg: dict = {"role": "user", "content": content}
    if cache_breakpoint is not None:
        msg["custom_fields"] = {"cache_breakpoint": cache_breakpoint}
    return msg


def _request(request: dict) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(request)


def _response(request: ChatCompletionRequest) -> Response:
    return Response(
        Request(
            headers={},
            api_key_secret=SecretStr("test"),
            deployment_id="test-deployment",
            original_request=StarletteRequest(
                {"type": "http", "method": "POST", "path": "/", "headers": []}
            ),
            **request.model_dump(exclude_none=True),
        )
    )


@pytest.fixture
async def adapter() -> ChatCompletionAdapter:
    return await create_adapter(
        deployment="test-deployment",
        storage=None,
        client=anthropic.AsyncAnthropic(),
        custom_tokenizer=ApproximateTokenizer(),
        default_max_tokens=1024,
        supports_thinking=True,
        supports_documents=True,
    )


async def _invoke_chat(
    adapter: ChatCompletionAdapter,
    monkeypatch: pytest.MonkeyPatch,
    request: dict,
) -> ChoiceConsumer:
    req = _request(request)
    consumer = ChoiceConsumer(_response(req))

    async def _fake_chat(
        self: Adapter,
        consumer: ChoiceConsumer,
        params: ModelParameters,
        messages: list,
    ) -> None:
        return None

    monkeypatch.setattr(Adapter, "chat", _fake_chat)
    await adapter.chat(consumer, ModelParameters.create(req), req.messages)
    return consumer


async def test_adapter_chat_sets_headers_for_top_level_breakpoint(
    adapter: ChatCompletionAdapter,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(caching_module.time, "time", lambda: 1000)

    consumer = await _invoke_chat(
        adapter,
        monkeypatch,
        {
            "messages": [_user("first"), _user("second")],
            "custom_fields": {"cache_breakpoint": {"ttl": "1h"}},
        },
    )

    assert consumer.response.headers == [
        (_DIAL_CACHE_BREAKPOINT_PATH, "prefix.body.messages[1]"),
        (_DIAL_CACHE_EXPIRE_AT, "4600"),
    ]


async def test_adapter_chat_sets_headers_for_last_message_breakpoint(
    adapter: ChatCompletionAdapter,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(caching_module.time, "time", lambda: 1000)

    consumer = await _invoke_chat(
        adapter,
        monkeypatch,
        {
            "messages": [
                _user("first", cache_breakpoint={"ttl": "1h"}),
                _user("second"),
                _user("third", cache_breakpoint={"ttl": "5m"}),
            ]
        },
    )

    assert consumer.response.headers == [
        (_DIAL_CACHE_BREAKPOINT_PATH, "prefix.body.messages[2]"),
        (_DIAL_CACHE_EXPIRE_AT, "4600"),
    ]


async def test_adapter_chat_does_not_set_headers_without_breakpoints(
    adapter: ChatCompletionAdapter,
    monkeypatch: pytest.MonkeyPatch,
):
    consumer = await _invoke_chat(
        adapter,
        monkeypatch,
        {"messages": [_user("first"), _user("second")]},
    )

    assert consumer.response.headers == []
