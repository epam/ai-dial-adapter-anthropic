"""Tests for the cache breakpoint the passthrough reports back to DIAL Core.

Like the rest of the passthrough tests, every test body runs against each
supported Anthropic backend (see ``anthropic_mocks.py``).
"""

import logging

import httpx
import pytest

from aidial_adapter_anthropic.passthrough import _caching as caching_module
from aidial_adapter_anthropic.passthrough import _proxy as proxy_module
from tests.unit_tests.anthropic_mocks import (
    BASE_MESSAGES_REQUEST,
    MESSAGES_REQUEST,
    AnthropicAPIMock,
    AnthropicMocker,
    read_fixture,
    split_sse_events,
)

_LOGGER_NAME = "aidial_adapter_anthropic.passthrough"

_DIAL_CACHE_BREAKPOINT_PATH = "x-dial-cache-breakpoint-path"
_DIAL_CACHE_EXPIRE_AT = "x-dial-cache-expire-at"

_EPHEMERAL = {"type": "ephemeral"}


def _text(text: str, **extra) -> dict:
    return {"type": "text", "text": text, **extra}


def _tool(name: str, **extra) -> dict:
    return {"name": name, "input_schema": {"type": "object"}, **extra}


def _messages_request(**overrides) -> dict:
    return {**MESSAGES_REQUEST, **overrides}


class TestCacheBreakpointReporting:
    """The X-DIAL-CACHE-* headers reported back for POST /v1/messages.

    The reported path addresses a block of the request body the way DIAL Core
    hashes it: tools, then system, then messages per content block. The last
    marked block wins, since it denotes the longest cached prefix.
    """

    @pytest.fixture(autouse=True)
    def mock_current_time_1000s(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(caching_module.time, "time", lambda: 1000)

    @pytest.mark.parametrize(
        "body, expected",
        [
            pytest.param(
                _messages_request(),
                None,
                id="no-breakpoint",
            ),
            pytest.param(
                _messages_request(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                _text("doc", cache_control=_EPHEMERAL),
                                _text("summarize it"),
                            ],
                        }
                    ]
                ),
                ("prefix.body.messages[0].content[0]", "1300"),
                id="content-block",
            ),
            pytest.param(
                _messages_request(
                    messages=[
                        {
                            "role": "user",
                            "content": [_text("doc", cache_control=_EPHEMERAL)],
                        },
                        {"role": "assistant", "content": "Sure."},
                        {
                            "role": "user",
                            "content": [
                                _text("more"),
                                _text("and now?", cache_control=_EPHEMERAL),
                            ],
                        },
                    ]
                ),
                ("prefix.body.messages[2].content[1]", "1300"),
                id="last-of-many-content-blocks",
            ),
            pytest.param(
                _messages_request(
                    system=[
                        _text("be helpful"),
                        _text("be brief", cache_control=_EPHEMERAL),
                    ]
                ),
                ("prefix.body.system[1]", "1300"),
                id="system-block",
            ),
            pytest.param(
                _messages_request(
                    tools=[
                        _tool("get_weather"),
                        _tool("get_time", cache_control=_EPHEMERAL),
                    ]
                ),
                ("prefix.body.tools[1]", "1300"),
                id="tool",
            ),
            pytest.param(
                _messages_request(
                    tools=[_tool("get_weather", cache_control=_EPHEMERAL)],
                    system=[_text("be helpful", cache_control=_EPHEMERAL)],
                    messages=[
                        {
                            "role": "user",
                            "content": [_text("hi", cache_control=_EPHEMERAL)],
                        }
                    ],
                ),
                ("prefix.body.messages[0].content[0]", "1300"),
                id="message-wins-over-system-and-tool",
            ),
            pytest.param(
                _messages_request(
                    tools=[_tool("get_weather", cache_control=_EPHEMERAL)],
                    system=[_text("be helpful", cache_control=_EPHEMERAL)],
                ),
                ("prefix.body.system[0]", "1300"),
                id="system-wins-over-tool",
            ),
            pytest.param(
                _messages_request(
                    system=[
                        _text(
                            "be helpful",
                            cache_control={**_EPHEMERAL, "ttl": "1h"},
                        )
                    ]
                ),
                ("prefix.body.system[0]", "4600"),
                id="explicit-ttl",
            ),
            pytest.param(
                _messages_request(
                    system=[
                        _text(
                            "be helpful",
                            cache_control={**_EPHEMERAL, "ttl": "bogus"},
                        )
                    ]
                ),
                ("prefix.body.system[0]", "1300"),
                id="malformed-ttl-falls-back-to-default",
            ),
            pytest.param(
                _messages_request(
                    tools=[
                        _tool(
                            "get_weather",
                            cache_control={**_EPHEMERAL, "ttl": "1h"},
                        )
                    ],
                    system=[
                        _text(
                            "be helpful",
                            cache_control={**_EPHEMERAL, "ttl": "5m"},
                        )
                    ],
                ),
                ("prefix.body.system[0]", "4600"),
                id="max-ttl-across-breakpoints",
            ),
            pytest.param(
                _messages_request(
                    cache_control=_EPHEMERAL,
                    system="be helpful",
                    messages=[
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "Hi there"},
                        {"role": "user", "content": "and now?"},
                    ],
                ),
                ("prefix.body.messages[2].content[0]", "1300"),
                id="automatic-caching-reports-longest-prefix",
            ),
            pytest.param(
                _messages_request(
                    cache_control={**_EPHEMERAL, "ttl": "1h"},
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                _text("doc", cache_control=_EPHEMERAL),
                                _text("summarize it"),
                            ],
                        }
                    ],
                ),
                ("prefix.body.messages[0].content[1]", "4600"),
                id="automatic-caching-wins-over-explicit-breakpoint",
            ),
            pytest.param(
                {**BASE_MESSAGES_REQUEST, "cache_control": _EPHEMERAL},
                ("prefix.body.messages[0].content[0]", "1300"),
                id="automatic-caching-with-scalar-content",
            ),
        ],
    )
    async def test_reported_breakpoint(
        self,
        mocker: AnthropicMocker,
        http_client: httpx.AsyncClient,
        body: dict,
        expected: tuple[str, str] | None,
    ):
        class _Mock(AnthropicAPIMock):
            def on_block_messages(self, request) -> bytes:
                return read_fixture("messages_non_streaming_response.json")

        mocker.mock(_Mock())

        response = await http_client.post("/v1/messages", json=body)

        assert response.status_code == 200
        reported = (
            response.headers.get(_DIAL_CACHE_BREAKPOINT_PATH),
            response.headers.get(_DIAL_CACHE_EXPIRE_AT),
        )
        assert reported == (expected or (None, None))

    async def test_streaming_response(
        self, mocker: AnthropicMocker, http_client: httpx.AsyncClient
    ):
        chunks = split_sse_events(
            read_fixture("messages_streaming_response.txt")
        )

        class _Mock(AnthropicAPIMock):
            def on_stream_messages(self, request) -> list[bytes]:
                return chunks

        mocker.mock(_Mock())

        # The headers travel on the response head, so a streaming response
        # reports them just like a blocking one does.
        response = await http_client.post(
            "/v1/messages",
            json=_messages_request(
                stream=True,
                system=[_text("be helpful", cache_control=_EPHEMERAL)],
            ),
        )

        assert response.status_code == 200
        assert (
            response.headers[_DIAL_CACHE_BREAKPOINT_PATH]
            == "prefix.body.system[0]"
        )
        assert response.headers[_DIAL_CACHE_EXPIRE_AT] == "1300"

    async def test_upstream_error_reports_nothing(
        self, mocker: AnthropicMocker, http_client: httpx.AsyncClient
    ):
        class _Mock(AnthropicAPIMock):
            def on_block_messages(self, request) -> httpx.Response:
                return httpx.Response(
                    429,
                    json={
                        "type": "error",
                        "error": {
                            "type": "rate_limit_error",
                            "message": "slow down",
                        },
                    },
                )

        mocker.mock(_Mock())

        response = await http_client.post(
            "/v1/messages",
            json=_messages_request(
                system=[_text("be helpful", cache_control=_EPHEMERAL)]
            ),
        )

        # A retriable failure sends the next attempt to another upstream, whose
        # provider cache is cold, so nothing may be reported as cached.
        assert response.status_code == 429
        assert _DIAL_CACHE_BREAKPOINT_PATH not in response.headers
        assert _DIAL_CACHE_EXPIRE_AT not in response.headers

    async def test_count_tokens_reports_nothing(
        self, mocker: AnthropicMocker, http_client: httpx.AsyncClient
    ):
        class _Mock(AnthropicAPIMock):
            def on_count_tokens(self, request) -> bytes:
                return read_fixture("count_tokens_response.json")

        mocker.mock(_Mock())

        # Token counting doesn't generate, so it must neither create nor consume
        # cache affinity.
        response = await http_client.post(
            "/v1/messages/count_tokens",
            json={
                **BASE_MESSAGES_REQUEST,
                "system": [_text("be helpful", cache_control=_EPHEMERAL)],
            },
        )

        assert _DIAL_CACHE_BREAKPOINT_PATH not in response.headers
        assert _DIAL_CACHE_EXPIRE_AT not in response.headers

    async def test_batches_report_nothing(
        self, mocker: AnthropicMocker, http_client: httpx.AsyncClient
    ):
        class _Mock(AnthropicAPIMock):
            def on_batches(self, request) -> bytes:
                return read_fixture("batches_response.json")

        mocker.mock(_Mock())

        response = await http_client.post(
            "/v1/messages/batches",
            json={
                "requests": [
                    {
                        "custom_id": "req-1",
                        "params": {
                            **MESSAGES_REQUEST,
                            "cache_control": _EPHEMERAL,
                        },
                    }
                ]
            },
        )

        assert _DIAL_CACHE_BREAKPOINT_PATH not in response.headers
        assert _DIAL_CACHE_EXPIRE_AT not in response.headers

    async def test_broken_cache_logic_does_not_fail_the_request(
        self,
        mocker: AnthropicMocker,
        http_client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        class _Mock(AnthropicAPIMock):
            def on_block_messages(self, request) -> bytes:
                return read_fixture("messages_non_streaming_response.json")

        mocker.mock(_Mock())

        def _boom(request) -> dict[str, str]:
            raise ValueError("cannot read the body")

        # The proxy imported the function by name, so the reference it calls is
        # the one to replace.
        monkeypatch.setattr(proxy_module, "get_cache_headers", _boom)

        with caplog.at_level(logging.ERROR, logger=_LOGGER_NAME):
            response = await http_client.post(
                "/v1/messages",
                json=_messages_request(
                    system=[_text("be helpful", cache_control=_EPHEMERAL)]
                ),
            )

        # The upstream has already answered; a failure to work out the cached
        # prefix costs cache hits, never the response.
        assert response.status_code == 200
        assert response.json()["type"] == "message"
        assert _DIAL_CACHE_BREAKPOINT_PATH not in response.headers
        assert _DIAL_CACHE_EXPIRE_AT not in response.headers

        errors = [r.getMessage() for r in caplog.records if r.exc_info]
        assert errors == ["Failed to compute the DIAL cache headers."]

    async def test_request_without_messages_is_relayed(
        self, mocker: AnthropicMocker, http_client: httpx.AsyncClient
    ):
        class _Mock(AnthropicAPIMock):
            def on_block_messages(self, request) -> bytes:
                return read_fixture("messages_non_streaming_response.json")

        mocker.mock(_Mock())

        # A body the cache logic can't read at all: `messages` is what the
        # prefix paths are built from, and it is missing.
        response = await http_client.post(
            "/v1/messages",
            json={"model": "claude-3-5-sonnet-20241022", "max_tokens": 1024},
        )

        assert response.status_code == 200
        assert _DIAL_CACHE_BREAKPOINT_PATH not in response.headers
