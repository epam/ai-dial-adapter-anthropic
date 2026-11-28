"""Tests for the cache breakpoint the passthrough reports back to DIAL Core.

Like the rest of the passthrough tests, every test body runs against each
supported Anthropic backend (see ``anthropic_mocks.py``).
"""

import json
import logging
from typing import Any

import httpx
import pytest

from aidial_adapter_anthropic.passthrough import _caching as caching_module
from aidial_adapter_anthropic.passthrough import _proxy as proxy_module
from aidial_adapter_anthropic.passthrough._middleware import MessagesAPICloud
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


_ONE_HOUR = {"type": "ephemeral", "ttl": "1h"}
_FIRST_BLOCK = "prefix.body.messages[0].content[0]"


def _turns(first: object, *, last_control: dict | None = _EPHEMERAL) -> list:
    """A user turn, the assistant's answer and a new user turn: the shape of
    the request that continues the one which had its marker on the first."""
    last = {} if last_control is None else {"cache_control": last_control}
    return [
        {"role": "user", "content": first},
        {"role": "assistant", "content": [_text("answer")]},
        {"role": "user", "content": [_text("next", **last)]},
    ]


def _tool_turns(first: object) -> list:
    """Like `_turns`, with a marker on a block inside the tool result."""
    return [
        {"role": "user", "content": first},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "f", "input": {}}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [_text("out", cache_control=_EPHEMERAL)],
                },
                _text("next", cache_control=_EPHEMERAL),
            ],
        },
    ]


class TestCacheMarkerRestore:
    """Vertex AI routes a request to the region holding its cache only when the
    request carries cache_control in the same blocks as the one that wrote it.
    Unless RESTORE_CACHE_MARKER is switched off, the passthrough puts the
    marker back on the block DIAL Core names in X-DIAL-CACHE-BREAKPOINT-PATH:
    the last marker of the request this one continues.
    """

    @pytest.mark.parametrize(
        "body, path, env, restored_first",
        [
            pytest.param(
                _messages_request(messages=_turns([_text("doc")])),
                _FIRST_BLOCK,
                None,
                [_text("doc", cache_control=_EPHEMERAL)],
                id="on-by-default",
            ),
            pytest.param(
                _messages_request(messages=_turns("doc")),
                _FIRST_BLOCK,
                None,
                [_text("doc", cache_control=_EPHEMERAL)],
                id="string-content-becomes-a-block",
            ),
            pytest.param(
                _messages_request(
                    messages=_turns([_text("doc")], last_control=_ONE_HOUR)
                ),
                _FIRST_BLOCK,
                None,
                [_text("doc", cache_control=_ONE_HOUR)],
                id="copies-the-ttl-of-the-last-marker",
            ),
            pytest.param(
                _messages_request(
                    messages=[
                        {"role": "user", "content": [_text("doc")]},
                        {"role": "assistant", "content": [_text("answer")]},
                        {
                            "role": "user",
                            "content": [
                                _text("new", cache_control=_ONE_HOUR),
                                _text("next", cache_control=_EPHEMERAL),
                            ],
                        },
                    ]
                ),
                _FIRST_BLOCK,
                None,
                [_text("doc", cache_control=_ONE_HOUR)],
                id="longest-ttl-after-the-block-keeps-the-order",
            ),
            pytest.param(
                _messages_request(
                    system=[_text("sys", cache_control=_EPHEMERAL)],
                    messages=_turns([_text("doc")], last_control=None),
                ),
                _FIRST_BLOCK,
                None,
                None,
                id="markers-only-before-the-block",
            ),
            pytest.param(
                {
                    **_messages_request(
                        messages=_turns([_text("doc")], last_control=None)
                    ),
                    "cache_control": _EPHEMERAL,
                },
                _FIRST_BLOCK,
                None,
                [_text("doc", cache_control=_EPHEMERAL)],
                id="automatic-caching",
            ),
            pytest.param(
                _messages_request(messages=_turns([_text("doc")])),
                _FIRST_BLOCK,
                "true",
                [_text("doc", cache_control=_EPHEMERAL)],
                id="switched-on",
            ),
            pytest.param(
                _messages_request(messages=_turns([_text("doc")])),
                _FIRST_BLOCK,
                "false",
                None,
                id="switched-off",
            ),
            pytest.param(
                _messages_request(messages=_turns([_text("doc")])),
                None,
                None,
                None,
                id="no-pinned-path",
            ),
            pytest.param(
                _messages_request(
                    messages=_turns([_text("doc", cache_control=_ONE_HOUR)])
                ),
                _FIRST_BLOCK,
                None,
                None,
                id="already-marked",
            ),
            pytest.param(
                _messages_request(
                    messages=_turns([_text("doc")], last_control=None)
                ),
                _FIRST_BLOCK,
                None,
                None,
                id="request-asks-for-no-caching",
            ),
            pytest.param(
                _messages_request(
                    system=[
                        _text(f"s{i}", cache_control=_EPHEMERAL)
                        for i in range(3)
                    ],
                    messages=_turns([_text("doc")]),
                ),
                _FIRST_BLOCK,
                None,
                None,
                id="four-markers-already",
            ),
            pytest.param(
                _messages_request(
                    system=[
                        _text(f"s{i}", cache_control=_EPHEMERAL)
                        for i in range(2)
                    ],
                    messages=_tool_turns([_text("doc")]),
                ),
                _FIRST_BLOCK,
                None,
                None,
                id="four-markers-with-one-inside-a-tool-result",
            ),
            pytest.param(
                _messages_request(messages=_turns([_text("doc")])),
                "prefix.body.messages[7].content[0]",
                None,
                None,
                id="path-out-of-range",
            ),
            pytest.param(
                _messages_request(messages=_turns([_text("doc")])),
                "prefix.body.messages",
                None,
                None,
                id="unknown-path",
            ),
        ],
    )
    async def test_restored_marker(
        self,
        mocker: AnthropicMocker,
        http_client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        body: dict,
        path: str | None,
        env: str | None,
        restored_first: list | None,
    ):
        if env is None:
            monkeypatch.delenv("RESTORE_CACHE_MARKER", raising=False)
        else:
            monkeypatch.setenv("RESTORE_CACHE_MARKER", env)

        seen: list[dict] = []

        class _Mock(AnthropicAPIMock):
            def on_block_messages(self, request) -> bytes:
                seen.append(json.loads(request.content))
                return read_fixture("messages_non_streaming_response.json")

        mocker.mock(_Mock())

        response = await http_client.post(
            "/v1/messages",
            json=body,
            headers={} if path is None else {_DIAL_CACHE_BREAKPOINT_PATH: path},
        )

        assert response.status_code == 200
        expected_first = body["messages"][0]["content"]
        if restored_first is not None and mocker.cloud is MessagesAPICloud.GCP:
            expected_first = restored_first
        assert seen[0]["messages"][0]["content"] == expected_first
        assert seen[0]["messages"][1:] == body["messages"][1:]

    @pytest.mark.parametrize(
        "first, system, path, keys",
        [
            pytest.param(
                [{"text": "doc", "type": "text"}],
                None,
                _FIRST_BLOCK,
                ["cache_control", "text", "type"],
                id="message-block-as-core-forwards-it",
            ),
            pytest.param(
                "doc",
                None,
                _FIRST_BLOCK,
                ["cache_control", "text", "type"],
                id="string-content",
            ),
            pytest.param(
                "doc",
                [_text("sys")],
                "prefix.body.system[0]",
                ["type", "text", "cache_control"],
                id="block-in-client-order",
            ),
        ],
    )
    def test_key_order_of_restored_block(
        self, first: object, system: list | None, path: str, keys: list
    ):
        """Vertex AI matches the bytes of the request: the restored block must
        read as in the request that wrote the cache. DIAL Core forwards message
        blocks with sorted keys; other blocks keep the client's order."""
        body: Any = _messages_request(
            messages=_turns(first), **({"system": system} if system else {})
        )
        assert caching_module.restore_cache_marker(
            body, httpx.Headers({_DIAL_CACHE_BREAKPOINT_PATH: path})
        )
        section, index = (
            (body["system"], 0)
            if system
            else (body["messages"][0]["content"], 0)
        )
        assert list(section[index]) == keys
