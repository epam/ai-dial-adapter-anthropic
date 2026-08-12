"""Passthrough tests parametrized over every supported Anthropic backend.

Each backend contributes an :class:`AnthropicMocker` (see ``anthropic_mocks.py``)
and the same test body runs against all of them.
"""

import gzip
import logging
from collections.abc import Awaitable, Callable

import anthropic
import httpx
import pytest
from anthropic.types.messages import MessageBatch

from aidial_adapter_anthropic.passthrough import create_anthropic_api_app
from tests.unit_tests.anthropic_mocks import (
    BASE_MESSAGES_REQUEST,
    MESSAGES_REQUEST,
    AnthropicAPIMock,
    AnthropicMocker,
    asgi_client,
    read_fixture,
    split_sse_events,
)

_LOGGER_NAME = "aidial_adapter_anthropic.passthrough"


@pytest.fixture
async def anthropic_client(mocker: AnthropicMocker):
    app = create_anthropic_api_app(mocker.make_client())
    async with anthropic.AsyncAnthropic(
        api_key="test-dial-api-key",
        http_client=asgi_client(app),
        max_retries=0,
    ) as client:
        yield client


async def _assert_unsupported(run: Callable[[], Awaitable[object]]) -> None:
    with pytest.raises(anthropic.NotFoundError) as excinfo:
        await run()
    assert excinfo.value.status_code == 404
    assert "not supported" in excinfo.value.message


class TestMessagesNonStreaming:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker: AnthropicMocker):
        class _Mock(AnthropicAPIMock):
            def on_block_messages(self, request) -> bytes:
                return read_fixture("messages_non_streaming_response.json")

        mocker.mock(_Mock())

    async def test_http_client(self, http_client: httpx.AsyncClient):
        response = await http_client.post("/v1/messages", json=MESSAGES_REQUEST)

        assert response.status_code == 200
        body = response.json()
        assert body["type"] == "message"
        assert body["role"] == "assistant"
        assert (
            body["content"][0]["text"] == "Hello! How can I assist you today?"
        )

    async def test_anthropic_client(
        self, anthropic_client: anthropic.AsyncAnthropic
    ):
        response = await anthropic_client.messages.create(**MESSAGES_REQUEST)

        assert response.type == "message"
        assert response.role == "assistant"
        text_block = response.content[0]
        assert isinstance(text_block, anthropic.types.TextBlock)
        assert text_block.text == "Hello! How can I assist you today?"


class TestMessagesStreaming:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker: AnthropicMocker):
        chunks = split_sse_events(
            read_fixture("messages_streaming_response.txt")
        )

        class _Mock(AnthropicAPIMock):
            def on_stream_messages(self, request) -> list[bytes]:
                return chunks

        mocker.mock(_Mock())

    async def test_http_client(self, http_client: httpx.AsyncClient):
        response = await http_client.post(
            "/v1/messages", json={**MESSAGES_REQUEST, "stream": True}
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        # The legacy Bedrock backend decodes its AWS event stream and re-encodes
        # it back to SSE, inferring each event's name from the frame's "type";
        # every other backend relays SSE verbatim. Either way the same event
        # names and text must appear, so only assert on that common output.
        body = response.content
        assert b"event: message_start" in body
        assert b"event: content_block_delta" in body
        assert b"event: message_stop" in body
        for text in (b"Hello!", b" How can I", b" assist you today?"):
            assert text in body

    async def test_anthropic_client(
        self, anthropic_client: anthropic.AsyncAnthropic
    ):
        async with anthropic_client.messages.stream(
            **MESSAGES_REQUEST
        ) as stream:
            text = await stream.get_final_text()

        assert text == "Hello! How can I assist you today?"

    async def test_http_client_relays_bytes(
        self, mocker: AnthropicMocker, http_client: httpx.AsyncClient
    ):
        chunks = split_sse_events(
            read_fixture("messages_streaming_response.txt")
        )
        response = await http_client.post(
            "/v1/messages", json={**MESSAGES_REQUEST, "stream": True}
        )

        assert response.status_code == 200
        if mocker.reencodes_stream:
            # Legacy Bedrock decodes the AWS event stream and re-encodes it as
            # SSE, so the bytes are transformed rather than relayed verbatim.
            assert response.content != b"".join(chunks)
            assert b"event: message_start" in response.content
        else:
            # Every other backend already speaks SSE, so bytes pass through
            # unchanged.
            assert response.content == b"".join(chunks)


class TestCountTokens:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker: AnthropicMocker):
        class _Mock(AnthropicAPIMock):
            def on_count_tokens(self, request) -> bytes:
                return read_fixture("count_tokens_response.json")

        mocker.mock(_Mock())

    async def test_http_client(
        self, mocker: AnthropicMocker, http_client: httpx.AsyncClient
    ):
        async def _run() -> httpx.Response:
            return await http_client.post(
                "/v1/messages/count_tokens", json=BASE_MESSAGES_REQUEST
            )

        response = await _run()
        if mocker.supports_count_tokens:
            assert response.status_code == 200
            assert response.json() == {"input_tokens": 14}
        else:
            assert response.status_code == 404
            assert response.json() == {
                "type": "error",
                "error": {
                    "type": "not_found_error",
                    "message": "Endpoint not supported",
                },
            }

    async def test_anthropic_client(
        self,
        mocker: AnthropicMocker,
        anthropic_client: anthropic.AsyncAnthropic,
    ):
        async def _run() -> anthropic.types.MessageTokensCount:
            return await anthropic_client.messages.count_tokens(
                **BASE_MESSAGES_REQUEST
            )

        if mocker.supports_count_tokens:
            response = await _run()
            assert response.input_tokens == 14
        else:
            await _assert_unsupported(_run)


class TestMessageBatches:
    _BATCHES_REQUEST = {
        "requests": [
            {"custom_id": "req-1", "params": MESSAGES_REQUEST},
        ],
    }

    @pytest.fixture(autouse=True)
    def _setup(self, mocker: AnthropicMocker):
        class _Mock(AnthropicAPIMock):
            def on_batches(self, request) -> bytes:
                return read_fixture("batches_response.json")

        mocker.mock(_Mock())

    async def test_http_client(
        self, mocker: AnthropicMocker, http_client: httpx.AsyncClient
    ):
        async def _run() -> httpx.Response:
            return await http_client.post(
                "/v1/messages/batches", json=self._BATCHES_REQUEST
            )

        response = await _run()
        if mocker.supports_batches:
            assert response.status_code == 200
            body = response.json()
            assert body["type"] == "message_batch"
            assert body["id"] == "msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d"
            assert body["processing_status"] == "in_progress"
        else:
            assert response.status_code == 404
            assert response.json() == {
                "type": "error",
                "error": {
                    "type": "not_found_error",
                    "message": "Endpoint not supported",
                },
            }

    async def test_anthropic_client(
        self,
        mocker: AnthropicMocker,
        anthropic_client: anthropic.AsyncAnthropic,
    ):
        async def _run() -> MessageBatch:
            return await anthropic_client.messages.batches.create(
                requests=[
                    {"custom_id": "req-1", "params": MESSAGES_REQUEST}  # type: ignore[typeddict-item]
                ]
            )

        if mocker.supports_batches:
            batch = await _run()
            assert batch.type == "message_batch"
            assert batch.id == "msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d"
            assert batch.processing_status == "in_progress"
        else:
            await _assert_unsupported(_run)


class TestRequestHeaderPassthrough:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker: AnthropicMocker):
        class _Mock(AnthropicAPIMock):
            def on_block_messages(self, request) -> bytes:
                return read_fixture("messages_non_streaming_response.json")

        mocker.mock(_Mock())

    async def test_http_client(
        self, mocker: AnthropicMocker, http_client: httpx.AsyncClient
    ):
        response = await http_client.post(
            "/v1/messages",
            json=MESSAGES_REQUEST,
            headers={
                "anthropic-beta": "token-efficient-tools-2025-02-19",
                "Accept-Encoding": "identity",
                "x-not-forwarded": "should-be-dropped",
                "Authorization": "Bearer downstream-secret",
            },
        )

        assert response.status_code == 200
        upstream_headers = mocker.router.calls.last.request.headers
        # Anthropic-specific headers must reach the upstream.
        assert (
            upstream_headers["anthropic-beta"]
            == "token-efficient-tools-2025-02-19"
        )
        # The encoding-control header must reach the upstream too.
        assert upstream_headers["accept-encoding"] == "identity"
        # Unrelated headers must not be forwarded.
        assert "x-not-forwarded" not in upstream_headers
        # The downstream credentials must not leak upstream: the backend authns
        # with its own client-supplied credentials, never the caller's header.
        assert (
            upstream_headers.get("authorization") != "Bearer downstream-secret"
        )


class TestAnthropicBetaAdaptation:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker: AnthropicMocker):
        class _Mock(AnthropicAPIMock):
            def on_block_messages(self, request) -> bytes:
                return read_fixture("messages_non_streaming_response.json")

        mocker.mock(_Mock())

    async def test_default_forwards_header_untouched(
        self, mocker: AnthropicMocker, http_client: httpx.AsyncClient
    ):
        # Without a custom on_anthropic_beta_header the package performs no
        # feature adaptation: the header reaches every backend verbatim.
        beta_header = (
            "oauth-2025-04-20,"
            "token-efficient-tools-2025-02-19,"
            "thinking-token-count-2026-05-13"
        )

        response = await http_client.post(
            "/v1/messages",
            json=MESSAGES_REQUEST,
            headers={"anthropic-beta": beta_header},
        )

        assert response.status_code == 200
        upstream_headers = mocker.router.calls.last.request.headers
        assert upstream_headers["anthropic-beta"] == beta_header

    async def test_custom_handler_rewrites_features(
        self, mocker: AnthropicMocker
    ):
        # The handler receives the upstream client and the parsed feature list,
        # and the list it returns is forwarded (here: one flag dropped).
        seen: dict[str, object] = {}

        def on_beta(client, features: list[str]) -> list[str]:
            seen["client"] = client
            seen["features"] = list(features)
            return [f for f in features if f != "drop-me"]

        client = mocker.make_client()
        app = create_anthropic_api_app(client, on_anthropic_beta_header=on_beta)
        async with asgi_client(app) as http_client:
            response = await http_client.post(
                "/v1/messages",
                json=MESSAGES_REQUEST,
                headers={"anthropic-beta": "keep-me,drop-me"},
            )

        assert response.status_code == 200
        assert seen["client"] is client
        assert seen["features"] == ["keep-me", "drop-me"]
        upstream_headers = mocker.router.calls.last.request.headers
        assert upstream_headers["anthropic-beta"] == "keep-me"

    async def test_custom_handler_dropping_all_removes_header(
        self, mocker: AnthropicMocker
    ):
        # An empty result drops the header entirely rather than forwarding an
        # empty anthropic-beta (which the upstream would reject).
        app = create_anthropic_api_app(
            mocker.make_client(),
            on_anthropic_beta_header=lambda client, features: [],
        )
        async with asgi_client(app) as http_client:
            response = await http_client.post(
                "/v1/messages",
                json=MESSAGES_REQUEST,
                headers={"anthropic-beta": "oauth-2025-04-20"},
            )

        assert response.status_code == 200
        upstream_headers = mocker.router.calls.last.request.headers
        assert "anthropic-beta" not in upstream_headers


class TestErrorPassthrough:
    _ERROR_BODY = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "boom"},
    }

    @pytest.fixture(autouse=True)
    def _setup(self, mocker: AnthropicMocker):
        error_body = self._ERROR_BODY

        class _Mock(AnthropicAPIMock):
            def on_block_messages(self, request) -> httpx.Response:
                return httpx.Response(
                    400,
                    json=error_body,
                    headers={"Retry-After": "13"},
                )

        mocker.mock(_Mock())

    async def test_http_client(self, http_client: httpx.AsyncClient):
        response = await http_client.post("/v1/messages", json=MESSAGES_REQUEST)

        # The upstream status, body and rate-limit headers must reach the
        # downstream client unchanged.
        assert response.status_code == 400
        assert response.json() == self._ERROR_BODY
        assert response.headers["Retry-After"] == "13"


class TestUnexpectedError:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker: AnthropicMocker):
        class _Mock(AnthropicAPIMock):
            def on_block_messages(self, request) -> httpx.Response:
                raise RuntimeError("upstream blew up")

        mocker.mock(_Mock())

    async def test_http_client(self, http_client: httpx.AsyncClient):
        response = await http_client.post("/v1/messages", json=MESSAGES_REQUEST)

        # An unexpected failure is reported in the Anthropic error schema as a
        # generic api_error, not the DIAL/OpenAI error shape. The SDK surfaces
        # the mock's blow-up as a connection error.
        assert response.status_code == 500
        assert response.json() == {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "Internal server error",
            },
        }


class TestResponseEncodingStripped:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker: AnthropicMocker):
        content = read_fixture("messages_non_streaming_response.json")

        class _Mock(AnthropicAPIMock):
            def on_block_messages(self, request) -> httpx.Response:
                return httpx.Response(
                    200,
                    content=gzip.compress(content),
                    headers={
                        "Content-Type": "application/json",
                        "Content-Encoding": "gzip",
                    },
                )

        mocker.mock(_Mock())

    async def test_http_client(self, http_client: httpx.AsyncClient):
        response = await http_client.post(
            "/v1/messages",
            json=MESSAGES_REQUEST,
            headers={"Accept-Encoding": "gzip"},
        )

        # The upstream response is decompressed before relaying, so the stale
        # content-encoding header must not leak to the downstream client.
        assert response.status_code == 200
        assert "content-encoding" not in response.headers
        assert response.json()["type"] == "message"


class TestDebugLogging:
    # The proxy logs the request/response only when DEBUG logging is enabled.
    @pytest.mark.parametrize(
        "level, expected", [(logging.DEBUG, True), (logging.INFO, False)]
    )
    async def test_request_and_response_logging(
        self,
        mocker: AnthropicMocker,
        http_client: httpx.AsyncClient,
        caplog: pytest.LogCaptureFixture,
        level: int,
        expected: bool,
    ):
        class _Mock(AnthropicAPIMock):
            def on_block_messages(self, request) -> bytes:
                return read_fixture("messages_non_streaming_response.json")

        mocker.mock(_Mock())

        with caplog.at_level(level, logger=_LOGGER_NAME):
            response = await http_client.post(
                "/v1/messages", json=MESSAGES_REQUEST
            )

        assert response.status_code == 200

        messages = [record.getMessage() for record in caplog.records]
        request_logs = [m for m in messages if m.startswith("request: ")]
        response_logs = [m for m in messages if m.startswith("response: ")]

        if expected:
            assert len(request_logs) == 1
            assert "Say hello." in request_logs[0]

            assert len(response_logs) == 1
            assert "Hello! How can I assist you today?" in response_logs[0]
        else:
            assert not request_logs
            assert not response_logs

    async def test_streaming_response_chunk_logging(
        self,
        mocker: AnthropicMocker,
        http_client: httpx.AsyncClient,
        caplog: pytest.LogCaptureFixture,
    ):
        chunks = split_sse_events(
            read_fixture("messages_streaming_response.txt")
        )

        class _Mock(AnthropicAPIMock):
            def on_stream_messages(self, request) -> list[bytes]:
                return chunks

        mocker.mock(_Mock())

        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            response = await http_client.post(
                "/v1/messages", json={**MESSAGES_REQUEST, "stream": True}
            )

        assert response.status_code == 200

        messages = [record.getMessage() for record in caplog.records]
        chunk_logs = [m for m in messages if m.startswith("response chunk: ")]

        # The streamed chunks are logged as-is; the number of chunks isn't
        # deterministic, so we only assert the streamed content is logged.
        assert chunk_logs
        streamed = "\n".join(chunk_logs)
        assert "message_start" in streamed
        assert "Hello!" in streamed
        assert "message_stop" in streamed

    @pytest.mark.parametrize(
        "level, expected_encoding",
        [(logging.DEBUG, "identity"), (logging.INFO, None)],
    )
    async def test_debug_disables_upstream_compression(
        self,
        mocker: AnthropicMocker,
        http_client: httpx.AsyncClient,
        caplog: pytest.LogCaptureFixture,
        level: int,
        expected_encoding: str | None,
    ):
        # With debug logging on, the upstream is asked not to compress the
        # response (Accept-Encoding: identity) so it can be logged as-is.
        # Otherwise the client's default (compressed) negotiation is left alone.
        captured: dict[str, str | None] = {}

        class _Mock(AnthropicAPIMock):
            def on_block_messages(self, request) -> bytes:
                captured["accept-encoding"] = request.headers.get(
                    "accept-encoding"
                )
                return read_fixture("messages_non_streaming_response.json")

        mocker.mock(_Mock())

        with caplog.at_level(level, logger=_LOGGER_NAME):
            response = await http_client.post(
                "/v1/messages", json=MESSAGES_REQUEST
            )

        assert response.status_code == 200
        if expected_encoding is None:
            assert captured["accept-encoding"] != "identity"
        else:
            assert captured["accept-encoding"] == expected_encoding
