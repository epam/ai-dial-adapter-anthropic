import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self, TypeVar

import httpx
import respx
from anthropic import (
    AsyncAnthropic,
    AsyncAnthropicBedrock,
    AsyncAnthropicBedrockMantle,
    AsyncAnthropicFoundry,
    AsyncAnthropicVertex,
)

from aidial_adapter_anthropic.passthrough._proxy import AnthropicClient
from tests.utils.bedrock import AMAZON_SSE_CONTENT_TYPE, sse_to_event_stream

_JSON = "application/json"
_SSE = "text/event-stream"

# Fixed test credentials/coordinates shared by the backend mockers.
_REGION = "test-region"
_PROJECT = "test-project"
_RESOURCE = "test-resource"


def read_fixture(name: str) -> bytes:
    fixtures = Path(__file__).parent / "fixtures" / "passthrough"
    return (fixtures / name).read_bytes()


_Handler = TypeVar("_Handler", bound=Callable[..., object])


def not_implemented(handler: _Handler) -> _Handler:
    """Mark a base stub the concrete mock is expected to override."""
    handler.not_implemented = True  # type: ignore[attr-defined]
    return handler


def is_implemented(handler: Callable[..., object]) -> bool:
    """Whether `handler` overrides its `not_implemented`-marked base stub."""
    return not getattr(handler, "not_implemented", False)


# A handler may return raw bytes — served as a 200 with the endpoint's default
# content type — or a full httpx.Response to control status, headers and body.
_BlockResponse = bytes | httpx.Response
_StreamResponse = list[bytes] | httpx.Response


class AnthropicAPIMock:
    @not_implemented
    def on_block_messages(self, request: httpx.Request) -> _BlockResponse:
        raise NotImplementedError("on_block_messages not implemented")

    @not_implemented
    def on_stream_messages(self, request: httpx.Request) -> _StreamResponse:
        raise NotImplementedError("on_stream_messages not implemented")

    @not_implemented
    def on_count_tokens(self, request: httpx.Request) -> _BlockResponse:
        raise NotImplementedError("on_count_tokens not implemented")

    @not_implemented
    def on_batches(self, request: httpx.Request) -> _BlockResponse:
        raise NotImplementedError("on_batches not implemented")


@dataclass
class AnthropicMocker(ABC):
    id: ClassVar[str]
    # Not every backend exposes token counting and batching (the legacy Bedrock
    # runtime exposes neither, Vertex has no batches endpoint). When it doesn't,
    # the SDK raises before making any HTTP call and the proxy maps that to a 404.
    supports_count_tokens: ClassVar[bool] = True
    supports_batches: ClassVar[bool] = True
    # The legacy Bedrock runtime streams the AWS event stream format, which the
    # proxy decodes and re-encodes as SSE; every other backend relays SSE bytes
    # verbatim.
    reencodes_stream: ClassVar[bool] = False
    router: respx.MockRouter

    @classmethod
    @abstractmethod
    def create(cls) -> Self:
        raise NotImplementedError

    @abstractmethod
    def make_client(self) -> AnthropicClient:
        raise NotImplementedError

    @abstractmethod
    def mock(self, base: AnthropicAPIMock) -> None:
        raise NotImplementedError


def _responder(
    body: Callable[[httpx.Request], _BlockResponse], content_type: str
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        result = body(request)
        if isinstance(result, httpx.Response):
            return result
        return httpx.Response(
            200, content=result, headers={"Content-Type": content_type}
        )

    return handler


_StreamMock = Callable[[httpx.Request], _StreamResponse]


def _stream_responder(
    stream: _StreamMock, content_type: str
) -> Callable[[httpx.Request], httpx.Response]:
    """Serve the wire chunks one at a time as a genuine streaming response."""

    def handler(request: httpx.Request) -> httpx.Response:
        result = stream(request)
        if isinstance(result, httpx.Response):
            return result

        async def body() -> AsyncIterator[bytes]:
            for chunk in result:
                yield chunk

        return httpx.Response(
            200, content=body(), headers={"Content-Type": content_type}
        )

    return handler


def _event_stream_chunks(stream: _StreamMock) -> _StreamMock:
    """Transcode each SSE event chunk into one legacy Bedrock stream frame."""

    def transcode(request: httpx.Request) -> _StreamResponse:
        result = stream(request)
        if isinstance(result, httpx.Response):
            return result
        return [sse_to_event_stream([chunk]) for chunk in result]

    return transcode


def _messages_dispatch(
    base: AnthropicAPIMock,
) -> Callable[[httpx.Request], httpx.Response]:
    block = _responder(base.on_block_messages, _JSON)
    stream = _stream_responder(base.on_stream_messages, _SSE)

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content or b"{}")
        return stream(request) if payload.get("stream") else block(request)

    return handler


class AnthropicPlatformMocker(AnthropicMocker):
    id = "platform"

    @classmethod
    def create(cls) -> Self:
        return cls(
            router=respx.mock(base_url="https://api.anthropic.com"),
        )

    def make_client(self) -> AsyncAnthropic:
        return AsyncAnthropic(api_key="test-claude-api-key", max_retries=0)

    def mock(self, base: AnthropicAPIMock) -> None:
        if is_implemented(base.on_block_messages) or is_implemented(
            base.on_stream_messages
        ):
            self.router.post("/v1/messages").mock(
                side_effect=_messages_dispatch(base)
            )
        if is_implemented(base.on_count_tokens):
            self.router.post("/v1/messages/count_tokens").mock(
                side_effect=_responder(base.on_count_tokens, _JSON)
            )
        if is_implemented(base.on_batches):
            self.router.post("/v1/messages/batches").mock(
                side_effect=_responder(base.on_batches, _JSON)
            )


class AnthropicFoundryMocker(AnthropicMocker):
    id = "foundry"

    @classmethod
    def create(cls) -> Self:
        # Foundry keeps the native Anthropic URLs; it only adapts auth headers.
        return cls(
            router=respx.mock(
                base_url=f"https://{_RESOURCE}.services.ai.azure.com/anthropic",
            ),
        )

    def make_client(self) -> AsyncAnthropicFoundry:
        return AsyncAnthropicFoundry(
            resource=_RESOURCE, api_key="test-foundry-key", max_retries=0
        )

    def mock(self, base: AnthropicAPIMock) -> None:
        if is_implemented(base.on_block_messages) or is_implemented(
            base.on_stream_messages
        ):
            self.router.post("/v1/messages").mock(
                side_effect=_messages_dispatch(base)
            )
        if is_implemented(base.on_count_tokens):
            self.router.post("/v1/messages/count_tokens").mock(
                side_effect=_responder(base.on_count_tokens, _JSON)
            )
        if is_implemented(base.on_batches):
            self.router.post("/v1/messages/batches").mock(
                side_effect=_responder(base.on_batches, _JSON)
            )


class AnthropicMantleMocker(AnthropicMocker):
    id = "bedrock-mantle"

    @classmethod
    def create(cls) -> Self:
        # Mantle also keeps the native Anthropic URLs.
        return cls(
            router=respx.mock(
                base_url=f"https://bedrock-mantle.{_REGION}.api.aws/anthropic",
            ),
        )

    def make_client(self) -> AsyncAnthropicBedrockMantle:
        return AsyncAnthropicBedrockMantle(
            aws_region=_REGION,
            aws_access_key="test-access-key",
            aws_secret_key="test-secret-key",  # noqa: S106
            max_retries=0,
        )

    def mock(self, base: AnthropicAPIMock) -> None:
        if is_implemented(base.on_block_messages) or is_implemented(
            base.on_stream_messages
        ):
            self.router.post("/v1/messages").mock(
                side_effect=_messages_dispatch(base)
            )
        if is_implemented(base.on_count_tokens):
            self.router.post("/v1/messages/count_tokens").mock(
                side_effect=_responder(base.on_count_tokens, _JSON)
            )
        if is_implemented(base.on_batches):
            self.router.post("/v1/messages/batches").mock(
                side_effect=_responder(base.on_batches, _JSON)
            )


class AnthropicBedrockLegacyMocker(AnthropicMocker):
    id = "bedrock-legacy"
    supports_count_tokens = False
    supports_batches = False
    reencodes_stream = True

    @classmethod
    def create(cls) -> Self:
        return cls(
            router=respx.mock(
                base_url=f"https://bedrock-runtime.{_REGION}.amazonaws.com",
            ),
        )

    def make_client(self) -> AsyncAnthropicBedrock:
        return AsyncAnthropicBedrock(
            aws_region=_REGION,
            aws_access_key="test-access-key",
            aws_secret_key="test-secret-key",  # noqa: S106
            max_retries=0,
        )

    def mock(self, base: AnthropicAPIMock) -> None:
        # The legacy runtime rewrites messages to /model/{id}/invoke[-with-...]
        # and supports neither token counting nor batches.
        if is_implemented(base.on_block_messages):
            self.router.post(path__regex=r"/model/.+/invoke$").mock(
                side_effect=_responder(base.on_block_messages, _JSON)
            )
        if is_implemented(base.on_stream_messages):
            self.router.post(
                path__regex=r"/model/.+/invoke-with-response-stream$"
            ).mock(
                side_effect=_stream_responder(
                    _event_stream_chunks(base.on_stream_messages),
                    AMAZON_SSE_CONTENT_TYPE,
                )
            )


class AnthropicVertexMocker(AnthropicMocker):
    id = "vertex"
    supports_batches = False

    @classmethod
    def create(cls) -> Self:
        return cls(
            router=respx.mock(
                base_url=f"https://{_REGION}-aiplatform.googleapis.com/v1",
            ),
        )

    def make_client(self) -> AsyncAnthropicVertex:
        return AsyncAnthropicVertex(
            region=_REGION,
            project_id=_PROJECT,
            access_token="test-access-token",  # noqa: S106
            max_retries=0,
        )

    def mock(self, base: AnthropicAPIMock) -> None:
        # Vertex rewrites messages to …/models/{model}:rawPredict (and
        # :streamRawPredict), token counting to a count-tokens:rawPredict
        # endpoint, and does not support batches. The count-tokens route is
        # registered first so it wins over the generic :rawPredict match.
        if is_implemented(base.on_count_tokens):
            self.router.post(path__regex=r"count-tokens:rawPredict$").mock(
                side_effect=_responder(base.on_count_tokens, _JSON)
            )
        if is_implemented(base.on_stream_messages):
            self.router.post(path__regex=r":streamRawPredict$").mock(
                side_effect=_stream_responder(base.on_stream_messages, _SSE)
            )
        if is_implemented(base.on_block_messages):
            self.router.post(path__regex=r":rawPredict$").mock(
                side_effect=_responder(base.on_block_messages, _JSON)
            )


def get_mocker_types() -> list[type[AnthropicMocker]]:
    return [
        AnthropicPlatformMocker,
        AnthropicFoundryMocker,
        AnthropicMantleMocker,
        AnthropicBedrockLegacyMocker,
        AnthropicVertexMocker,
    ]
