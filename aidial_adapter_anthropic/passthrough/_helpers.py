"""Low-level helpers for the Anthropic API passthrough.

These functions perform the mechanical manipulations required to forward a
request to an upstream Anthropic client and relay its response back: header
filtering, SSE (re)formatting, response-body decoding and the Bedrock event
stream conversion.
"""

import functools
import json
from collections.abc import AsyncIterator, Callable
from enum import Enum
from json.scanner import make_scanner
from typing import Any, Literal, TypeGuard, assert_never, cast, overload

import httpx
from anthropic._streaming import ServerSentEvent
from anthropic.types.beta.message_count_tokens_params import (
    MessageCountTokensParams,
)
from anthropic.types.beta.message_create_params import MessageCreateParamsBase
from anthropic.types.beta.messages.batch_create_params import BatchCreateParams
from starlette.datastructures import Headers as StarletteHeaders

BETA_HEADER = "anthropic-beta"


class MessagesAPIEndpoint(Enum):
    POST_MESSAGES = "postMessages"
    POST_BATCHES = "postBatches"
    POST_COUNT_TOKENS = "postCountTokens"

    def to_endpoints(self) -> tuple[str, str]:
        match self:
            case MessagesAPIEndpoint.POST_MESSAGES:
                return "POST", "/v1/messages"
            case MessagesAPIEndpoint.POST_BATCHES:
                return "POST", "/v1/messages/batches"
            case MessagesAPIEndpoint.POST_COUNT_TOKENS:
                return "POST", "/v1/messages/count_tokens"
            case _:
                assert_never(self)


@overload
def typecast_request_body(
    body: Any, endpoint: Literal[MessagesAPIEndpoint.POST_MESSAGES]
) -> TypeGuard[MessageCreateParamsBase]: ...


@overload
def typecast_request_body(
    body: Any, endpoint: Literal[MessagesAPIEndpoint.POST_BATCHES]
) -> TypeGuard[BatchCreateParams]: ...


@overload
def typecast_request_body(
    body: Any, endpoint: Literal[MessagesAPIEndpoint.POST_COUNT_TOKENS]
) -> TypeGuard[MessageCountTokensParams]: ...


def typecast_request_body(body: Any, endpoint: MessagesAPIEndpoint) -> bool:
    return isinstance(body, dict)


# The scanner `json.loads` parses with: a JSON value at an index of a string.
_scan_once: Callable[[str, int], tuple[Any, int]] = make_scanner(
    cast(Any, json.JSONDecoder())
)


def _skip_ws(s: str, idx: int) -> int:
    # What `json` itself takes for whitespace.
    while s.startswith((" ", "\t", "\n", "\r"), idx):
        idx += 1
    return idx


def parse_raw_fields(content: bytes) -> tuple[Any, dict[str, tuple[Any, str]]]:
    """
    ``json.loads`` that also returns, for a JSON object, each top-level field's
    value along with its source text. It costs no more than ``json.loads``:
    the fields are parsed by the same scanner, one at a time.
    """
    s = content.decode(json.detect_encoding(content), "surrogatepass")
    idx = _skip_ws(s, 0)
    if not s.startswith("{", idx):
        return json.loads(s), {}

    obj: dict[str, Any] = {}
    fields: dict[str, tuple[Any, str]] = {}
    idx = _skip_ws(s, idx + 1)
    if s.startswith("}", idx):
        idx += 1
    else:
        while True:
            if not s.startswith('"', idx):
                raise json.JSONDecodeError(
                    "Expecting property name enclosed in double quotes", s, idx
                )
            key, idx = _scan_once(s, idx)
            idx = _skip_ws(s, idx)
            if not s.startswith(":", idx):
                raise json.JSONDecodeError("Expecting ':' delimiter", s, idx)
            start = _skip_ws(s, idx + 1)
            try:
                value, idx = _scan_once(s, start)
            except StopIteration as e:
                raise json.JSONDecodeError(
                    "Expecting value", s, e.value
                ) from None
            obj[key] = value
            fields[key] = (value, s[start:idx])
            idx = _skip_ws(s, idx)
            if s.startswith(",", idx):
                idx = _skip_ws(s, idx + 1)
            elif s.startswith("}", idx):
                idx += 1
                break
            else:
                raise json.JSONDecodeError("Expecting ',' delimiter", s, idx)

    if _skip_ws(s, idx) != len(s):
        raise json.JSONDecodeError("Extra data", s, idx)
    return obj, fields


class RawJSON:
    """Source text of a JSON value, to be written out as is."""

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text

    def __deepcopy__(self, memo: dict) -> "RawJSON":
        # Immutable: the clients deep-copy the body they adapt.
        return self


_dumps = functools.partial(
    json.dumps, ensure_ascii=False, separators=(",", ":"), allow_nan=False
)


def dumps_with_raw(body: Any) -> bytes:
    """The SDK's JSON encoding of a dict whose ``RawJSON`` fields go verbatim."""
    return (
        "{"
        + ",".join(
            f"{_dumps(k)}:{v.text if isinstance(v, RawJSON) else _dumps(v)}"
            for k, v in body.items()
        )
        + "}"
    ).encode()


def is_streaming_request(
    body: dict | None, endpoint: MessagesAPIEndpoint
) -> bool:
    """Whether the request asks for a streamed response.

    Checking the response ``Content-Type`` header is not enough: the Bedrock
    backend returns the stream in its own event stream format
    (``application/vnd.amazon.eventstream``) rather than ``text/event-stream``.
    """
    if endpoint is MessagesAPIEndpoint.POST_MESSAGES and isinstance(body, dict):
        return bool(body.get("stream"))

    return False


async def bedrock_stream_to_sse(
    iterator: AsyncIterator[bytes],
) -> AsyncIterator[ServerSentEvent]:
    try:
        from botocore.eventstream import EventStreamBuffer
    except ImportError as e:
        raise ImportError(
            "The Bedrock backend requires botocore, which is not installed. "
            "Install it with `pip install aidial-adapter-anthropic[bedrock]`."
        ) from e

    from anthropic.lib.bedrock._stream_decoder import AWSEventStreamDecoder

    decoder = AWSEventStreamDecoder()
    event_stream_buffer = EventStreamBuffer()
    async for chunk in iterator:
        event_stream_buffer.add_data(chunk)
        for event in event_stream_buffer:
            message = decoder._parse_message_from_event(event)
            if message:
                try:
                    event_type = json.loads(message).get("type") or "completion"
                except Exception:
                    event_type = "completion"
                yield ServerSentEvent(data=message, event=event_type)


def _show_sse_event(event: ServerSentEvent) -> str:
    lines: list[str] = []
    if event.id is not None:
        lines.append(f"id: {event.id}\n")
    if event.event is not None:
        lines.append(f"event: {event.event}\n")
    for line in event.data.split("\n"):
        lines.append(f"data: {line}\n")
    if event.retry is not None:
        lines.append(f"retry: {event.retry}\n")
    lines.append("\n")
    return "".join(lines)


async def sse_to_bytes_iterator(
    events: AsyncIterator[ServerSentEvent],
) -> AsyncIterator[bytes]:
    async for event in events:
        yield _show_sse_event(event).encode()


def strip_content_headers(response_headers: httpx.Headers) -> None:
    # The adapter decodes the response body before forwarding it, so the
    # Content-Encoding header no longer applies. Leaving it would cause the
    # downstream client to attempt a second decompression and fail.
    response_headers.pop("Content-Encoding", None)
    # Content-Length reflected the compressed size; after decoding it no longer
    # matches the body, so drop it and let the framework recompute it.
    # And even when the content was uncompressed to begin with,
    # the content length can change do to the SSE reformatting.
    response_headers.pop("Content-Length", None)


# Hop-by-hop headers (RFC 9110 7.6.1) describe the upstream connection, not
# the response this app sends. Relaying "transfer-encoding" is the harmful one:
# the ASGI server computes its own "content-length" for the relayed body, and
# a message carrying both is rejected by a strict HTTP parser (RFC 9112 6.2) -
# DIAL Core being one.
_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

# The ASGI server prepends its own "server" and "date" to every response.
# Both are singleton fields (RFC 9110 5.6.6), so the upstream's copy must go.
_SERVER_MANAGED_HEADERS = {"server", "date"}

_NON_FORWARDABLE_HEADERS = _HOP_BY_HOP_HEADERS | _SERVER_MANAGED_HEADERS


def strip_non_forwardable_headers(response_headers: httpx.Headers) -> None:
    for header in _NON_FORWARDABLE_HEADERS:
        response_headers.pop(header, None)


def build_request_headers(headers: StarletteHeaders) -> dict[str, str]:
    def _keep_header(header: str) -> bool:
        header = header.lower()
        return header.startswith("anthropic-") or header == "accept-encoding"

    return {k.lower(): v for (k, v) in headers.items() if _keep_header(k)}
