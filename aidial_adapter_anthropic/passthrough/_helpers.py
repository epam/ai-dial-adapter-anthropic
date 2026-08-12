"""Low-level helpers for the Anthropic API passthrough.

These functions perform the mechanical manipulations required to forward a
request to an upstream Anthropic client and relay its response back: header
filtering, SSE (re)formatting, response-body decoding and the Bedrock event
stream conversion.
"""

import json
from collections.abc import AsyncIterator, Callable

import httpx
from anthropic._streaming import ServerSentEvent
from starlette.datastructures import Headers as StarletteHeaders

# The only generating endpoint: the one that streams and takes part in the DIAL
# upstream cache affinity.
MESSAGES_PATH = "/v1/messages"


def is_streaming_request(body: dict | None, path: str) -> bool:
    """Whether the request asks for a streamed response.

    Checking the response ``Content-Type`` header is not enough: the Bedrock
    backend returns the stream in its own event stream format
    (``application/vnd.amazon.eventstream``) rather than ``text/event-stream``.
    """
    if path == MESSAGES_PATH and isinstance(body, dict):
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


def build_request_headers(headers: StarletteHeaders) -> dict[str, str]:
    def _keep_header(header: str) -> bool:
        header = header.lower()
        return header.startswith("anthropic-") or header == "accept-encoding"

    return {k.lower(): v for (k, v) in headers.items() if _keep_header(k)}


def apply_anthropic_beta_features(
    headers: dict[str, str],
    transform: Callable[[list[str]], list[str]],
) -> None:
    raw = headers.get("anthropic-beta")
    features = [f for f in raw.split(",") if f] if raw else []
    features = transform(features)
    if features:
        headers["anthropic-beta"] = ",".join(features)
    else:
        headers.pop("anthropic-beta", None)
