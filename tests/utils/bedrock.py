"""Helpers for mocking streaming responses from the AWS Bedrock upstream."""

import base64
import binascii
import json
import struct
from collections.abc import Iterable

_STRING_HEADER_TYPE = 7  # AWS event stream header value type for UTF-8 strings
AMAZON_SSE_CONTENT_TYPE = "application/vnd.amazon.eventstream"


def _encode_header(name: str, value: str) -> bytes:
    name_b = name.encode()
    value_b = value.encode()
    return (
        struct.pack("B", len(name_b))
        + name_b
        + struct.pack("B", _STRING_HEADER_TYPE)
        + struct.pack(">H", len(value_b))
        + value_b
    )


def _encode_message(headers: dict[str, str], payload: bytes) -> bytes:
    """Encode a single AWS event stream frame.

    Frame layout (all integers big-endian): prelude (total length, headers
    length, prelude CRC32), the headers, the payload, then a trailing message
    CRC32. Each CRC covers all the bytes preceding it.
    """
    headers_b = b"".join(_encode_header(k, v) for k, v in headers.items())
    total_len = 4 + 4 + 4 + len(headers_b) + len(payload) + 4
    prelude = struct.pack(">II", total_len, len(headers_b))
    prelude += struct.pack(">I", binascii.crc32(prelude) & 0xFFFFFFFF)
    body = prelude + headers_b + payload
    return body + struct.pack(">I", binascii.crc32(body) & 0xFFFFFFFF)


def _encode_chunk(message: dict) -> bytes:
    """Encode one Anthropic streaming message as a Bedrock ``chunk`` event."""
    inner = json.dumps(message).encode()
    payload = json.dumps({"bytes": base64.b64encode(inner).decode()}).encode()
    return _encode_message(
        {
            ":event-type": "chunk",
            ":content-type": "application/json",
            ":message-type": "event",
        },
        payload,
    )


def encode_event_stream(messages: Iterable[dict]) -> bytes:
    """Encode Anthropic streaming messages as a Bedrock AWS event stream body.

    The returned bytes are what a respx mock should serve (with
    ``EVENT_STREAM_CONTENT_TYPE``) to emulate a legacy Bedrock streamed
    response.
    """
    return b"".join(_encode_chunk(message) for message in messages)


def _sse_chunk_to_message(chunk: bytes) -> dict:
    """Recover the Anthropic message carried by an SSE event chunk."""
    data = "\n".join(
        line.removeprefix("data:").strip()
        for line in chunk.decode().splitlines()
        if line.startswith("data:")
    )
    return json.loads(data)


def sse_to_event_stream(sse_chunks: Iterable[bytes]) -> bytes:
    """Re-encode SSE event chunks as a Bedrock AWS event stream body.

    The legacy Bedrock runtime speaks the AWS event stream format rather than
    SSE, so a mock serving it must transcode the canonical SSE chunks.
    """
    return encode_event_stream(
        _sse_chunk_to_message(chunk) for chunk in sse_chunks
    )
