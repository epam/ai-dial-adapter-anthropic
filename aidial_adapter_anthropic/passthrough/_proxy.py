"""Anthropic API passthrough proxy.

Builds a FastAPI application that transparently forwards a fixed set of
Anthropic Messages API endpoints to an upstream Anthropic client. Unlike a
dumb reverse proxy it goes through the Anthropic SDK client, so it inherits the
client's authentication, retries and — for the legacy Bedrock backend — the
AWS event stream decoding.

The client used for a given request is supplied by the caller via a
``ClientFactory``, which is the sole point of variation between deployments
(platform key, Bedrock credentials, Vertex project, Foundry endpoint, ...).
"""

import contextlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable

import httpx
from anthropic import (
    AsyncAnthropic,
    AsyncAnthropicBedrock,
    AsyncAnthropicBedrockMantle,
    AsyncAnthropicFoundry,
    AsyncAnthropicVertex,
)
from anthropic._models import FinalRequestOptions
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from aidial_adapter_anthropic.passthrough._errors import (
    anthropic_response_decorator,
)
from aidial_adapter_anthropic.passthrough._helpers import (
    bedrock_stream_to_sse,
    build_request_headers,
    is_streaming_request,
    sse_to_bytes_iterator,
    strip_content_headers,
)
from aidial_adapter_anthropic.passthrough._logging import logging_decorator

_log = logging.getLogger(__name__)

AnthropicClient = (
    AsyncAnthropic
    | AsyncAnthropicBedrock
    | AsyncAnthropicBedrockMantle
    | AsyncAnthropicVertex
    | AsyncAnthropicFoundry
)

ClientFactory = (
    Callable[[Request], Awaitable[AnthropicClient]] | AnthropicClient
)


@anthropic_response_decorator
@logging_decorator
async def _proxy(
    request: Request, path: str, client: AnthropicClient
) -> Response:
    json_body = None
    if content := await request.body():
        with contextlib.suppress(json.JSONDecodeError):
            json_body = json.loads(content)

    is_streaming = is_streaming_request(json_body, path)

    is_bedrock = isinstance(
        client, (AsyncAnthropicBedrock | AsyncAnthropicBedrockMantle)
    )
    headers = build_request_headers(request.headers, is_bedrock=is_bedrock)
    if _log.isEnabledFor(logging.DEBUG):
        # Ask the upstream not to compress the response so its body (and
        # streamed chunks) can be logged as-is. Forgoing compression on the
        # upstream hop is a fair price for painless logging.
        headers["accept-encoding"] = "identity"

    options = FinalRequestOptions.construct(
        method=request.method.lower(),
        url=path,
        json_data=json_body,
        headers=headers,
    )

    response = await client.request(
        cast_to=httpx.Response,
        options=options,
        stream=is_streaming,
        stream_cls=None,
    )

    if is_streaming:

        async def _stream() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()

        stream = _stream()
        if isinstance(client, AsyncAnthropicBedrock):
            # The legacy Bedrock backend streams in the AWS event stream format
            # rather than SSE, so it must be decoded and re-encoded as SSE.
            response.headers["Content-Type"] = "text/event-stream"
            strip_content_headers(response.headers)
            stream = sse_to_bytes_iterator(bedrock_stream_to_sse(stream))

        return StreamingResponse(
            content=stream,
            status_code=response.status_code,
            headers=response.headers,
        )

    else:
        content = await response.aread()
        strip_content_headers(response.headers)
        return Response(
            content=content,
            status_code=response.status_code,
            headers=response.headers,
        )


def _create_proxy_handler(
    path: str, get_client: ClientFactory
) -> Callable[[Request], Awaitable[Response]]:
    async def handler(request: Request) -> Response:
        client = (
            get_client
            if isinstance(get_client, AnthropicClient)
            else await get_client(request)
        )
        return await _proxy(request, path, client)

    return handler


_PROXIED_ENDPOINTS = [
    ("POST", "/v1/messages"),
    ("POST", "/v1/messages/batches"),
    ("POST", "/v1/messages/count_tokens"),
]


def create_anthropic_api_app(get_client: ClientFactory) -> FastAPI:
    app = FastAPI()
    for method, path in _PROXIED_ENDPOINTS:
        app.router.add_api_route(
            path=path,
            methods=[method],
            endpoint=_create_proxy_handler(path, get_client),
        )
    return app
