"""Anthropic API passthrough proxy.

Builds a FastAPI application that transparently forwards a fixed set of
Anthropic Messages API endpoints to an upstream Anthropic client. Unlike a
dumb reverse proxy it goes through the Anthropic SDK client, so it inherits the
client's authentication, retries and — for the legacy Bedrock backend — the
AWS event stream decoding.

The client used for a given request is supplied by the caller via a
``ClientFactory``, which is the sole point of variation between deployments
(platform key, Bedrock credentials, Vertex project, Foundry endpoint, ...).
The adaptation each upstream cloud requires is the package's own business:
see ``_middleware``.
"""

import contextlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from http import HTTPStatus
from typing import TypeVar

import httpx
from anthropic import AsyncAnthropicBedrock
from anthropic._models import FinalRequestOptions
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from aidial_adapter_anthropic.passthrough._caching import (
    get_cache_headers,
    is_message_params,
)
from aidial_adapter_anthropic.passthrough._errors import (
    anthropic_response_decorator,
)
from aidial_adapter_anthropic.passthrough._helpers import (
    MessagesAPIEndpoint,
    bedrock_stream_to_sse,
    build_request_headers,
    is_streaming_request,
    sse_to_bytes_iterator,
    strip_content_headers,
)
from aidial_adapter_anthropic.passthrough._logging import logging_decorator
from aidial_adapter_anthropic.passthrough._middleware import (
    AnthropicClient,
    get_cloud,
    get_cloud_middlewares,
)

_log = logging.getLogger(__name__)

ClientT = TypeVar("ClientT", bound=AnthropicClient)
ClientFactory = Callable[[Request], Awaitable[ClientT]] | ClientT


@anthropic_response_decorator
@logging_decorator
async def _proxy(
    request: Request, endpoint: MessagesAPIEndpoint, client: AnthropicClient
) -> Response:
    json_body = None
    if content := await request.body():
        with contextlib.suppress(json.JSONDecodeError):
            json_body = json.loads(content)

    is_streaming = is_streaming_request(json_body, endpoint)

    headers = build_request_headers(request.headers)

    for middleware in get_cloud_middlewares(get_cloud(client)):
        middleware.on_request(headers, json_body, endpoint)

    if _log.isEnabledFor(logging.DEBUG):
        # Ask the upstream not to compress the response so its body (and
        # streamed chunks) can be logged as-is. Forgoing compression on the
        # upstream hop is a fair price for painless logging.
        headers["accept-encoding"] = "identity"

    options = FinalRequestOptions.construct(
        method=request.method.lower(),
        url=endpoint.value,
        json_data=json_body,
        headers=headers,
    )

    response = await client.request(
        cast_to=httpx.Response,
        options=options,
        stream=is_streaming,
        stream_cls=None,
    )

    if (
        # Only the generating endpoint takes part in the cache affinity, and
        # only on a success: a retriable failure makes DIAL Core try another
        # upstream, whose provider cache is cold.
        endpoint is MessagesAPIEndpoint.MESSAGES
        and response.status_code == HTTPStatus.OK
        and is_message_params(json_body)
    ):
        try:
            response.headers.update(get_cache_headers(json_body))
        except Exception:
            # The cache affinity is an optimization on top of a response the
            # upstream has already produced: a body this logic fails to read
            # costs cache hits, it must never fail the request.
            _log.exception("Failed to compute the DIAL cache headers.")

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
    endpoint: MessagesAPIEndpoint, get_client: ClientFactory[ClientT]
) -> Callable[[Request], Awaitable[Response]]:
    async def handler(request: Request) -> Response:
        client = (
            get_client
            if isinstance(get_client, AnthropicClient)
            else await get_client(request)
        )
        return await _proxy(request, endpoint, client)

    return handler


def create_anthropic_api_app(get_client: ClientFactory[ClientT]) -> FastAPI:
    app = FastAPI()
    for endpoint in MessagesAPIEndpoint:
        app.router.add_api_route(
            path=endpoint.value,
            methods=["POST"],
            endpoint=_create_proxy_handler(endpoint, get_client),
        )
    return app
