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
from typing import Any, TypeVar

import httpx
from anthropic import AsyncAnthropicBedrock
from anthropic._models import FinalRequestOptions
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from aidial_adapter_anthropic.passthrough._caching import get_cache_headers
from aidial_adapter_anthropic.passthrough._errors import (
    anthropic_response_decorator,
)
from aidial_adapter_anthropic.passthrough._helpers import (
    MessagesAPIEndpoint,
    RawJSON,
    bedrock_stream_to_sse,
    build_request_headers,
    dumps_with_raw,
    is_streaming_request,
    parse_raw_fields,
    sse_to_bytes_iterator,
    strip_content_headers,
    strip_non_forwardable_headers,
    typecast_request_body,
)
from aidial_adapter_anthropic.passthrough._logging import logging_decorator
from aidial_adapter_anthropic.passthrough._middleware import (
    AnthropicClient,
    apply_middlewares,
)

_log = logging.getLogger(__name__)

ClientT = TypeVar("ClientT", bound=AnthropicClient)
ClientFactory = Callable[[Request], Awaitable[ClientT]] | ClientT


async def _with_raw_body(
    client: AnthropicClient,
    options: FinalRequestOptions,
    headers: dict[str, str],
    raw_fields: dict[str, tuple[Any, str]],
) -> FinalRequestOptions:
    """
    Sends the body with the top-level containers left as received in their
    source text: re-encoding a large prompt costs more CPU than the rest of the
    request. The client still adapts the body to its cloud (Vertex and Bedrock
    move "model" into the URL), but only ever sees the containers as opaque
    stand-ins, which are cheap to copy.
    """
    assert isinstance(options.json_data, dict)
    shell = {
        k: RawJSON(raw[1])
        if (raw := raw_fields.get(k)) is not None
        and v is raw[0]
        and isinstance(v, dict | list)
        else v
        for k, v in options.json_data.items()
    }
    prepared = await client._prepare_options(
        FinalRequestOptions.construct(
            method=options.method,
            url=options.url,
            json_data=shell,
            headers=headers,
        )
    )
    # The client prepares the options once more on sending: a rewritten URL and
    # a body that isn't a dict make that second pass a no-op for the body. The
    # original headers are kept so auth is added by that pass, on every retry.
    return FinalRequestOptions.construct(
        method=prepared.method,
        url=prepared.url,
        content=dumps_with_raw(prepared.json_data),
        headers=headers,
    )


@anthropic_response_decorator
@logging_decorator
async def _proxy(
    request: Request, endpoint: MessagesAPIEndpoint, client: AnthropicClient
) -> Response:
    json_body = None
    raw_fields: dict[str, tuple[Any, str]] = {}
    if content := await request.body():
        with contextlib.suppress(json.JSONDecodeError):
            json_body, raw_fields = parse_raw_fields(content)

    is_streaming = is_streaming_request(json_body, endpoint)

    method, path = endpoint.to_endpoints()
    headers = build_request_headers(request.headers)

    apply_middlewares(client, headers, json_body, endpoint)

    if _log.isEnabledFor(logging.DEBUG):
        # Ask the upstream not to compress the response so its body (and
        # streamed chunks) can be logged as-is. Forgoing compression on the
        # upstream hop is a fair price for painless logging.
        headers["accept-encoding"] = "identity"

    options = FinalRequestOptions.construct(
        method=method.lower(),
        url=path,
        json_data=json_body,
        headers=headers,
    )

    # The middlewares only ever replace a top-level field of these endpoints,
    # unlike the batches one, whose tools sit inside "requests".
    if raw_fields and endpoint is not MessagesAPIEndpoint.POST_BATCHES:
        options = await _with_raw_body(client, options, headers, raw_fields)

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
        endpoint is MessagesAPIEndpoint.POST_MESSAGES
        and response.status_code == HTTPStatus.OK
        and typecast_request_body(json_body, endpoint)
    ):
        try:
            response.headers.update(get_cache_headers(json_body))
        except Exception:
            # The cache affinity is an optimization on top of a response the
            # upstream has already produced: a body this logic fails to read
            # costs cache hits, it must never fail the request.
            _log.exception("Failed to compute the DIAL cache headers.")

    strip_non_forwardable_headers(response.headers)

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
        method, path = endpoint.to_endpoints()
        app.router.add_api_route(
            path=path,
            methods=[method],
            endpoint=_create_proxy_handler(endpoint, get_client),
        )
    return app
