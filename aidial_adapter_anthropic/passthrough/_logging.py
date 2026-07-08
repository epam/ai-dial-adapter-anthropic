"""Debug logging for the Anthropic API passthrough.

``logging_decorator`` wraps the proxy handler and, only when DEBUG logging is
enabled, logs the incoming request body and the outgoing response — streaming
responses are logged chunk by chunk as they are relayed.
"""

import contextlib
import logging
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from functools import wraps
from typing import Concatenate, ParamSpec

from fastapi import Request
from fastapi.responses import Response, StreamingResponse

_log = logging.getLogger(__name__)

_Content = str | bytes | memoryview


def _as_text(data: _Content) -> str:
    if isinstance(data, str):
        return data
    return bytes(data).decode("utf-8", errors="replace")


async def _log_stream_chunks(
    iterator: AsyncIterable[_Content],
) -> AsyncIterator[_Content]:
    async for chunk in iterator:
        with contextlib.suppress(Exception):
            _log.debug(f"response chunk: {_as_text(chunk).rstrip()}")
        yield chunk


# The trailing parameters (the upstream client) are relayed untouched, so the
# decorator stays agnostic of their type and preserves the handler signature.
_P = ParamSpec("_P")

_Handler = Callable[Concatenate[Request, str, _P], Awaitable[Response]]


def logging_decorator(func: _Handler[_P]) -> _Handler[_P]:
    def one_line(text: str) -> str:
        return "".join(text.splitlines())

    @wraps(func)
    async def wrapper(
        request: Request, path: str, *args: _P.args, **kwargs: _P.kwargs
    ) -> Response:
        if not _log.isEnabledFor(logging.DEBUG):
            return await func(request, path, *args, **kwargs)

        with contextlib.suppress(Exception):
            _log.debug(f"request: {one_line(_as_text(await request.body()))}")

        response = await func(request, path, *args, **kwargs)

        if isinstance(response, StreamingResponse):
            response.body_iterator = _log_stream_chunks(response.body_iterator)
        else:
            with contextlib.suppress(Exception):
                _log.debug(f"response: {one_line(_as_text(response.body))}")

        return response

    return wrapper
