"""Exception handling for the Anthropic API passthrough.

``error_response_decorator`` wraps the proxy handler and turns any failure into
an Anthropic-style error response, so the downstream client sees the same error
schema it would get talking to the Anthropic API directly:

    {"type": "error", "error": {"type": "<error type>", "message": "..."}}

An upstream ``APIStatusError`` is relayed verbatim (original status, body and
rate-limit headers); every other exception is mapped to the closest Anthropic
error type. See https://platform.claude.com/docs/en/api/errors.
"""

import logging
from collections.abc import Awaitable, Callable
from functools import wraps

import anthropic
import fastapi
from aidial_sdk.exceptions import HTTPException as DialException
from fastapi.responses import JSONResponse

_log = logging.getLogger(__name__)

_Handler = Callable[..., Awaitable[fastapi.Response]]

# https://platform.claude.com/docs/en/api/errors#http-errors
_ANTHROPIC_ERROR_TYPES: dict[int, str] = {
    400: "invalid_request_error",
    401: "authentication_error",
    402: "billing_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    500: "api_error",
    504: "timeout_error",
    529: "overloaded_error",
}


def _anthropic_error_type(status_code: int) -> str:
    if error_type := _ANTHROPIC_ERROR_TYPES.get(status_code):
        return error_type
    return "invalid_request_error" if 400 <= status_code < 500 else "api_error"


def _copy_anthropic_headers(
    e: anthropic.APIStatusError,
) -> dict[str, str] | None:
    headers = e.response.headers
    # Preserve the rate-limit hints so the client can back off correctly:
    # https://platform.claude.com/docs/en/api/rate-limits#tier-1
    copied = {
        key: value
        for key in ("Retry-After", "Retry-After-Ms")
        if (value := headers.get(key)) is not None
    }
    return copied or None


def _error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "type": "error",
            "error": {
                "type": _anthropic_error_type(status_code),
                "message": message,
            },
        },
    )


def _classify_exception(e: Exception) -> tuple[int, str]:
    if isinstance(e, DialException):
        return e.status_code, e.message

    if (
        isinstance(e, anthropic.AnthropicError)
        and not isinstance(e, anthropic.APIError)
        and "not supported" in str(e).lower()
    ):
        # The SDK raises a bare AnthropicError (never an APIError, which covers
        # every HTTP and connection failure) with a "not supported" message when
        # the selected backend has no such endpoint — e.g. Bedrock has no
        # token-counting or batches route. That is a missing resource (404), not
        # a server fault. Any other bare AnthropicError (missing credentials,
        # misconfiguration, ...) stays a 500.
        return 404, str(e)

    return 500, str(e)


def anthropic_response_decorator(func: _Handler) -> _Handler:
    @wraps(func)
    async def wrapper(*args, **kwargs) -> fastapi.Response:
        try:
            return await func(*args, **kwargs)
        except anthropic.APIStatusError as e:
            return fastapi.Response(
                content=e.response.content,
                status_code=e.status_code,
                headers=_copy_anthropic_headers(e),
            )
        except Exception as e:
            status_code, message = _classify_exception(e)
            _log.exception(
                f"Caught exception "
                f"{type(e).__module__}.{type(e).__name__}: {e!r}. "
                f"Returning a {status_code} Anthropic error response."
            )
            return _error_response(status_code, message)

    return wrapper
