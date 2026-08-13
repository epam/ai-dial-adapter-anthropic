import time
from collections.abc import Iterable, Iterator
from typing import Any, Self, TypeGuard

from anthropic.types.beta import (
    BetaCacheControlEphemeralParam,
    BetaContentBlockParam,
    BetaTextBlockParam,
    BetaToolUnionParam,
)
from anthropic.types.beta.message_create_params import MessageCreateParamsBase

from aidial_adapter_anthropic._utils.cache import parse_ttl_sec

_DIAL_CACHE_BREAKPOINT_PATH = "X-DIAL-CACHE-BREAKPOINT-PATH"
_DIAL_CACHE_EXPIRE_AT = "X-DIAL-CACHE-EXPIRE-AT"


class CacheBreakpointPath:
    """A position in the Anthropic request body DIAL Core is able to address."""

    path: str

    def __init__(self, path: str) -> None:
        self.path = path

    @classmethod
    def tools(cls, idx: int) -> Self:
        return cls(f"prefix.body.tools[{idx}]")

    @classmethod
    def system(cls, idx: int) -> Self:
        return cls(f"prefix.body.system[{idx}]")

    @classmethod
    def messages(cls, idx: int, block_idx: int) -> Self:
        return cls(f"prefix.body.messages[{idx}].content[{block_idx}]")


def is_message_params(body: Any) -> TypeGuard[MessageCreateParamsBase]:
    return isinstance(body, dict)


_Block = (
    BetaToolUnionParam  # tools
    | BetaTextBlockParam  # system content
    | BetaContentBlockParam  # messages
)


def _is_ephemeral(value: Any) -> TypeGuard[BetaCacheControlEphemeralParam]:
    return isinstance(value, dict) and value.get("type") == "ephemeral"


def _get_cache_control(block: _Block) -> BetaCacheControlEphemeralParam | None:
    if not isinstance(block, dict):
        return None

    cache_control = block.get("cache_control")
    return cache_control if _is_ephemeral(cache_control) else None


def _get_cache_controls(
    blocks: str | Iterable[_Block] | None,
) -> Iterator[tuple[int, BetaCacheControlEphemeralParam | None]]:
    if isinstance(blocks, list):
        for i, block in enumerate(blocks):
            yield i, _get_cache_control(block)
    elif blocks is not None:
        # A scalar where an array is expected counts as a one-element array,
        # the way DIAL Core hashes it: `"system": "be helpful"` is `system[0]`,
        # `"content": "hi"` is `content[0]`. In a valid request that scalar is a
        # string, which carries no marker.
        yield 0, None


_Candidate = tuple[CacheBreakpointPath, BetaCacheControlEphemeralParam | None]


def _get_cache_control_paths(
    request: MessageCreateParamsBase,
) -> Iterator[_Candidate]:
    for i, control in _get_cache_controls(request.get("tools")):
        yield CacheBreakpointPath.tools(i), control

    for i, control in _get_cache_controls(request.get("system")):
        yield CacheBreakpointPath.system(i), control

    # Unlike tools and system, a non-array `messages` yields no candidate at
    # all: DIAL Core doesn't scalar-normalize it either.
    for i, message in enumerate(request.get("messages")):
        content = message.get("content") if isinstance(message, dict) else None
        for j, control in _get_cache_controls(content):
            yield CacheBreakpointPath.messages(i, j), control


def get_cache_headers(request: MessageCreateParamsBase) -> dict[str, str]:
    """
    The cache affinity headers to report for a successful response.
    Empty when the request carries no cache breakpoint at all.
    """

    ttl = 0
    breakpoint_path = None
    paths = list(_get_cache_control_paths(request))

    for path, control in paths:
        if control is not None:
            ttl = max(ttl, parse_ttl_sec(control.get("ttl")))
            breakpoint_path = path

    if paths and (control := request.get("cache_control")):
        # Automatic caching leaves the placement of the breakpoint to Anthropic,
        # which caches the longest cacheable prefix - so the last candidate
        # block is the one to report.
        ttl = max(ttl, parse_ttl_sec(control.get("ttl")))
        breakpoint_path = paths[-1][0]

    if breakpoint_path is None:
        return {}

    return {
        _DIAL_CACHE_BREAKPOINT_PATH: breakpoint_path.path,
        _DIAL_CACHE_EXPIRE_AT: str(int(time.time()) + ttl),
    }
