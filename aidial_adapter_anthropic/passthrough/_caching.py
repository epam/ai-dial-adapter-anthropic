import os
import time
from collections.abc import Iterable, Iterator, Mapping
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

# Switches restore_cache_marker on (the default) or off
_RESTORE_CACHE_MARKER_ENV = "RESTORE_CACHE_MARKER"

# Anthropic's limit of cache breakpoints per request
_MAX_MARKERS = 4


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


def restore_cache_marker(
    request: MessageCreateParamsBase, headers: Mapping[str, str]
) -> bool:
    """
    Puts `cache_control` back on the block DIAL Core names in
    X-DIAL-CACHE-BREAKPOINT-PATH: the end of the prefix it pinned, where the
    request this one continues had its last marker. Vertex AI serves a request
    from the region holding its cache only when the request carries
    `cache_control` in the same blocks, while clients like Claude Code move
    the marker to the newest message on every turn.

    Only when RESTORE_CACHE_MARKER isn't switched off, the request asks for
    caching after that block and has a marker slot left. Returns whether a
    marker was added.
    """

    path = headers.get(_DIAL_CACHE_BREAKPOINT_PATH)
    enabled = os.getenv(_RESTORE_CACHE_MARKER_ENV, "true").lower()
    if not path or enabled not in ("1", "true"):
        return False

    # Plain JSON from here: the typed params don't index or mix block types.
    body: Any = request
    blocks = list(_prompt_blocks(body))
    at = next((n for n, (p, *_) in enumerate(blocks) if p == path), None)
    if at is None:
        return False

    _, parent, key, idx = blocks[at]
    value = parent[key]
    # A string is the same tokens as one text block: DIAL Core hashes both as
    # content[0].
    block = (
        value[idx]
        if isinstance(value, list)
        else {"type": "text", "text": value}
    )
    if (
        not isinstance(block, dict)
        or "cache_control" in block
        or block.get("type") in ("thinking", "redacted_thinking")
    ):
        return False

    top = body.get("cache_control")
    after = _markers(
        [
            p[k][i] if isinstance(p[k], list) else None
            for _, p, k, i in blocks[at + 1 :]
        ]
    )
    after += [top] if _is_ephemeral(top) else []
    everything = _markers(
        [body.get(f) for f in ("tools", "system", "messages")]
    )
    if not after or len(everything) + _is_ephemeral(top) >= _MAX_MARKERS:
        return False

    # A longer TTL may not follow a shorter one. Every marker before this block
    # has at least the TTL of every marker after it, so the longest TTL after
    # it fits on both sides.
    template = max(after, key=lambda c: parse_ttl_sec(c.get("ttl")))
    block["cache_control"] = dict(template)
    if not isinstance(value, list):
        parent[key] = [block]
    return True


def _prompt_blocks(body: Any) -> Iterator[tuple[str, Any, str, int]]:
    """
    (path, parent, key, index) of every block in the order of the prompt, with
    the paths of DIAL Core: a scalar where an array is expected is content[0].
    """
    sections = [("prefix.body.tools", body, "tools")]
    sections.append(("prefix.body.system", body, "system"))
    messages = body.get("messages")
    for i, message in enumerate(messages if isinstance(messages, list) else []):
        if isinstance(message, dict):
            sections.append(
                (f"prefix.body.messages[{i}].content", message, "content")
            )
    for prefix, parent, key in sections:
        value = parent.get(key)
        count = (
            len(value) if isinstance(value, list) else int(value is not None)
        )
        for i in range(count):
            yield f"{prefix}[{i}]", parent, key, i


def _markers(value: Any) -> list[BetaCacheControlEphemeralParam]:
    """
    The markers at any depth, e.g. also on a block inside a tool result: all of
    them count toward Anthropic's limit. Counting one too many only skips a
    restore; one too few would get the request rejected.
    """
    found: list[BetaCacheControlEphemeralParam] = []
    stack = [value]
    while stack:
        v = stack.pop()
        if isinstance(v, dict):
            if _is_ephemeral(control := v.get("cache_control")):
                found.append(control)
            stack.extend(v.values())
        elif isinstance(v, list):
            stack.extend(v)
    return found
