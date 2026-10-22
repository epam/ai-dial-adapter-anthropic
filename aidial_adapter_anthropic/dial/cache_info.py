import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Self

from aidial_sdk.chat_completion import CacheBreakpointPath

from aidial_adapter_anthropic._utils.cache import CacheBreakpoint
from aidial_adapter_anthropic.dial.request import AdapterRequest
from aidial_adapter_anthropic.dial.tools import ToolDefinition


@dataclass
class DialCacheInfo:
    """The cache information reported back to DIAL Core."""

    breakpoint_path: CacheBreakpointPath
    expired_at: str

    @classmethod
    def create(cls, request: AdapterRequest) -> Self | None:
        if (path := _breakpoint_path(request)) is None:
            return None

        ttl_sec = max(breakpoint.ttl_sec for breakpoint in _all(request))
        return cls(path, str(int(time.time()) + ttl_sec))


def _tools(request: AdapterRequest) -> list[ToolDefinition]:
    return request.tool_config.tools if request.tool_config else []


def _all(request: AdapterRequest) -> Iterator[CacheBreakpoint]:
    if request.cache_breakpoint is not None:
        yield request.cache_breakpoint
    for message in request.messages.raw_list:
        yield from message.cache_breakpoints.all()
    for tool in _tools(request):
        if tool.cache_breakpoint is not None:
            yield tool.cache_breakpoint


def _breakpoint_path(request: AdapterRequest) -> CacheBreakpointPath | None:
    """The last position of the request body that is expected to be cached."""

    # The paths are reported to DIAL Core, so they must address the messages
    # of the original request, not the preprocessed ones.
    messages = request.messages.lst

    if request.cache_breakpoint is not None and messages:
        return CacheBreakpointPath.messages(max(messages[-1][1]))

    for message, indices in reversed(messages):
        if message.cache_breakpoints:
            return CacheBreakpointPath.messages(max(indices))

    for tool in reversed(_tools(request)):
        if tool.cache_breakpoint is not None:
            return CacheBreakpointPath.tools(tool.index)

    return None
